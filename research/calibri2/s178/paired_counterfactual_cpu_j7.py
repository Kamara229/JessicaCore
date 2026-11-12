#!/usr/bin/env python3
"""S178: exact paired S176-vs-consensus counterfactual in research-only CPU J7.

For each S93 batch, run the existing frozen S176 inference, and then run the
same frozen model with the S177 5/10 proposal under ALL original S176 guards.
Only after BOTH generations do we look at precommitted TRAIN expected tokens.
No answer-key routing, no QA/TEST, no training or LIVE writes, no promotion.
Never confuse 36 S177 changed *S160 selectors* with 36 changed final answers.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import torch

from calibri2.s176 import s174_legacy_compatible_j7 as s176
from calibri2.s168 import novel_train_frames_generalization_cpu as s168
from calibri2.s177.adaptive_consensus import adaptive_consensus

ROOT = Path("research/calibri2/s178")
DENOM = 43_261_760
ORIGINAL_SLOT_SELECTOR = s168.ORIGINAL_SELECTOR
ORIGINAL_AUDIT = s168.s160.audit_cpu_outputs
AUDIT_COUNTER = 0
RECORDED: list[dict] = []
ACTIVE_TRACES: list[dict] = []


def _sha(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _safe_counterfactual_selector(head, hidden, ids, tok):
    """A candidate may ONLY arise from original S160 + verified prompt source.

    Never uses gold/expected labels or a sample identity. S170/S172/S174/S176
    guard hierarchy still applies AFTER this selector returns. No model copy.
    """
    legacy = ORIGINAL_SLOT_SELECTOR(head, hidden, ids, tok)
    question = s168.OWNER_PROMPT
    if not isinstance(question, str):
        raise AssertionError("S178 must capture literal prompt")
    phrase = s176.owner_value_compatible(question)
    owner_ids = tok.encode(phrase, add_special_tokens=False) if phrase else None
    if owner_ids is None or len(owner_ids) != 2:
        return legacy
    with torch.inference_mode():
        canonical = s168.s160.normalized_candidate_spans(tok, ids[0].tolist())
        span_logits = head(hidden.unsqueeze(0).float(), ids)["span_logits"][0]
        decision = adaptive_consensus(
            span_logits.detach().cpu(), canonical,
            question=question, verified_owner_ids=owner_ids,
            legacy_choice=legacy, eos_id=tok.eos_token_id)
    if decision.changes_legacy:
        ACTIVE_TRACES.append({
            "sha256": _sha(question), "legacy_span": (list(decision.legacy) if decision.legacy is not None else None),
            "candidate_span": list(decision.proposed), "votes": decision.votes,
            "branches": decision.branch_count, "reason": decision.reason,
        })
    return list(decision.proposed) if decision.proposed is not None else legacy


def compare_after_decode(group: str, base: dict, trial: dict, old: dict, new: dict) -> dict:
    """Gold comparison exclusively AFTER BOTH CPU decoders returned."""
    if set(base) != set(trial) or set(base) != set(old) | set(new):
        raise AssertionError("paired audit identity coverage mismatch")
    cases = []
    for key in sorted(base):
        row = old.get(key) or new[key]
        assert (isinstance(row["question"], str) and isinstance(row["expected"], list)
                and len(row["expected"]) >= 2
                and all(isinstance(token, int) for token in row["expected"]))
        gold = row["expected"]
        before, after = base[key], trial[key]
        cases.append({
            "group": group, "train_id": key,
            "prompt_sha256": _sha(row["question"]),
            "base_correct": before == gold,
            "counterfactual_correct": after == gold,
            "base_first_correct": bool(before and before[0] == gold[0]),
            "counterfactual_first_correct": bool(after and after[0] == gold[0]),
            "changed_full_EOS": before != after,
            "baseline_full_EOS_ids": before,
            "counterfactual_full_EOS_ids": after,
            "TRAIN_expected_full_EOS_ids": gold,
        })
    before_right = [c for c in cases if c["base_correct"]]
    after_right = [c for c in cases if c["counterfactual_correct"]]
    lost_full = [c["train_id"] for c in cases if c["base_correct"] and not c["counterfactual_correct"]]
    lost_first = [c["train_id"] for c in cases if c["base_first_correct"] and not c["counterfactual_first_correct"]]
    gained_full = [c["train_id"] for c in cases if not c["base_correct"] and c["counterfactual_correct"]]
    final_changes = [c["train_id"] for c in cases if c["changed_full_EOS"]]
    return {
        "group": group, "number": len(cases),
        "baseline_full_correct": len(before_right),
        "counterfactual_full_correct": len(after_right),
        "newly_correct_full_EOS_ids": gained_full,
        "lost_previous_full_EOS_ids": lost_full,
        "lost_previous_first_token_ids": lost_first,
        "changed_final_EOS_ids": final_changes,
        "all_cases": cases,
    }


def shadow_paired_audit(cpu, old, new, old_ids, new_ids, routes, tok, bank,
                        copy_net, bridge, full, expansion, parent_old,
                        parent_new, head):
    """Two runs with identical FP16 weights, source cohorts and S176 guards."""
    global AUDIT_COUNTER
    group = ("original456plusnew128", "protected64", "novel64")[AUDIT_COUNTER]
    AUDIT_COUNTER += 1
    # Use precisely the same original S176 S160 selector; no added sampling.
    s168.ORIGINAL_SELECTOR = ORIGINAL_SLOT_SELECTOR
    try:
        baseline = ORIGINAL_AUDIT(
            cpu, old, new, old_ids, new_ids, routes, tok, bank, copy_net,
            bridge, full, expansion, parent_old, parent_new, head)
        s168.ORIGINAL_SELECTOR = _safe_counterfactual_selector
        experiment = ORIGINAL_AUDIT(
            cpu, old, new, old_ids, new_ids, routes, tok, bank, copy_net,
            bridge, full, expansion, parent_old, parent_new, head)
    finally:
        # Keep S178 active for the subsequent paired native CPU timing;
        # each group explicitly switches to baseline before control decode.
        s168.ORIGINAL_SELECTOR = _safe_counterfactual_selector
    before = {**baseline[0], **baseline[1]}
    after = {**experiment[0], **experiment[1]}
    audit = compare_after_decode(group, before, after, old, new)
    RECORDED.append(audit)
    # Save even on rejection: a failed strict J7 must still leave evidence.
    destination = Path(sys.argv[-1])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "S178_PAIRED_POSTHOC_GROUPS.json").write_text(
        json.dumps(RECORDED, ensure_ascii=False, sort_keys=True, indent=2))
    return experiment


def main():
    if len(sys.argv) < 2:
        raise AssertionError("S178 requires original S176 pinned inputs")
    out = Path(sys.argv[-1])
    s168.s160.audit_cpu_outputs = shadow_paired_audit
    try:
        # S176 always tests original S142-protected 456+128+64+64 real CPU J7.
        # Candidate is only run in isolated research, never in production.
        s176.main()
    finally:
        s168.s160.audit_cpu_outputs = ORIGINAL_AUDIT
        s168.ORIGINAL_SELECTOR = ORIGINAL_SLOT_SELECTOR
    if len(RECORDED) != 3:
        raise AssertionError(f"S178 missing paired J7 groups {len(RECORDED)}")
    checks = {
        "original456plusnew128": (584, 428 + 128),
        "protected64": (64, 64),
        "novel64": (64, 64),
    }
    per_group = {}
    for record in RECORDED:
        name = record["group"]
        assert name in checks
        required_rows, required_base_full = checks[name]
        if record["number"] != required_rows or record["baseline_full_correct"] != required_base_full:
            raise AssertionError(f"Frozen S176 baseline drift in {name}")
        per_group[name] = {
            k: v for k, v in record.items() if k != "all_cases"
        }
    s176_gate = json.loads(
        (out / "S176_FINAL_SOURCE_COMPAT_REAL_CPU_J7_GATE.json").read_text())
    s177 = json.loads(
        Path("inputs/s178_s177/S177_SHADOW_VOTES_REAL_CPU_J7.json").read_text())
    assert s177["summary"]["safe_shadows_only"]
    assert s177["summary"]["would_change_original_S160_selector_only"] == 36
    prior_changes = collections.Counter(
        x["prompt_sha256"] for x in s177["cases"]
        if x.get("changes") and x.get("prompt_sha256"))
    if sum(prior_changes.values()) != 36:
        raise AssertionError("S177 36 candidate trace checksum changed")
    # S177 trace *includes speed checks*, so it must NOT be misinterpreted as
    # final answer changes or exhaustive one-to-one sample identities.
    all_changed = [x for rec in RECORDED for x in rec["all_cases"]
                   if x["changed_full_EOS"]]
    possible = [x for x in ACTIVE_TRACES if x["sha256"] in prior_changes]
    extra_sources = (
        Path("research/calibri2/s177/adaptive_consensus.py"),
        Path("research/calibri2/s177/s177_shadow_real_j7.py"),
        ROOT / "paired_counterfactual_cpu_j7.py",
    )
    conservative_bpp = (s176_gate["bpp_all_sources"] +
                        8 * sum(p.stat().st_size for p in extra_sources) / DENOM)
    good = all(
        not record["lost_previous_full_EOS_ids"] and
        not record["lost_previous_first_token_ids"]
        for record in RECORDED)
    gained = sum(len(rec["newly_correct_full_EOS_ids"]) for rec in RECORDED)
    physical_ok = conservative_bpp <= 0.7 and s176_gate["cpu_ratio"] >= 0.9
    report = {
        "stage": "CALIBRI_S178_PAIRED_S176_VS_ADAPTIVE_5_10_REAL_CPU_J7",
        "source_run_S177": 43312279029,
        "source_S177_selector_changes_NOT_final_outputs": 36,
        "prior_selector_changed_distinct_prompt_hashes": len(prior_changes),
        "original_S177_proposals_seen": len(possible),
        "active_selector_changes_in_J7_and_speed": len(ACTIVE_TRACES),
        "changed_final_full_EOS": len(all_changed),
        "gained_new_correct_full_EOS": gained,
        "cohorts": per_group,
        "all_in_bpp_conservative": conservative_bpp,
        "real_native_cpu_ratio": s176_gate["cpu_ratio"],
        "strict_no_forgetting_vs_actual_S176": good,
        "physical_limit_passed": physical_ok,
        "research_candidate_safe": good and physical_ok and s176_gate["safe"],
        "model_accepted": False,
        "no_training_or_weight_changes": True,
        "original_S176_decoder_preserved_as_control": True,
        "PROMPT_only_at_inference": True,
        "TRAIN_expected_only_after_both_generations": True,
        "no_QA_TEST": True,
        "LIVE_Jessica_Render_Supabase_APK_unchanged": True,
    }
    (out / "S178_FINAL_PAIRED_COUNTERFACTUAL_J7_GATE.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if not report["research_candidate_safe"]:
        raise AssertionError("S178 rejected: forgetting or physical limit")
    # A net gain is NOT presumed. A valid no-gain result is a diagnostic only.


if __name__ == "__main__":
    main()
