#!/usr/bin/env python3
"""CALIBRI S177: REAL CPU J7 shadow audit, never promote unvalidated consensus.

Run the actual frozen S176 full 456+128+protected64+novel64 J7 pipeline.
Intercept the ORIGINAL S160 selection after the existing single prefill and
score 5/10 source-bound proposals, but return the ORIGINAL selector unchanged.
This measures whether consensus proposes anything new, CPU overhead and bpp.
No true/expected labels ever enter the selector; no model weights are changed.
Do not activate the alternative fastpath before future protected J7 approval.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import torch

from calibri2.s176 import s174_legacy_compatible_j7 as s176
from calibri2.s168 import novel_train_frames_generalization_cpu as s168
from calibri2.s177.adaptive_consensus import adaptive_consensus

ROOT = Path("research/calibri2/s177")
DENOM = 43_261_760
BASE_SELECTOR = s168.ORIGINAL_SELECTOR
TRACE = []


def s177_shadow_select(head, hidden, ids, tok):
    """ONE existing S142 model prefill; second SMALL head evaluation only."""
    frozen = BASE_SELECTOR(head, hidden, ids, tok)
    question = s168.OWNER_PROMPT
    if question is None:
        raise AssertionError("S177 requires literal prompt capture")
    owner_phrase = s176.owner_value_compatible(question)
    owner_ids = tok.encode(owner_phrase, add_special_tokens=False) if owner_phrase else None
    # Fast skip when no independently verifiable source-owner claim exists.
    # The frozen legacy route remains unchanged regardless of recommendation.
    if owner_ids is None or len(owner_ids) != 2:
        TRACE.append({"reason": "no_verified_owner", "candidate": None,
                      "legacy": frozen, "branches": 0, "changes": False})
        return frozen
    with torch.inference_mode():
        canonical = s168.s160.normalized_candidate_spans(tok, ids[0].tolist())
        out = head(hidden.unsqueeze(0).float(), ids)
        logits = out["span_logits"][0].detach().to("cpu")
        decision = adaptive_consensus(
            logits, canonical, question=question, verified_owner_ids=owner_ids,
            legacy_choice=frozen, eos_id=tok.eos_token_id)
    TRACE.append({
        "prompt_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "reason": decision.reason,
        "candidate": list(decision.proposed) if decision.proposed else None,
        "legacy": list(decision.legacy) if decision.legacy else None,
        "branches": decision.branch_count,
        "votes": decision.votes,
        "agreement": decision.agreement,
        "unique_source_spans": decision.unique_span_count,
        "changes": decision.changes_legacy,
    })
    return frozen  # CRITICAL: research-only shadow, no promotion whatsoever.


def main():
    # S168's ORIGINAL_SELECTOR is the hook actually called inside the frozen
    # S160 cached CPU decode. Existing S170/S172/S174/S176 guards are retained.
    s168.ORIGINAL_SELECTOR = s177_shadow_select
    s176.main()
    out = Path(sys.argv[-1])
    report = json.loads((out / "S176_FINAL_SOURCE_COMPAT_REAL_CPU_J7_GATE.json").read_text())
    sources = [ROOT / "adaptive_consensus.py", ROOT / "s177_shadow_real_j7.py"]
    extra_bpp = 8 * sum(p.stat().st_size for p in sources) / DENOM
    bpp = report["bpp_all_sources"] + extra_bpp
    ratio = report["cpu_ratio"]  # computed during real CPU J7 WITH shadow work
    reason_counts = dict(sorted(collections.Counter(r["reason"] for r in TRACE).items()))
    proposal_changes = sum(bool(r["changes"]) for r in TRACE)
    accepted = bool(
        report["safe"] and report["old_full"] >= 428 and report["new_full"] >= 128
        and report["protected_probe_64"] == 64 and report["novel_full"] == 64
        and report["source_only_negatives_abstained"] == 64
        and bpp <= 0.7 and ratio >= 0.9)
    summary = {
        "stage": "S177_ADAPTIVE_PARALLEL_5_10_SHADOW_REAL_J7",
        "source_run_S176": 43302404878,
        "full_CPU_J7_S176_unchanged": True,
        "base_original_full_EOS": report["old_full"],
        "base_new_full_EOS": report["new_full"],
        "base_protected64_full_EOS": report["protected_probe_64"],
        "base_novel64_full_EOS": report["novel_full"],
        "total_shadow_selector_calls": len(TRACE),
        "would_change_original_S160_selector_only": proposal_changes,
        "does_NOT_prove_net_S176_or_accuracy_gain": True,
        "branch_reason_counts": reason_counts,
        "all_in_bpp_conservative": bpp,
        "native_CPU_ratio_including_shadow": ratio,
        "safe_shadows_only": accepted,
        "consensus_promotion_ENABLED": False,
        "model_accepted": False,
        "no_training_or_weight_changes": True,
        "no_expected_or_gold_at_inference": True,
        "no_QA_TEST": True,
        "no_Jessica_LIVE_Render_Supabase_APK_changes": True,
    }
    out.mkdir(exist_ok=True, parents=True)
    (out / "S177_SHADOW_VOTES_REAL_CPU_J7.json").write_text(
        json.dumps({"summary": summary, "cases": TRACE}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
    if not accepted:
        raise AssertionError("S177 shadow rejected: real J7 retention/bpp/CPU gate")

if __name__ == "__main__":
    main()
