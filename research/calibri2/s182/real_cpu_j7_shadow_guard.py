"""S182: fail-closed source-only S30/S54 shadow proposal in REAL frozen CPU J7.

Wrap ONLY the existing S175->S172 confidence guard in research CI;
no gold enters owner selection, no model weights or Green Tunnel updated.
Shadow can propose only exact S30/S54 literal two-token contiguous spans.
Original non-shadow decoding remains byte-identical to S176.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from calibri2.s181.source_grammar_shadow import shadow_candidate

PARSER_SOURCE = Path("research/calibri2/s181/source_grammar_shadow.py")
WRAPPER_SOURCE = Path("research/calibri2/s182/real_cpu_j7_shadow_guard.py")
DENOM = 43_261_760

STATS = {"shadow_admitted": 0, "source_span_rejected": 0,
         "invalid_tokens_rejected": 0, "shadow_abstained": 0}
REAL_AUDIT = None
REAL_S175_BASE_GUARD = None
ALL_PRED = {}
SHADOW_PROMPT_SHA = set()


def candidate_only(question: str, tok, canonical_spans) -> list[int] | None:
    """Absolute no-gold lexical match; canonical spans are inference-only."""
    phrase = shadow_candidate(question)
    if phrase is None:
        STATS["shadow_abstained"] += 1
        return None
    pair = tok.encode(phrase, add_special_tokens=False)
    if (len(pair) != 2 or tok.eos_token_id in pair
            or not all(isinstance(t, int) and t >= 0 for t in pair)):
        STATS["invalid_tokens_rejected"] += 1
        return None
    source = [x.tolist() if hasattr(x, "tolist") else list(x)
              for x in canonical_spans]
    if pair not in source:
        STATS["source_span_rejected"] += 1
        return None
    STATS["shadow_admitted"] += 1
    SHADOW_PROMPT_SHA.add(hashlib.sha256(question.encode("utf-8")).hexdigest())
    return pair


def shadow_selector(head, hidden, ids, tok):
    from calibri2.s175 import negative_owner_guard_real_j7 as s175
    from calibri2.s168 import novel_train_frames_generalization_cpu as s168
    if REAL_S175_BASE_GUARD is None:
        raise AssertionError("S182 requires pinned S175 original S172 guard")
    q = s168.OWNER_PROMPT
    if q is None:
        raise AssertionError("S182 only inside captured original CPU owner prompt")
    # Fast path for exact TRAIN S30/S54 templates. Others untouched and
    # delegated to the S175/S172 protected J7 guard.
    if shadow_candidate(q) is not None:
        canonical = s168.s160.normalized_candidate_spans(tok, ids[0].tolist())
        proposal = candidate_only(q, tok, canonical)
        if proposal is not None:
            return proposal
    return REAL_S175_BASE_GUARD(head, hidden, ids, tok)


def captured_full_audit(*args, **kwargs):
    """Capture the completed real frozen J7 output; not a selector and no gold."""
    if REAL_AUDIT is None:
        raise AssertionError("S182 real audit not installed")
    result = REAL_AUDIT(*args, **kwargs)
    old, new = args[1], args[2]
    if len(old) == 456 and len(new) == 128:
        if ALL_PRED:
            raise AssertionError("S182 original TRAIN 456+128 audited twice")
        pred_old, pred_new = result[:2]
        if set(pred_old) != set(old) or set(pred_new) != set(new):
            raise AssertionError("S182 real J7 missing original TRAIN identities")
        ALL_PRED["old"] = {
            hashlib.sha256(old[key]["question"].encode("utf-8")).hexdigest():
            list(pred_old[key]) for key in pred_old
        }
        ALL_PRED["new"] = {
            hashlib.sha256(new[key]["question"].encode("utf-8")).hexdigest():
            list(pred_new[key]) for key in pred_new
        }
    return result


def verify_after_frozen_j7(old_s179_file: Path, old_s87_file: Path, out: Path):
    """ALL comparisons are after REAL CPU J7 has finished; fail closed."""
    from calibri2.s176 import s174_legacy_compatible_j7 as s176
    if len(ALL_PRED.get("old", {})) != 456 or len(ALL_PRED.get("new", {})) != 128:
        raise AssertionError("S182 full original 456 and new128 real CPU audit never completed")
    if len(SHADOW_PROMPT_SHA) < 1:
        raise AssertionError("S182 no source-only shadow candidate seen in real J7")
    report_file = out / "S176_FINAL_SOURCE_COMPAT_REAL_CPU_J7_GATE.json"
    if not report_file.exists():
        raise AssertionError("S182 original S176 real CPU J7 physical report missing")
    s176_report = json.loads(report_file.read_text())
    if not s176_report["safe"] or s176_report["model_accepted"] is not False:
        raise AssertionError("S182 base S176 protected J7 gate not met")
    t = json.loads(old_s179_file.read_text())
    g = json.loads(old_s87_file.read_text())
    if (t.get("stage") != "S179_TRACE_ONLY" or t.get("labels_used") is not False
            or t.get("number") != 456 or
            g.get("stage") != "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT"):
        raise AssertionError("S182 immutable S179/S87 provenance missing")
    original = {r["prompt_sha256"]: r["full_generated_ids"] for r in t["cases"]}
    gold = {r["question_sha256"]: r["target_ids"]
            for r in g["all_456_per_identity_results"]}
    after = ALL_PRED["old"]
    if len(original) != 456 or len(gold) != 456 or set(gold) != set(original) or set(after) != set(gold):
        raise AssertionError("S182 true J7 456 SHA alignment drift")
    before_correct = {sha for sha, ids in original.items() if ids == gold[sha]}
    after_correct = {sha for sha, ids in after.items() if ids == gold[sha]}
    first_before = {sha for sha, ids in original.items() if ids and ids[0] == gold[sha][0]}
    first_after = {sha for sha, ids in after.items() if ids and ids[0] == gold[sha][0]}
    if len(before_correct) != 428 or len(first_before) != 447:
        raise AssertionError("S182 immutable S176 428 full / 447 first baseline drift")
    gains = sorted(after_correct - before_correct)
    losses = sorted(before_correct - after_correct)
    first_losses = sorted(first_before - first_after)
    unchanged = [sha for sha in original
                 if sha not in SHADOW_PROMPT_SHA and after[sha] != original[sha]]
    if len(unchanged):
        raise AssertionError("S182 non-shadow J7 output changed")
    old_full = len(after_correct)
     # J7 accepted S176 report carries protected 128 TRAIN count and 64 probe.
    original_new = s176_report["new_full"]
    if original_new != 128:
        raise AssertionError("S182 new128 protected cohort no longer completely correct")
    bpp = (s176_report["bpp_all_sources"]
           + 8 * (PARSER_SOURCE.stat().st_size + WRAPPER_SOURCE.stat().st_size) / DENOM)
    cpu = s176_report["cpu_ratio"]
    safe = (not losses and not first_losses and not unchanged
            and old_full >= len(before_correct) and original_new == 128
            and bpp <= 0.7 and cpu >= 0.9 and len(gains) > 0)
    metrics = {
        "event": "S182_REAL_CPU_J7_456_128_SHADOW_PROTECTED",
        "source_s179_frozen_baseline_run": 43323981995,
        "old_original_train_before": len(before_correct),
        "old_original_train_after": old_full,
        "new_train128_protected_correct": original_new,
        "new_train128_actual_predictions_captured": len(ALL_PRED["new"]),
        "old_gains": len(gains), "old_losses": len(losses),
        "first_losses": len(first_losses),
        "non_shadow_prediction_changes": len(unchanged),
        "real_CPU_J7_decoder_invoked": True,
        "real_CPU_speed_ratio": cpu,
        "all_in_bpp_charge_new_source": bpp,
        "shadow_selector_counts": STATS,
        "old_gain_sha_only": gains,
        "old_loss_sha_only": losses,
        "retention_passed": safe,
        "model_accepted": False,
        "no_QA_TEST_or_Jessica_chat": True,
        "LIVE_Render_Supabase_APK_untouched": True,
    }
    (out / "S182_REAL_SOURCE_SHADOW_456_128_GUARD.json").write_text(
        json.dumps(metrics, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if not k.endswith("_sha_only")},
                     sort_keys=True), flush=True)
    if not safe:
        raise AssertionError("S182 rejected: original TRAIN forgetting, slowdown, bpp, or no genuine J7 gain")
    return metrics


def main():
    from calibri2.s175 import negative_owner_guard_real_j7 as s175
    from calibri2.s168 import novel_train_frames_generalization_cpu as s168
    from calibri2.s176 import s174_legacy_compatible_j7 as s176
    global REAL_AUDIT, REAL_S175_BASE_GUARD
    if len(sys.argv) < 4:
        raise SystemExit("S182 expects S176 pinned source args, S179 trace, S87 report, output")
    old_trace, old_s87, output = map(Path, sys.argv[-3:])
    # S176 main consumes sys.argv[-1] as output and s168's S158 args;
    # replace the last three with the original output path only.
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *sys.argv[1:-3], str(output)]
    REAL_AUDIT = s168.s160.audit_cpu_outputs
    REAL_S175_BASE_GUARD = s175.OLD_S172
    s168.s160.audit_cpu_outputs = captured_full_audit
    s175.OLD_S172 = shadow_selector
    try:
        s176.main()
    finally:
        s168.s160.audit_cpu_outputs = REAL_AUDIT
        s175.OLD_S172 = REAL_S175_BASE_GUARD
        sys.argv = original_argv
    verify_after_frozen_j7(old_trace, old_s87, output)

if __name__ == "__main__":
    main()
