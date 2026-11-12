#!/usr/bin/env python3
"""S170: guarded S169 structural source-only fallback atop frozen S167 real J7.

No neural weight updates, QA/TEST, LIVE, Jessica/Ren der/Supabase writes.
S168 64 out-of-template TRAIN probes are assessed only after full+EOS decoding.
This is lexical source-copy routing, NOT proof of learned neural generalization.
"""
import json
import sys
from pathlib import Path

from calibri2.s168 import novel_train_frames_generalization_cpu as s168
from calibri2.s169.owner_segment_offline import owner_value, failclosed_checks

COUNTERS = s168.COUNTERS
S167_SELECTOR = s168.owner_guarded_parallel
S170 = "S170_STRUCTURAL_OWNER_INTEGRATED_CPU_J7"
WRAPPER_PATH = Path("research/calibri2/s170/structural_owner_integrated_cpu.py")
STRUCTURAL_PATH = Path("research/calibri2/s169/owner_segment_offline.py")


def s170_owner_guarded_parallel(head, hidden, ids, tok):
    """Only rescue a rejected neural candidate from unambiguous prompt evidence."""
    prev_low = COUNTERS["S160_confidence_fallback"]
    prev_conflict = COUNTERS["S161_same_attribute_conflict_parent_fallback"]
    kept = S167_SELECTOR(head, hidden, ids, tok)
    if kept is not None:
        return kept
    low = COUNTERS["S160_confidence_fallback"] > prev_low
    conflict = COUNTERS["S161_same_attribute_conflict_parent_fallback"] > prev_conflict
    if not (low or conflict):
        return None
    question = s168.OWNER_PROMPT
    if question is None:
        raise AssertionError("S170 requires literal prompt")
    source_phrase = owner_value(question)
    if source_phrase is None:
        COUNTERS["S170_source_owner_abstained"] += 1
        return None
    candidate = tok.encode(source_phrase, add_special_tokens=False)
    if len(candidate) != 2 or tok.eos_token_id in candidate:
        COUNTERS["S170_token_guard_abstained"] += 1
        return None
    source_spans = s168.s160.normalized_candidate_spans(tok, ids[0].tolist())
    if not any(row.tolist() == candidate for row in source_spans):
        COUNTERS["S170_non_source_span_abstained"] += 1
        return None
    COUNTERS["S170_strict_structural_owner_rescues"] += 1
    return candidate


def main():
    positive, negatives = failclosed_checks()
    if len(positive) != 8 or len(negatives) != 7 or not all(positive + negatives):
        raise AssertionError("S170 S169 parser requires 8/8 positives and 7/7 negatives")
    s168.STAGE = S170
    s168.owner_guarded_parallel = s170_owner_guarded_parallel
    original_out = Path(sys.argv[-1])
    s168.main()  # freezes S142/S160, real CPU 456+128+64+64 J7 / EOS
    novel = json.loads((original_out/"S168_NOVEL_TRAIN_FRAMES_64_GENERALIZATION.json").read_text())
    j7 = json.loads((original_out/"S167_REAL_OWNER_BINDING_J7_SUMMARY.json").read_text())
    full = novel["candidate_full_correct"]
    pairs = novel["novel_pairs_both_correct"]
    losses = novel["lost_previous_correct_novel_cases"]
    original_bpp = j7["all_in_bpp"]
    # Count both newly required source files; 43,261,760 is a conservative
    # LOWER denominator than the true core + head physical parameter total.
    extra_source_bytes = WRAPPER_PATH.stat().st_size + STRUCTURAL_PATH.stat().st_size
    charged_bpp = original_bpp + 8 * extra_source_bytes / 43_261_760
    speed = j7["cpu_ratio"]
    safe = bool(j7["safe_research_candidate"] and
                full == 64 and pairs == 32 and not losses and
                charged_bpp <= 0.7 and speed >= 0.9)
    summary = {
        "stage":S170,
        "S168_baseline_run":43289943629,
        "S169_offline_parser_run":43292379800,
        "protected_J7_safe":j7["safe_research_candidate"],
        "protected_456_128_64_retention":j7["J7_retention"],
        "protected_probe_64_retention":j7["probe_retention"],
        "novel_S168_64_full_EOS":full,
        "novel_32_pairs_both_full_EOS":pairs,
        "novel_previous_correct_losses":losses,
        "S170_structural_rescues":COUNTERS["S170_strict_structural_owner_rescues"],
        "S170_source_owner_abstained":COUNTERS["S170_source_owner_abstained"],
        "all_in_bpp_conservative":charged_bpp,
        "cpu_ratio":speed,
        "safe_in_research":safe,
        "model_accepted":False,
        "new_training_steps":0,
        "not_neural_generalization_proof":True,
        "QA_TEST_never_run":True,
        "LIVE_untouched":True,
        "Jessica_tests_never_run":True,
    }
    (original_out/"S170_INTEGRATED_STRUCTURAL_OWNER_FINAL_GATE.json").write_text(
        json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2))
    print(json.dumps({
        "event":"S170_INTEGRATED_STRUCTURAL_OWNER_REAL_J7",
        "novel_full":full,"novel_pairs":pairs,"novel_losses":len(losses),
        "J7_safe":j7["safe_research_candidate"],
        "structural_rescues":summary["S170_structural_rescues"],
        "bpp":charged_bpp,"cpu_ratio":speed,"safe":safe,
        "model_accepted":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
