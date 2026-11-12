#!/usr/bin/env python3
"""CALIBRI S172: source-only owner-consistency veto on confident neural choice.

S171 identified 1/64 S170 error as an unambiguous yet wrong two-token neural
span that bypassed owner checking; this research wrapper validates the
requested owner's phrase against a real contiguous source span on BOTH
confident and fallback paths. Strictly NO GOLD at inference. All weights
frozen, real protected J7 CPU, no QA/TEST, Jessica or LIVE modifications.
"""
import json
import sys
from pathlib import Path

from calibri2.s170 import structural_owner_integrated_cpu as s170

s168=s170.s168
COUNTERS=s170.COUNTERS
S170_GUARD=s170.s170_owner_guarded_parallel
S172_STAGE="S172_CONFIDENT_OWNER_SOURCE_CONSISTENCY_REAL_J7"
SOURCE=Path("research/calibri2/s172/confident_owner_consistency_cpu.py")


def owner_consistent_confident_guard(head, hidden, ids, tok):
    base = S170_GUARD(head, hidden, ids, tok)
    # Never force a fast path: preserve original S170/S167 fallback behavior.
    if base is None:
        return None
    question = s168.OWNER_PROMPT
    if question is None:
        raise AssertionError("S172 requires captured literal prompt")
    source = s170.owner_value(question)
    if source is None:
        COUNTERS["S172_confident_parser_abstained"]+=1
        return base
    pair=tok.encode(source,add_special_tokens=False)
    if len(pair)!=2 or tok.eos_token_id in pair:
        COUNTERS["S172_confident_token_guard_abstained"]+=1
        return base
    canonical=s168.s160.normalized_candidate_spans(tok,ids[0].tolist())
    if not any(x.tolist()==pair for x in canonical):
        COUNTERS["S172_confident_non_source_abstained"]+=1
        return base
    old=base.tolist() if hasattr(base,"tolist") else list(base)
    if old==pair:
        COUNTERS["S172_source_owner_confident_consistent"]+=1
        return base
    # This semantic veto uses only the literal owner's contiguous phrase,
    # NEVER expected ids, QA labels, or whether S170 was correct.
    COUNTERS["S172_confident_wrong_owner_span_vetoed"]+=1
    return pair


def main():
    s170.s170_owner_guarded_parallel=owner_consistent_confident_guard
    s170.main()
    out=Path(sys.argv[-1])
    r=json.loads((out/"S170_INTEGRATED_STRUCTURAL_OWNER_FINAL_GATE.json").read_text())
    bpp=r["all_in_bpp_conservative"] + 8*SOURCE.stat().st_size/43_261_760
    ratio=r["cpu_ratio"]
    # Full existing J7 strict zero-forgetting must hold, and 64 new both-sides.
    passed=(r["safe_in_research"] and bpp<=0.7 and ratio>=0.9)
    summary={
        "stage":S172_STAGE,
        "S171_one_miss_diagnosis_run":43295373246,
        "S170_frozen_baseline_run":43293552917,
        "old_S170_before_63_novel":True,
        "new_novel_64_full_EOS":r["novel_S168_64_full_EOS"],
        "new_novel_32_pairs_both":r["novel_32_pairs_both_full_EOS"],
        "protected_J7_456_128_64_safe":r["protected_J7_safe"],
        "old_probe_regression":r["novel_previous_correct_losses"],
        "S172_confident_owner_vetoes":COUNTERS["S172_confident_wrong_owner_span_vetoed"],
        "bpp_including_S172_runtime_source":bpp,"cpu_ratio":ratio,
        "safe_research_routing_candidate":passed,
        "model_accepted":False,
        "not_neural_generalization_proof":True,
        "no_training_steps":True,"no_QA_TEST":True,
        "LIVE_untouched":True,"Jessica_tests_never_run":True
    }
    (out/"S172_CONFIDENT_OWNER_CONSISTENCY_FINAL_GATE.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True,ensure_ascii=False))
    print(json.dumps({
        "event":"S172_OWNER_CONSISTENCY_FULL_J7_GATE",
        "novel_full":summary["new_novel_64_full_EOS"],
        "novel_pairs":summary["new_novel_32_pairs_both"],
        "protected_J7_safe":r["protected_J7_safe"],
        "confident_vetoes":summary["S172_confident_owner_vetoes"],
        "bpp":bpp,"cpu_ratio":ratio,
        "safe":passed,"model_accepted":False
    },sort_keys=True),flush=True)
if __name__=="__main__":
    main()
