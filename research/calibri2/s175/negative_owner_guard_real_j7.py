#!/usr/bin/env python3
"""S175: isolated research S174 fail-closed owner routing + full real CPU J7.
Frozen neural and Green Tunnel weights; no QA/TEST, Jessica, LIVE, Render/Supabase.
"""
import json,os,sys,re
from pathlib import Path
from calibri2.s172 import confident_owner_consistency_cpu as s172
from calibri2.s174.negation_veto_conjunction_offline import safe_owner_value,NEGATIVE_CLAIMS,new_adversarial
from calibri2.s173.novel_train_owner_adversarial_offline import synthesize_frozen_cases
from calibri2.s169.owner_segment_offline import failclosed_checks

s170=s172.s170
s168=s170.s168
COUNTERS=s170.COUNTERS
OLD_S172=s172.owner_consistent_confident_guard
SOURCE=Path("research/calibri2/s175/negative_owner_guard_real_j7.py")
PHYSICAL_DENOM=43_261_760

def guard(head,hidden,ids,tok):
    q=s168.OWNER_PROMPT
    if q is None: raise AssertionError("S175 requires captured literal source question")
    if q.startswith("TRAIN S82 ") and ". " in q:
        source,req=q.split(": ",1)[-1].rsplit(". ",1)
        unsafe=NEGATIVE_CLAIMS.search(source) or re.search(
            r"\\b(?:both owners|both owner's|for [A-Z][a-z]+ and [A-Z][a-z]+)\\b",req,re.I)
        if unsafe:
            COUNTERS["S175_disputed_owner_fastpath_abstained"]+=1
            # Only source copy is vetoed. S142 may still answer.
            return None
    return OLD_S172(head,hidden,ids,tok)

def source_only_preflight():
    src=json.loads(Path(os.environ["S175_S168_JSON"]).read_text())
    assert src["stage"]=="S168_NOVEL_TRAIN_FRAMES_GENERALIZATION_DIAGNOSTIC"
    assert len(src["cases"])==64 and src["model_accepted"] is False
    new,neg=synthesize_frozen_cases(src["cases"])
    neg+=new_adversarial(src["cases"])
    # The parser sees only the question: score after all selections.
    pp=[safe_owner_value(r["question"]) for r in new]
    nn=[safe_owner_value(r["question"]) for r in neg]
    orig=[safe_owner_value(r["question"]) for r in src["cases"]]
    new_correct=sum(v==r["posthoc_answer"] for v,r in zip(pp,new))
    old_correct=sum(v==r["answer"] for v,r in zip(orig,src["cases"]))
    abstain=sum(v is None for v in nn)
    p,n=failclosed_checks()
    assert new_correct==old_correct==64 and abstain==64 and all(p+n)
    s172old=json.loads((Path(os.environ["S175_S172_DIR"])/
        "S170_INTEGRATED_STRUCTURAL_OWNER_FINAL_GATE.json").read_text())
    s172final=json.loads((Path(os.environ["S175_S172_DIR"])/
        "S172_CONFIDENT_OWNER_CONSISTENCY_FINAL_GATE.json").read_text())
    s174final=json.loads((Path(os.environ["S175_S174_DIR"])/
        "S174_64_NEW_TRAIN_PLUS_64_NEGATIVES_OFFLINE.json").read_text())
    assert s172final["safe_research_routing_candidate"] and s174final["safe_offline_prompt_heuristic"]
    assert s172old["novel_S168_64_full_EOS"]==64
    return {"S173_new_form_source_only_correct":new_correct,
            "S174_64_adversarial_source_only_abstained":abstain,
            "S168_source_only_preserved":old_correct,
            "old_S172_old":s172old["protected_456_128_64_retention"]["old_full_after"],
            "old_S172_new":s172old["protected_456_128_64_retention"]["new_full_after"]}

def main():
    pre=source_only_preflight()
    # Integrate ONLY newly verified S174 prompt-to-owner source selection.
    # Unchanged S172/S170 and S142/S160 neural weights and original CPU J7.
    s170.owner_value=safe_owner_value
    s172.owner_consistent_confident_guard=guard
    s172.main()
    out=Path(sys.argv[-1])
    original=json.loads((out/"S170_INTEGRATED_STRUCTURAL_OWNER_FINAL_GATE.json").read_text())
    recent=json.loads((out/"S172_CONFIDENT_OWNER_CONSISTENCY_FINAL_GATE.json").read_text())
    retention=original["protected_456_128_64_retention"]
    probes=original["protected_probe_64_retention"]
    physical=recent["bpp_including_S172_runtime_source"]
    physical+=8*sum(Path(p).stat().st_size for p in (
      "research/calibri2/s173/novel_train_owner_adversarial_offline.py",
      "research/calibri2/s174/negation_veto_conjunction_offline.py",
      SOURCE))/PHYSICAL_DENOM
    zero_regression_counts=(retention["old_full_after"]>=pre["old_S172_old"]
       and retention["new_full_after"]>=pre["old_S172_new"]
       and probes["new_full_after"]==64)
    gate=bool(recent["safe_research_routing_candidate"] and zero_regression_counts
              and original["novel_S168_64_full_EOS"]==64 and physical<=0.7
              and recent["cpu_ratio"]>=0.9)
    payload={
      "stage":"S175_S174_NEGATION_GUARD_REAL_CPU_J7",
      "S174_source_only_preflight":pre,
      "J7_retention_vs_original_S142":retention,"protected_64":probes,
      "S168_64_full_EOS":original["novel_S168_64_full_EOS"],
      "S168_32_pairs_full_EOS":original["novel_32_pairs_both_full_EOS"],
      "S172_cohort_scores_not_regressed":zero_regression_counts,
      "S174_new_form64_offline_only":pre["S173_new_form_source_only_correct"],
      "S174_negative64_offline_only":pre["S174_64_adversarial_source_only_abstained"],
      "negative_FULL_neural_generation_NOT_tested":True,
      "all_in_bpp":physical,"cpu_ratio":recent["cpu_ratio"],
      "research_safe":gate,"model_accepted":False,
      "not_neural_generalization":True,"no_weight_training":True,
      "Green_Tunnel_unchanged":True,"no_QA_TEST":True,
      "Jessica_LIVE_Render_Supabase_unchanged":True
    }
    (out/"S175_RESEARCH_J7_AND_SOURCE_ONLY_NEGATIVE_GATE.json").write_text(
      json.dumps(payload,sort_keys=True,ensure_ascii=False,indent=2))
    print(json.dumps({"event":"S175_FULL_J7_RESEARCH_ONLY",
      "old_full":retention["old_full_after"],"new_full":retention["new_full_after"],
      "protected64":probes["new_full_after"],"novel64":original["novel_S168_64_full_EOS"],
      "source_only_negatives_abstained":pre["S174_64_adversarial_source_only_abstained"],
      "bpp":physical,"cpu_ratio":recent["cpu_ratio"],"safe":gate,
      "model_accepted":False},sort_keys=True),flush=True)

if __name__=="__main__": main()
