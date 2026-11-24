"""Z44 READ-ONLY pinned Z43 prior legal TRAIN84 causal second-token rank audit.

Consumes one immutable successful Z43 model report; no model, optimizer,
DEV, sealed J7/EVAL128/GOLD or new examples. This is diagnostic *TRAIN*
fit/reachability, not independent accuracy evidence or promotion.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter,defaultdict
from math import log2
from pathlib import Path

STAGE="CALIBRI_S119Q2I8Z43_REAL43M_LEGAL_TRAIN84_ONPOLICY_NATIVE_BPE2_3_4_G0"
SRC_SHA="a66833e8f4e83e4ba080dc3db0b3032a3315bcfc2fe7cee473e7e9ea7569023e"
BINS=(1,2,4,8,16,32,64)
def require(cond,message):
    if not cond:raise ValueError("Z44 pinned TRAIN84 only fail-closed: "+message)
def stats(values):
    require(values and all(isinstance(x,int) and x>=1 for x in values),
            "positive native ranks")
    ordered=sorted(values)
    n=len(ordered)
    return {
        "n":n,
        "rank_min":ordered[0],
        "rank_median_lower":ordered[(n-1)//2],
        "rank_p75_nearest_rank":ordered[(3*n+3)//4-1],
        "rank_max":ordered[-1],
        "topk_cumulative_counts":{str(k):sum(v<=k for v in ordered) for k in BINS},
        "beyond_original_TOP64":sum(v>64 for v in ordered),
    }
def empirical(ids):
    require(ids and all(isinstance(x,int) and x>=0 for x in ids),"actual native IDs")
    d=Counter(ids);n=len(ids)
    return {"distinct_IDs":len(d),"empirical_cross_prompt_entropy_bits":round(
        -sum((v/n)*log2(v/n) for v in d.values()),6),
        "majority_fraction":round(max(d.values())/n,6)}
def audit(report):
    require(report["stage"]==STAGE and report["candidate_accepted"] is False and
            report["output_gate"]==0 and
            report["Z40_previous_legal_TRAIN84_SHA256"]==SRC_SHA and
            report["no_new_DEV_or_GOLD_evaluation"] is True and
            report["no_fitted_shadow_training_or_optimizer_steps"] is True and
            report["previous_Z40_DEV14_remains_SPENT_not_used"] is True and
            report["no_616J7_EVAL128_GOLD_bpp_CB59_CPU_certification"] is True and
            report["original_S142_43M_GT12_Core_NRA3_NPA1_and_Z10_A_B_unchanged"] is True and
            report["real_original43M_native_49152_decoder_forwards"]==332 and
            report["first_native_exact_on_84_TRAIN"]==77 and
            report["first_three_native_exact_on_84_TRAIN"]==0,
            "immutable original S142 43M prior physical Z43 research source")
    cases=report["raw_training_samples"]
    require(len(cases)==84 and len({r["source_id"] for r in cases})==84 and
            len({r["prompt_SHA256"] for r in cases})==84 and
            sum(r["physical_original43M_forwards"] for r in cases)==332,
            "all 84 physical TRAIN examples unique and paid forwards parity")
    segments=defaultdict(list)
    for row in cases:
        require(row["source_only_previous_legitimate_TRAIN"] is True and
                len(row["positions"])==3 and
                1<=len(row["own_original_greedy_native_first4"])<=4 and
                row["own_first_correct"]==
                  (row["own_contiguous_correct_prefix_first4"]>=1),
                "past legal TRAIN-only self-greedy sequence")
        own=row["own_original_greedy_native_first4"]
        if len(own)>1:segments["all_generated_BPE2"].append(row)
        if row["own_first_correct"]:segments["first_native_correct"].append(row)
        else:segments["first_native_wrong"].append(row)
        segments["kind_"+str(row["kind"])].append(row)
        for pos,rec in zip((2,3,4),row["positions"]):
            require(rec["position"]==pos and
                    rec["generated"]==(len(own)>=pos) and
                    (not rec["generated"] or
                     (rec["native_gold_rank_on_own_prefix"] is not None and
                      rec["native_gold_rank_on_own_prefix"]>=1)) and
                    rec["prior_prefix_gold"]==
                       (row["own_contiguous_correct_prefix_first4"]>=pos-1
                        if rec["generated"] else False),
                    "native BPE rank + preceding prefix consistency")
            if not rec["generated"]:continue
            require(rec["native_gold_in_original_TOP64_on_own_prefix"]==
                      (rec["native_gold_rank_on_own_prefix"]<=64) and
                    (not rec["native_gold_greedy_at_position"] or
                     rec["native_gold_rank_on_own_prefix"]==1),
                    "causal candidate gold rank parity")
    first=segments["first_native_correct"]
    wrong=segments["first_native_wrong"]
    def pos_score(rows,pos):
        records=[r["positions"][pos-2] for r in rows if r["positions"][pos-2]["generated"]]
        ranks=[r["native_gold_rank_on_own_prefix"] for r in records]
        return {"rank":stats(ranks) if ranks else None,
                "generated":len(records),
                "next_native_gold_greedy":sum(x["native_gold_greedy_at_position"] for x in records)}
    output={
      "first_BPE_correct_TRAIN":len(first),"first_BPE_wrong_TRAIN":len(wrong),
      "first2_on_own_prefix_all84":pos_score(cases,2),
      "first2_after_naturally_correct_first77":pos_score(first,2),
      "first2_after_wrong_first7":pos_score(wrong,2),
      "first3_after_naturally_correct_firsttwo":pos_score([
          r for r in cases if r["own_contiguous_correct_prefix_first4"]>=2],3),
      "first3_all_selfprefix_BPE2_mostly_wrong":pos_score(cases,3),
      "first4_all_selfprefix_BPE3_mostly_wrong":pos_score(cases,4),
      "empirical_original_greedy_second_IDs":empirical([
          r["own_original_greedy_native_first4"][1] for r in cases]),
      "per_kind_correct_first_and_second":{
        kind.removeprefix("kind_"):{
           "n":len(rows),
           "correct_first":sum(r["own_first_correct"] for r in rows),
           "correct_first2":sum(r["own_contiguous_correct_prefix_first4"]>=2 for r in rows),
           "gold_second_top64_after_gold_first":sum(
               r["positions"][0]["native_gold_in_original_TOP64_on_own_prefix"]
               for r in rows if r["own_first_correct"])
        } for kind,rows in sorted(segments.items()) if kind.startswith("kind_")
      },
    }
    require(output["first2_after_naturally_correct_first77"]["rank"][
                "topk_cumulative_counts"]["64"]==49 and
            output["first2_on_own_prefix_all84"]["rank"][
                "topk_cumulative_counts"]["64"]==51 and
            output["first3_after_naturally_correct_firsttwo"]["generated"]==1 and
            output["first3_after_naturally_correct_firsttwo"]["rank"][
                "topk_cumulative_counts"]["64"]==0,
            "physical Z43 causal counted gold top64 / reachable prefix parity")
    return {
      "stage":"S119Q2I8Z44_READONLY_Z43_PRIOR_TRAIN84_BPE2_DISCRIMINATION",
      "no_new_S142_43M_native_forwards":True,
      "no_new_optimizer_or_tests_through_Jessica":True,
      "old_LEGAL_TRAIN84_only":True,
      "not_a_independent_DEV_validation":True,
      "previous_DEV14_not_reused":True,
      "reference_previous_original43M_forwards":332,
      "candidate_accepted":False,"output_gate":0,
      "diagnostics":output,
      "conclusion":"SECOND_BPE_RANKER_MUST_SELECT_WITHOUT_COLLAPSE; DO_NOT_ASSUME_BPE3_TEACHER_OR_WRONGPREFIX_TOP64_IS_CAUSALLY_REACHABLE"
    }
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    out=audit(json.loads(a.input.read_text(encoding="utf8")))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
                     encoding="utf8")
    print("S119Q2I8Z44_TRAIN_ONLY_SECOND_RANK "+json.dumps(out,ensure_ascii=False,sort_keys=True),flush=True)
if __name__=="__main__":
    main()
