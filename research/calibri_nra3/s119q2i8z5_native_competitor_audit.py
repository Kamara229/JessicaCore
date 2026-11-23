"""S119q2i8z5: READ-ONLY competing native-token collapse from completed Z4.

Inspect 252 ALREADY COMPLETED actual physical original S142 43M
12GT+Core+NRA3+NPA1/native49152 gold previous human BPE2/BPE3
causal gain0/.25/1.0 logits on 14 ALREADY SPENT synthetic Z2 DEV
across three SHA-pinned previously trained FP16 model states.
Classify top1 native ID concentration, wrong EOS, and pathological
rank-better-but-gold-likelihood-worse on same prompt.
No model forward, no teacher, no training, no new DEV, no rollout.
No protected J7/EVAL128/GOLD. Gate0, no promotion.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from research.calibri_nra3.s119q2i8z4_native_gain_contract import (
    STAGE as Z4_STAGE, STATES, POSITIONS, GAINS, EXPECTED_PHYSICAL_FORWARDS,
    classify_summary, NATIVE_VOCAB
)
from research.calibri_nra3.s119q2i8z2_new_ru_train_dev_source import verify as z2_verify

STAGE="CALIBRI_S119Q2I8Z5_ORIGINAL43M_NATIVE49K_SECONDTHIRD_HEADGAIN_WRONG_COMPETITOR_AUDIT"
SOURCE_REPORT="S119Q2I8Z4_TRUE43M_NATIVE49152_HEADGAIN_SECOND_THIRD_ALL12GT_252_G0.json"
REPORT="S119Q2I8Z5_TRUE43M_252_NATIVE_HEAD_GAIN_WRONG_COMPETITOR_AND_EOS_AUDIT_G0.json"

def check(cond,msg):
    if not cond:raise ValueError("CALIBRI S119q2i8z5 read-only fail-closed: "+msg)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def examine(src):
    z2=z2_verify()
    check(src["stage"]==Z4_STAGE
          and src["candidate_accepted"] is False
          and src["output_gate"]==0
          and src["source_Z2_14_now_SPENT_author_DEV_sha256"]==z2["Z2_14_DEV_sha256"]
          and src["real_original_S142_43M_original_12GT_49152_native_forwards"]==
              EXPECTED_PHYSICAL_FORWARDS==252
          and src["all_three_SAVED_421793_FP16_neural_plus_fitted12_GT_verified"] is True
          and src["original_43M_and_ALL_fitted12GT_and_PREVIOUS_W_Z2_research_FP16_preserved"] is True
          and src["same14_ALREADY_SPENT_synthetic_author_DEV_ONLY_not_blind"] is True
          and src["two_positions_BPE2_and_BPE3_gold_previous_prefix_only_NOT_OWN"] is True
          and src["no_optimizer_backward_new_training_or_source135M_teacher"] is True,
          "only exactly physically finished 252 native49152 Z4 43M and same spent14 DEV")
    records=src["all_252_TRUE_native49152_logit_rank_records"]
    previous=src["per_state_by_gold_previous_native_BPE2_BPE3_full49k_rank_statistics"]
    check(classify_summary(records)==previous,
          "recompute all exact source/position/gain 252 actual original43M head stats before reporting")
    by={(x["state"],x["position_native_zero_based"],x["source_id"],x["native_head_gain"]):x
        for x in records}
    check(len(by)==EXPECTED_PHYSICAL_FORWARDS,"no duplicate source/position/gain winner IDs")
    out={}
    for state in STATES:
        out[state]={}
        for pos in POSITIONS:
            ids={x["source_id"] for x in records
                 if x["state"]==state and x["position_native_zero_based"]==pos}
            check(len(ids)==14,"exact 14 original scored synthetic Z2 prompts, never re-used as TRAIN")
            results={}
            for g in GAINS:
                rs=[by[(state,pos,id,g)] for id in ids]
                counts=Counter(x["greedy_native_id"] for x in rs)
                check(sum(counts.values())==14 and
                      all(0<=k<NATIVE_VOCAB for k in counts) and
                      all(x["correct_gold_native_id"] in range(NATIVE_VOCAB)
                          and x["correct_gold_native_id"]!=2
                          and len(x["top5_original_native_BPE_ids"])==5
                          and len(set(x["top5_original_native_BPE_ids"]))==5
                          and x["greedy_native_id"]==x["top5_original_native_BPE_ids"][0]
                          for x in rs),
                      "true direct physical original49k greedy and top5 token IDs, no invented text")
                probs=[v/14 for v in counts.values()]
                results[f"gain_{g:g}"]={
                    "source_count_ALREADY_SPENT":14,
                    "distinct_top1_competitor_native_IDs":len(counts),
                    "most_frequent_competitor_count":max(counts.values()),
                    "most_frequent_competitor_native_IDs":sorted(
                        k for k,v in counts.items() if v==max(counts.values())),
                    "top1_competitor_Shannon_entropy_nats":
                        round(-sum(p*math.log(p) for p in probs),6),
                    "wrong_EOS_greedy_count":counts.get(2,0),
                    "top1_competitor_distribution_NATIVE_ONLY_NOT_TEXT":{
                        str(k):v for k,v in counts.most_common()},
                    "gold_NATIVE_TOP1":sum(x["correct_gold_top1"] for x in rs),
                    "gold_NATIVE_TOP5":sum(x["correct_gold_rank_top5"] for x in rs),
                    "this_is_teacherforced_goldPREV_NOT_autoregressive_OWN":True,
                }
            # Paired gain=.25 vs gain=1.0 on original same source/gold ID,
            # exposing cases where a numerically better rank hides a
            # drastically worse normalized probability of the gold token.
            paired=[]
            for id in sorted(ids):
                baseline=by[(state,pos,id,.25)]
                amplified=by[(state,pos,id,1.)]
                check(baseline["correct_gold_native_id"]==
                      amplified["correct_gold_native_id"],
                      "causality same HUMAN gold native token / source")
                paired.append({
                    "source_id":id,
                    "amplified_native_gold_rank_improved":
                        amplified["correct_gold_native_rank_full49152"]<
                        baseline["correct_gold_native_rank_full49152"],
                    "amplified_native_gold_rank_worsened":
                        amplified["correct_gold_native_rank_full49152"]>
                        baseline["correct_gold_native_rank_full49152"],
                    "amplified_native_gold_NLL_worse":
                        amplified["correct_gold_native_NLL_nats"]>
                        baseline["correct_gold_native_NLL_nats"]+1e-6,
                    "native_gold_NLL_difference_gain1_minus_gain025":
                        round(amplified["correct_gold_native_NLL_nats"]-
                              baseline["correct_gold_native_NLL_nats"],7),
                    "native_top1_changed":
                        baseline["greedy_native_id"]!=amplified["greedy_native_id"],
                    "amplified_wrong_EOS_competitor":amplified["greedy_native_id"]==2,
                })
            results["same14_paired_gain1_vs_trained_original025"]={
                "rank_improves_but_gold_normalized_NLL_worsens":sum(
                    x["amplified_native_gold_rank_improved"] and
                    x["amplified_native_gold_NLL_worse"] for x in paired),
                "both_gold_rank_and_NLL_worsen":sum(
                    x["amplified_native_gold_rank_worsened"] and
                    x["amplified_native_gold_NLL_worse"] for x in paired),
                "native_NLL_worsens":sum(x["amplified_native_gold_NLL_worse"]
                                         for x in paired),
                "greedy_top1_changed":sum(x["native_top1_changed"] for x in paired),
                "wrong_EOS_after_amplification":sum(
                    x["amplified_wrong_EOS_competitor"] for x in paired),
                "source_pairs_ALREADY_SPENT_14":paired,
            }
            out[state][f"human_gold_next_BPE_position_{pos+1}"]=results
    return {
        "stage":STAGE,
        "original_real_43M_native49152_physical_forwards_reused":252,
        "new_neural_model_forwards_in_this_only_source_audit":0,
        "three_original_421793_FP16_neural_and_12GT_fitted_states":out,
        "source_Z2_14_SYNTHETIC_DEV_SPENT_sha256":z2["Z2_14_DEV_sha256"],
        "this_is_AFFINE_native_gain_gold_previous_BPE_NOT_own_generation":True,
        "no_training_or_model_selection_from_spent14":True,
        "no_J7_EVAL128_GOLD_bpp_CPU_tests":True,
        "no_main_LIVE_Render_Supabase_APK_Jessica_chat":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"POSTHOC_NATIVE_BPE2_BPE3_OVERSCALE_EOS_CONCENTRATION_ONLY_NO_PROMOTION",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--z4-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    src=args.z4_dir/SOURCE_REPORT
    check(src.is_file() and not args.out.exists(),
          "only completed original43M 252 native Z4 report, no overwrite")
    r=examine(json.loads(src.read_text(encoding="utf8")))
    r["source_completed_252_original43M_report_sha256"]=sha(src)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
                        encoding="utf8")
    print("S119Q2I8Z5_NATIVE49K_COMPETITOR_COLLAPSE "+json.dumps({
        name:{
            position:{key:(value if key.startswith("gain_") else {
                k:v for k,v in value.items() if k!="source_pairs_ALREADY_SPENT_14"})
                for key,value in results.items()
                if key in ("gain_0.25","gain_1",
                           "same14_paired_gain1_vs_trained_original025")}
            for position,results in positions.items()}
        for name,positions in r[
            "three_original_421793_FP16_neural_and_12GT_fitted_states"].items()
    },ensure_ascii=False,sort_keys=True),flush=True)
if __name__=="__main__":main()
