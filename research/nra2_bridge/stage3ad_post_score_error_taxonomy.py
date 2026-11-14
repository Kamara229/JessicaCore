"""S3AD: read-only post-score error taxonomy on S3AC once-only independent128.

S3AA has already been scored in S3AC. This diagnosis is NOT blind, does
NOT train or choose a checkpoint, and MUST NOT be reused to claim
fresh independent evaluation. This stage reads only immutable S3AC JSON.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

S3AA_SHA="c6815f9bbb39d105d31f896f42edf96c4daac064956d724a9844f9d97818df7e"
S3AB_SHA="fe240c2003b1251be7162b6fd1a516caba2070ece0184b59f43f2d65109d2c62"
S3O_SHA="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
S3Z_SHA="60a7121b8270f647909776b58dc21b07c6cb885fd6ebeb6cb3158b23221f6e9f"
N=128

def check(cond,msg):
    if not cond:
        raise AssertionError("S3AD "+msg)

def group(rows):
    d={"count":len(rows),"yes":sum(x["gold_yes"] for x in rows),
       "graph_exact":sum(x["predicted_graph_exact"] for x in rows)}
    for arm in ("S3O","S3Z"):
        d[arm]={
            "both_roles_correct":sum(x[arm]["source_correct"] and x[arm]["destination_correct"] for x in rows),
            "both_roles_wrong":sum(not x[arm]["source_correct"] and not x[arm]["destination_correct"] for x in rows),
            "source_only_wrong":sum(not x[arm]["source_correct"] and x[arm]["destination_correct"] for x in rows),
            "destination_only_wrong":sum(x[arm]["source_correct"] and not x[arm]["destination_correct"] for x in rows),
            "reachability_correct":sum(x[arm]["reachability_correct"] for x in rows),
            "roles_correct_but_reachability_wrong":sum(
                x[arm]["source_correct"] and x[arm]["destination_correct"]
                and not x[arm]["reachability_correct"] for x in rows),
            "roles_wrong_but_reachability_correct":sum(
                (not x[arm]["source_correct"] or not x[arm]["destination_correct"])
                and x[arm]["reachability_correct"] for x in rows),
        }
    d["paired"]={
        "S3O_only_reach_correct":sum(x["S3O"]["reachability_correct"] and not x["S3Z"]["reachability_correct"] for x in rows),
        "S3Z_only_reach_correct":sum(not x["S3O"]["reachability_correct"] and x["S3Z"]["reachability_correct"] for x in rows),
        "both_reach_correct":sum(x["S3O"]["reachability_correct"] and x["S3Z"]["reachability_correct"] for x in rows),
        "neither_reach_correct":sum(not x["S3O"]["reachability_correct"] and not x["S3Z"]["reachability_correct"] for x in rows),
    }
    return d

def main():
    if len(sys.argv)!=3:
        raise SystemExit("S3AD S3AC_RESULT_DIR NEW_OUTPUT_DIR")
    source,out=map(Path,sys.argv[1:])
    check(not out.exists() or not any(out.iterdir()),"immutable stage result must be created only once")
    original=json.loads((source/"NRA2_S3AC_INDEPENDENT128_ROLE_COMPARE_NO_RETUNE.json").read_text())
    check(original.get("stage")=="NRA2_S3AC_ONCE_ONLY_INDEPENDENT128_S3O_VS_S3Z_ROLE_COMPARISON"
          and original.get("S3AA_commitment_sha256")==S3AA_SHA
          and original.get("S3AB_native_tokens_fp16_sha256")==S3AB_SHA
          and original.get("S3O_original_FP16_sha256")==S3O_SHA
          and original.get("S3Z_role_head_FP16_sha256")==S3Z_SHA
          and original.get("original_616_S194_J7_digest_preserved_by_S3AB") is True
          and original.get("graph_weights_bitwise_equal") is True
          and original.get("graph_logits_bitwise_equal") is True
          and original.get("hard_edge_threshold_predeclared")==0.5
          and original.get("once_only_independent_128_rows_compared")==N
          and original.get("candidate_accepted") is False
          and original.get("output_gate")==0.0
          and original.get("train_or_retune_after_scoring") is False,
          "S3AC held-out provenance / original J7 / fixed graph gate")
    rows=original["question_level"]
    check(len(rows)==N and {x["index"] for x in rows}==set(range(N))
          and len({x["prompt_sha256"] for x in rows})==N,"question-level complete unique")
    check(all(x["predicted_graph_exact"] and x["family"] in ("chain","fork","cycle_sink","diamond")
              and x["query_style"] in (0,1,2,3)
              and x["destination_first"]==(x["query_style"] in (0,2))
              for x in rows),"original graph / style consistency")
    cats={"all":rows,
          "destination_first":[x for x in rows if x["destination_first"]],
          "source_first":[x for x in rows if not x["destination_first"]],
          "yes":[x for x in rows if x["gold_yes"]],
          "no":[x for x in rows if not x["gold_yes"]],
          **{"style_"+str(i):[x for x in rows if x["query_style"]==i] for i in range(4)},
          **{"family_"+f:[x for x in rows if x["family"]==f] for f in
              ("chain","fork","cycle_sink","diamond")}}
    breakdown={k:group(v) for k,v in cats.items()}
    for k,m in breakdown.items():
        prior=original["metrics"][k]
        check(prior["count"]==m["count"]
              and prior["exact_graph"]==m["graph_exact"]
              and prior["yes"]==m["yes"]
              and prior["paired_reachability"]["S3O_only_correct"]==m["paired"]["S3O_only_reach_correct"]
              and prior["paired_reachability"]["S3Z_only_correct"]==m["paired"]["S3Z_only_reach_correct"]
              and prior["paired_reachability"]["both_correct"]==m["paired"]["both_reach_correct"]
              and prior["paired_reachability"]["both_wrong"]==m["paired"]["neither_reach_correct"],
              "S3AC metric/question-level consistency "+k)
        for arm in ("S3O","S3Z"):
            check(prior[arm]["both_roles_correct"]==m[arm]["both_roles_correct"]
                  and prior[arm]["reachability_hard_correct"]==m[arm]["reachability_correct"]
                  and sum(m[arm][z] for z in
                        ("both_roles_correct","both_roles_wrong","source_only_wrong","destination_only_wrong"))==m["count"],
                  "S3AC source/destination role case taxonomy mismatch "+k+" "+arm)
    check(breakdown["all"]["count"]==128
          and breakdown["all"]["graph_exact"]==128
          and breakdown["destination_first"]["count"]==64
          and breakdown["source_first"]["count"]==64
          and breakdown["all"]["S3O"]["reachability_correct"]==70
          and breakdown["all"]["S3Z"]["reachability_correct"]==85,
          "original once-only independent128 fixed outcome drift")
    result={
        "stage":"NRA2_S3AD_POST_SCORE_INDEPENDENT128_ERROR_TAXONOMY_ONLY",
        "source_successful_S3AC_run":43535868240,
        "original_S3AA_commitment_sha256":S3AA_SHA,
        "original_S3AB_native_token_fp16_sha256":S3AB_SHA,
        "original_S3O_fp16_sha256":S3O_SHA,
        "trained_S3Z_fp16_sha256":S3Z_SHA,
        "original_S3AC_128_questions_scored_before_S3AD":True,
        "same_original_616_S194_J7_digest_preserved_in_S3AB":True,
        "original_graph_exact_on_independent128":128,
        "new_dataset_or_new_inference_or_training":False,
        "post_hoc_error_diagnosis_not_new_blind_eval":True,
        "S3AA_must_not_be_reused_for_fresh_independent_model_selection":True,
        "specific_weakness":"destination-first semantic source-target binding",
        "next_safe_remediation":"preregister separate TRAIN-only new grammar and independently held-out NEW evaluation topologies before training",
        "groups":breakdown,
        "candidate_accepted":False,"output_gate":0.0,
        "Jessica_LIVE_Render_Supabase_APK_main_QA_TEST_unchanged":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3AD_POST_SCORE_ROLE_ERROR_TAXONOMY_NO_RETUNE.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"stage":result["stage"],"overall":breakdown["all"],
       "destination_first":breakdown["destination_first"],
       "source_first":breakdown["source_first"],
       "no_training":True,"candidate_accepted":False,"output_gate":0.0},
       sort_keys=True),flush=True)

if __name__=="__main__":
    main()
