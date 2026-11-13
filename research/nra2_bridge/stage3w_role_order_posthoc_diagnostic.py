"""S3W: post-hoc diagnostic ONLY: role correctness by query mention order.
No weights, gradients, checkpoints, threshold tuning, oracle inference, or LIVE changes.
S3T/S3V are previously committed. This is NOT an independent blind success test.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from research.nra2_bridge.stage3t_unseen_wording_prereg import build_rows, commitment, GROUP

S3T_SHA="034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3"
S3V_RUN=43529457952
S3U_RUN=43528076399
S3T_RUN=43527496158
FP16_SHA="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
N=64

def main():
    if len(sys.argv)!=4:
        raise SystemExit("S3W S3V_ARTIFACT_DIR S3T_PREREG_DIR OUTPUT_DIR")
    vdir,tdir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3W no overwrite")
    score=json.loads((vdir/"NRA2_S3V_S3T64_PRECOMMITTED_SOFT_HARD_CPU_TRIAL.json").read_text())
    prior=json.loads((tdir/"NRA2_S3T_NEW_GRAMMAR64_PRECOMMITTED_NO_SCORE.json").read_text())
    rows=build_rows()
    if (prior != commitment(rows)
        or prior["commitment_sha256"] != S3T_SHA
        or prior.get("S3O_frozen_adapter_fp16_sha256") != FP16_SHA
        or score.get("stage") != "NRA2_S3V_PRECOMMITTED_S3T64_SOFT_VS_HARD_FROZEN_NATIVE_CPU_TRIAL"
        or score.get("S3T_prior_commitment_sha256") != S3T_SHA
        or score.get("S3U_successful_native_CPU_run") != S3U_RUN
        or score.get("S3T_successful_prereg_run") != S3T_RUN
        or score.get("S3O_frozen_adapter_sha256") != FP16_SHA
        or score.get("original_616_S194_J7_hashes_preserved_by_S3U") is not True
        or score.get("new_training_or_parameter_tuning") is not False
        or score.get("candidate_accepted") is not False
        or score.get("output_gate") != 0.0
        or score.get("oracle_true_graph_and_roles_correct") != N
        or len(rows) != N):
        raise AssertionError("S3W S3T/S3V provenance mismatch")
    original=score["all_and_family_yes_no_accuracy"]["all"]
    if original["count"]!=N or original["exact_predicted_graph"]!=N:
        raise AssertionError("S3W graph correctness mismatch")
    cases=score["disagreement_or_error_cases"]
    by_index={}
    for c in cases:
        i=c["index"]
        if (i<0 or i>=N or i in by_index or c["prompt_sha256"]!=rows[i].prompt_sha256
            or c["query_style"]!=rows[i].query_style
            or (c["soft_correct"] and c["hard_correct"])):
            raise AssertionError("S3W S3V error-case record drift")
        by_index[i]=c
    summary={}
    for i,r in enumerate(rows):
        rev=r.query_style in (0,3)
        key=("reversed" if rev else "forward")+"__style_"+str(r.query_style)
        s=summary.setdefault(key,{"count":0,"yes":0,"no":0,"soft_correct":0,
                                  "hard_correct":0,"correct_source":0,
                                  "correct_destination":0,"both_roles_wrong":0,
                                  "hard_false_positive":0,"hard_false_negative":0})
        d=by_index.get(i)
        soft_ok=d["soft_correct"] if d else True
        hard_ok=d["hard_correct"] if d else True
        src_ok=d["source_correct"] if d else True
        dst_ok=d["target_correct"] if d else True
        s["count"]+=1;s["yes"]+=int(r.reachable);s["no"]+=int(not r.reachable)
        s["soft_correct"]+=int(soft_ok);s["hard_correct"]+=int(hard_ok)
        s["correct_source"]+=int(src_ok);s["correct_destination"]+=int(dst_ok)
        s["both_roles_wrong"]+=int(not src_ok and not dst_ok)
        s["hard_false_positive"]+=int(not hard_ok and not r.reachable)
        s["hard_false_negative"]+=int(not hard_ok and r.reachable)
    total={k:sum(s[k] for s in summary.values()) for k in next(iter(summary.values()))}
    for k in ("count","yes","no","soft_correct","hard_correct",
              "correct_source","correct_destination","hard_false_positive","hard_false_negative"):
        if total[k]!=original[k]:
            raise AssertionError("S3W per-style original score discrepancy: "+k)
    if len(summary)!=4 or any(s["count"]!=16 or s["yes"]!=8 or s["no"]!=8 for s in summary.values()):
        raise AssertionError("S3W S3T four grammar families not balanced")
    groups={}
    for name,styles in (("forward_order",(1,2)),("reversed_mention_order",(0,3))):
        groups[name]={k:sum(summary[("reversed" if st in (0,3) else "forward")+"__style_"+str(st)][k] for st in styles)
                      for k in total}
    result={
        "stage":"NRA2_S3W_POST_HOC_S3V_NOVEL_GRAMMAR_ROLE_DIAGNOSIS_ONLY",
        "S3V_original_successful_run":S3V_RUN,
        "S3U_original_CPU_successful_run":S3U_RUN,
        "S3T_original_prereg_successful_run":S3T_RUN,
        "S3T_prior_commitment_sha256":S3T_SHA,
        "S3V_original_overall_immutable":original,
        "per_locked_query_style":summary,
        "by_textual_mention_order":groups,
        "predeclared_threshold_unchanged":0.5,
        "all_616_original_J7_preserved":True,
        "physically_frozen_graph_S3O_NRA_unchanged":True,
        "original_real_CPU_graphs_exact":N,
        "new_training":False,"post_hoc_diagnosis_only":True,
        "cannot_treat_this_as_unseen_blind_validation":True,
        "candidate_accepted":False,"output_gate":0.0,
        "decoder_full_EOS_or_speed_or_bpp_gain_tested":False,
        "LIVE_Render_Supabase_Jessica_QA_TEST_unchanged":True,
        "next_independent_stage_recommendation":"Precommit new truly unseen grammar with balanced mention order; fix role-model semantic generalization only in isolated TRAIN, then test independently.",
    }
    out.mkdir(parents=True)
    (out/"NRA2_S3W_S3V_NOVEL_GRAMMAR_ROLE_ORDER_DIAGNOSTIC.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:result[k] for k in
         ("stage","per_locked_query_style","by_textual_mention_order",
          "candidate_accepted","output_gate")},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
