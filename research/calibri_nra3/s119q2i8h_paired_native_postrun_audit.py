"""S119q2i8h: fail-closed paired post-training research evidence auditor.

Consumes ONLY completed S119q2i8g JSON report. Never loads pretrained
models, optimizer tensors, secrets, older protected 616-J7 answers, or DEV
target text for training. The 14 inspectable newly authored DEV prompts are
already spent once scored: this report is descriptive, NOT an independent
blind test nor selection of a production model. No acceptance allowed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

STAGE="CALIBRI_S119Q2I8H_RESEARCH_ONLY_PAIRED_OWN_NATIVE_POSTRUN_AUDIT"
EXPECTED_STAGE="CALIBRI_S119Q2I8G_REAL43M_21NEW_MULTIWORD_TRAIN_CE_VS_135M_KL_EXISTING_NRA3_NPA1_GT12"
BASE="PRIOR_Q2I8A_FROZEN_PARENT"  # historical key, physically S119q2i8d CE parent
A="CE_ONLY"
B="SOURCE135M_UTF8_CONTINUATION_KL"
ARMS=(BASE,A,B)
SPLITS=(("NEW_DEV",14),("OLD_DEV",6))
FLAGS=(
 "all_trained_rows_are_SOURCE_SELECTED_21_NEW_TRAIN_plus_6_old_TRAIN",
 "initial_research_child_Q2I8d_CE_ONLY_warmstart_not_prior_q2i8a",
 "no_original_q2i8a_or_other_previous_DEV_in_teacher_or_Adam",
 "two_matched_368051_param_GT12_Core_NRA3_NPA1_languagehead_optimizer_arms",
 "matched_current_architecture_NO_new_neural_areas",
 "true_full_native_49152_ce_at_EVERY_BPE_including_correct_EOS",
 "SOURCE135M_SOFT_KL_not_applied_to_EOS_or_other_native_positions",
 "existing_old_S115_53751_neural_FP16_preserved",
 "original_S142_and_teacher135m_UNCHANGED",
 "source_teacher_KL_never_on_native_EOS",
 "new_DEV_is_small_VISIBLE_internal_NOT_external_blind",
 "new_DEV_14_source_disjoint_new_training_multiword_prompts",
 "no_full_protected_616_J7_EOS_or_all_in_packed_bpp_KV_CPU",
 "no_S118_GOLD_EVAL128_J7_answers_or_Jessica_chat",
 "no_LIVE_main_Render_Supabase_APK_or_secrets",
)
OBS={
 "full_correct":"true_native_full_EOS_correct",
 "valid_utf8":"valid_full_UTF8_bytes",
 "native_eos":"native_EOS",
 "wrong_second_after_correct_first":"first_native_correct_but_wrong_second",
 "gold_first_correct":"first_gold_BPE_correct",
 "text_exact_no_eos_requirement":"text_exact_even_if_not_finished",
 "eos_after_invalid_utf8":"real_EOS_but_UNFINISHED_or_INVALID_UTF8",
}

def require(ok,detail):
    if not ok:
        raise ValueError("S119q2i8h fail-closed: "+detail)

def wilson_95(k,n):
    require(type(k)==int and type(n)==int and 0<=k<=n and n>0,
            "invalid Wilson k,n")
    z=1.959963984540054
    p=k/n
    denom=1+(z*z/n)
    centre=(p+z*z/(2*n))/denom
    half=(z/denom)*math.sqrt((p*(1-p)/n)+z*z/(4*n*n))
    return [round(max(0.,centre-half),6),round(min(1.,centre+half),6)]

def score_records(records,expected):
    require(len(records)==expected,"wrong whole DEV size")
    d={}
    for name,key in OBS.items():
        require(all(type(row.get(key)) is bool for row in records),
                "missing true boolean native observation "+key)
        d[name]=sum(row[key] for row in records)
    d["source_count"]=expected
    d["exact_whole_answer_wilson_95"]=wilson_95(d["full_correct"],expected)
    d["valid_utf8_wilson_95"]=wilson_95(d["valid_utf8"],expected)
    d["real_native_forward_calls"]=sum(
        int(x["original_full_decoder_calls"]) for x in records)
    return d

def paired_ab(a,b,key):
    require(len(a)==len(b),"missing paired cases")
    return {
        "both_pass":sum(x[key] and y[key] for x,y in zip(a,b)),
        "both_fail":sum(not x[key] and not y[key] for x,y in zip(a,b)),
        "CE_only_pass_KL_fail":sum(x[key] and not y[key] for x,y in zip(a,b)),
        "CE_only_fail_KL_pass":sum(not x[key] and y[key] for x,y in zip(a,b)),
        "no_hypothesis_or_promotion_from_tiny_visible_internal_DEV":True,
    }

def analyse(report):
    from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import verify as train_source
    from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import (
        rows as new_dev_rows,verify as new_source,
    )
    from research.calibri_nra3.s119q2i1_authored_ru_train_source import rows as old_source
    require(report.get("stage")==EXPECTED_STAGE,"wrong source training experiment")
    require(report.get("candidate_accepted") is False
            and report.get("output_gate")==0,"candidate must NEVER be promoted")
    require(all(report.get(k) is True for k in FLAGS),
            "one or more protected split/weight/architecture flags false")
    provenance=train_source()
    newer=new_source()
    require(report.get("registered_70_new_multisentence_TRAIN_source_sha")==
            provenance["TRAIN_sha256"],"registered 70 TRAIN source changed")
    require(report.get("registered_new_14_original_author_DEV_SHA")==
            newer["source_new_DEV_sha256"],"NEW 14 DEV source hash changed")
    require(report.get("only_NEW_russian_DEV_never_train_size")==14,
            "must record exactly 14 new DEV")
    require(report.get("teacher_temperature_T2")==2.0 and
            report.get("teacher_full_native_49152_KL_weight_at_UTF8_continuation_ONLY")==0.25,
            "wrong prespecified KL objective")
    training=report["physical_genuine_training_A_B"]
    require(set(training)=={A,B},"wrong paired optimizer arms")
    first=training[A];second=training[B]
    require(first["real_all_neural_and_gt_Adam_updates"]==
            second["real_all_neural_and_gt_Adam_updates"]==27,
            "matched 21+6 physical Adam steps per arm")
    require(first["trainable_parameter_count"]==
            second["trainable_parameter_count"]==368051,
            "matched full neural + GT12 budget")
    require(first["trained_original_decoder_calls"]==
            second["trained_original_decoder_calls"]>27,
            "same physical ORIGINAL 43M full 49k passes")
    require(first["physical_SOURCE135M_native_forward_calls_at_incomplete_prefix_only"]==0
            and second["physical_SOURCE135M_native_forward_calls_at_incomplete_prefix_only"]>0,
            "source 135M must be used only on KL arm")
    ev=report["physical_OWN_sixdomain_new_and_old_DEV"]
    require(set(ev)==set(ARMS),"require frozen parent + two trained paired children")
    expected={
        "NEW_DEV":{x["id"] for x in new_dev_rows()},
        "OLD_DEV":{x["id"] for x in old_source("DEV") if x["id"].endswith("_06")},
    }
    require(len(expected["NEW_DEV"])==14 and len(expected["OLD_DEV"])==6,
            "DEV provenance mismatch")
    totals={};paired={}
    for split,n in SPLITS:
        records={label:ev[label][split] for label in ARMS}
        for label,rs in records.items():
            require(len(rs)==n and len({x["source_id"] for x in rs})==n
                    and {x["source_id"] for x in rs}==expected[split],
                    "missing/duplicated/unknown "+split+" source IDs in "+label)
            require(all(x["teacher_forcing_in_DEV"] is False
                    and x["original_full_decoder_calls"]==
                    len(x["generated_own_native_BPE_ids"])
                    and 1<=x["original_full_decoder_calls"]<=40
                    for x in rs),"never teacher-forced generated DEV")
            scores=score_records(rs,n)
            original=ev[label][split+"_scores"]
            for display,original_key in (
                ("full_correct","correct_full_native_ANSWER_and_EOS"),
                ("valid_utf8","full_UTF8_valid_own_native"),
                ("native_eos","true_native_EOS_emitted"),
                ("gold_first_correct","first_native_gold_BPE_correct"),
                ("wrong_second_after_correct_first","first_native_correct_but_wrong_second"),
                ("eos_after_invalid_utf8","native_EOS_after_incomplete_or_INVALID_UTF8"),
                ("real_native_forward_calls","original_43M_Core12_native_decoder_forwards")
            ):
                require(scores[display]==original[original_key],
                        "mismatch physical per-sample vs aggregate "+display)
            totals.setdefault(split,{})[label]=scores
        a={r["source_id"]:r for r in records[A]}
        b={r["source_id"]:r for r in records[B]}
        aa=[a[k] for k in sorted(expected[split])]
        bb=[b[k] for k in sorted(expected[split])]
        paired[split]={
            metric:paired_ab(aa,bb,OBS[metric])
            for metric in ("full_correct","valid_utf8","native_eos",
                           "gold_first_correct","text_exact_no_eos_requirement")
        }
    return {
        "stage":STAGE,
        "source_stage":EXPECTED_STAGE,
        "real_43M_core_and_GT12_native_weights_reused":True,
        "completed_original43M_A_B_training_same_exact_budget":True,
        "trained_original43M_native_BPE_forwards_each":
            first["trained_original_decoder_calls"],
        "physically_measured_original135M_source_KL_train_forwards":
            second["physical_SOURCE135M_native_forward_calls_at_incomplete_prefix_only"],
        "metrics":totals,
        "paired_controlled_outcomes":paired,
        "interpretation":"descriptive paired internal source-disjoint DEV only; sample too small to prove general language competence",
        "DEV_already_spent_do_not_adapt_or_repeat_tune":True,
        "strict_full_616_J7_retention_not_tested":True,
        "all_in_packed_bpp_below_0p7_not_tested":True,
        "paired_efficient_KV_CPU_above_90pct_not_tested":True,
        "external_independently_blind_Russian_QA_not_tested":True,
        "candidate_accepted":False,
        "output_gate":0,
        "decision":"BLOCK_PROMOTION_UNTIL_PROTECTED_RETENTION_BPP_CPU_AND_INDEPENDENT_BLIND_EVAL"
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    require(args.input.exists() and not args.output.exists(),
            "source completed report required, never overwrite")
    original=json.loads(args.input.read_text(encoding="utf8"))
    result=analyse(original)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,
                                     indent=2)+"\n",encoding="utf8")
    print("S119Q2I8H_PAIRED_NATIVE_REPORT "+json.dumps({
        "new14":result["metrics"]["NEW_DEV"],
        "paired_new14":result["paired_controlled_outcomes"]["NEW_DEV"],
        "decision":result["decision"],
    },ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
