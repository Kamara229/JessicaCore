"""S119q2i8s — forensic read-only real native head FP16 after #954.

No pretrained original 43M model loaded, no new predictions/optimizer and no
gold answer text beyond read-only completed research observations.
Physically compare SHA-verified 421793 FP16 parameter states:
  K frozen parent vs learned CE-only (rank4 full49k gain1) vs
  same-train-batch learned CE+signed TRAIN-native EOS ranking margin.
The 14 R author DEV were scored already in #954: not a new blind study.
No model selection or promotion; compare real head numeric motion with the
first-BPE/EOS/self-generated outcomes from prior complete experiment.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from research.calibri_nra3.s119q2i8m_fp16_early23_delta_forensics import (
    quantized_diff,checked_safetensors,
)
# Pin the *completed* research report ABI; do not import heavyweight
# torch/transformers model-training modules in this read-only forensic.
R_STAGE="CALIBRI_S119Q2I8R_REAL43M_34_UNUSED_TRAIN_RANK4_HEAD_EOS_MARGIN_CAUSAL_AB"
R_REPORT="S119Q2I8R_REAL43M_RANK4_HEAD_ONLY_GAIN1_EOS_CE_VS_MARGIN_G0.json"
CE_FP16="S119Q2I8R_REAL43M_NATIVE_HEAD_GAIN1_CE_ONLY_FP16_G0.safetensors"
MARGIN_FP16="S119Q2I8R_REAL43M_NATIVE_HEAD_GAIN1_CE_PLUS_EOS_MARGIN_FP16_G0.safetensors"
R_ARMS=("BASELINE_CE_ONLY","TRAIN_ONLY_SUPERVISED_EOS_MARGIN")
K_STAGE="CALIBRI_S119Q2I8K_REAL43M_FRESH_NEXT21_MULTIWORD_CE_VS_SOURCE135M_KL_2ND3RD_BPE"
K_REPORT="S119Q2I8K_REAL43M_EARLY23BPE_TEACHERKL_VS_CEONLY_REPORT.json"
KL_FP16="S119Q2I8K_SOFT135M_EARLY23KL_ORIGINAL43M_JOINT_NEURAL_FP16_G0.safetensors"

STAGE="CALIBRI_S119Q2I8S_TRUE_FP16_HEAD_ONLY_EOS_MARGIN_VS_CE_PAIRED_AUDIT"
CHILD=421793
HEAD=247040
EOS=2
VOCAB=49152
PREFIX="model.nra3_calibri_joint.s119q2b."
SPLITS=(("NEW_DEV",14),("OLD_DEV",6))
SAMPLES=("OLD_KL_PREVIOUS_GAIN_0p25","OLD_KL_PREVIOUS_GAIN_1p0",
         "CE_ONLY_FITTED_GAIN_1p0","SUPERVISED_EOS_MARGIN_GAIN_1p0")


def check(ok,msg):
    if not ok:raise ValueError("S119q2i8s fail-closed: "+msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readout(d):
    result={}
    for part,suffix,shape in (
        ("bias",".language_head.output_bias",(VOCAB,)),
        ("up",".language_head.up.weight",(VOCAB,4)),
        ("down",".language_head.down.weight",(4,320)),
    ):
        ks=[k for k in d if k.endswith(suffix)]
        check(len(ks)==1 and ks[0].startswith(PREFIX)
              and d[ks[0]].shape==shape
              and d[ks[0]].dtype==np.float16
              and bool(np.isfinite(d[ks[0]]).all()),
              "exact original native physical 49k lowrank4 "+part)
        result[part]=d[ks[0]].astype(np.float32)
    return result


def head_numeric_state(d):
    x=readout(d)
    bias=x["bias"];up=x["up"]
    nonEOS=np.concatenate((bias[:EOS],bias[EOS+1:]))
    return {
        "native_vocab":VOCAB,
        "current_real_EOS_ID":EOS,
        "real_saved_output_bias_EOS":round(float(bias[EOS]),8),
        "output_bias_EOS_rank_descending":1+int(np.count_nonzero(bias>bias[EOS])),
        "output_bias_EOS_minus_median_nonEOS":round(float(bias[EOS]-np.median(nonEOS)),8),
        "true_native_lowrank4_EOS_up_norm":round(float(np.linalg.norm(up[EOS])),8),
        "true_native_lowrank4_median_nonEOS_up_norm":round(float(
            np.median(np.linalg.norm(np.concatenate((up[:EOS],up[EOS+1:])),
                                     axis=1))),8),
        "physical_rank4_down_frobenius_norm":round(float(np.linalg.norm(x["down"])),8),
        "physical_rank4_up_frobenius_norm":round(float(np.linalg.norm(up)),8),
    }


def paired_by_source(arm_a,arm_b,split,expected):
    a=arm_a[split]["OWN_samples"];b=arm_b[split]["OWN_samples"]
    check(len(a)==len(b)==expected and
          len(set(x["source_id"] for x in a))==expected and
          set(x["source_id"] for x in a)==set(x["source_id"] for x in b),
          "matched exactly "+str(expected)+" source-disjoint own native DEV")
    mapped={r["source_id"]:r for r in b}
    for z in a+b:
        check(z["source_internal_DEV_NOT_independent_blind"] is True
              and z["generated_own_native_BPE_ids"]
              and all(type(t)==int and 0<=t<VOCAB for t in
                      z["generated_own_native_BPE_ids"])
              and type(z["true_native_full_EOS_correct"]) is bool
              and type(z["first_gold_BPE_correct"]) is bool,
              "read-only genuine original 49152 own BPE result, no synthetic string proxy")
    return {
        "matched_sources":expected,
        "identical_full_native_BPE_sequences":sum(
            z["generated_own_native_BPE_ids"]==
            mapped[z["source_id"]]["generated_own_native_BPE_ids"]
            for z in a),
        "different_full_native_BPE_sequences":sum(
            z["generated_own_native_BPE_ids"]!=
            mapped[z["source_id"]]["generated_own_native_BPE_ids"]
            for z in a),
        "A_completed_full_EOS_correct":sum(z["true_native_full_EOS_correct"] for z in a),
        "B_completed_full_EOS_correct":sum(z["true_native_full_EOS_correct"] for z in b),
        "A_correct_first_BPE":sum(z["first_gold_BPE_correct"] for z in a),
        "B_correct_first_BPE":sum(z["first_gold_BPE_correct"] for z in b),
        "first_differing_native_position_1based":{
            z["source_id"]:next((i+1 for i,(t,u) in
                enumerate(zip(z["generated_own_native_BPE_ids"],
                              mapped[z["source_id"]]["generated_own_native_BPE_ids"]))
                if t!=u),
                min(len(z["generated_own_native_BPE_ids"]),
                    len(mapped[z["source_id"]]["generated_own_native_BPE_ids"]))+1)
            for z in a if z["generated_own_native_BPE_ids"]!=
                          mapped[z["source_id"]]["generated_own_native_BPE_ids"]
        },
        "not_an_independent_DEV_and_NO_model_selection":True,
    }


def run(previous_report,previous_dir,report,artifact_dir):
    check(previous_report["stage"]==K_STAGE
          and report["stage"]==R_STAGE
          and previous_report["output_gate"]==report["output_gate"]==0
          and previous_report["candidate_accepted"] is False
          and report["candidate_accepted"] is False
          and previous_report["original_S142_43M_sha256"]==
              report["original_S142_43M_sha256"]
          and previous_report["original_S93_GT_sha256"]==
              report["original_S93_GT_sha256"],
          "only exact completed genuine nonproduction original43M research")
    check(report["physical_frozen_S119q2i8k_KL_source_report_SHA256"]==
          sha(previous_dir/K_REPORT)
          and report["existing_S119q2i8k_fit_FP16_neural_421793_sha256"]==
          sha(previous_dir/KL_FP16),
          "exact SHA-pinned original 421793 fitted physical FP16 source K child")
    stage=report["matched_train_head_only_research_arms"]
    check(set(stage)==set(R_ARMS) and
          all(z["actual_matched_native_rank4_head_only_Adam_updates"]==34
              and z["trainable_parameters_original_native_head_ONLY"]==HEAD
              and z["NRA3_NPA1_original43M_GT12_unchanged"] is True
              and z["old_sixslot_mix_logits_36param_gradient_anchor_NOT_in_optimizer"] is True
              for z in stage.values())
          and report["physical_true43M_full_NATIVE_train_forwards"]==1632
          and report["all_original_S142_GT12_NRA3_NPA1_and_S115_unchanged"] is True,
          "only 34 exact A/B REAL original43M native49k HEAD updates with same physical budget")
    names=report["saved_FULL_FP16_neural_children_old_GT12_UNCHANGED_gate0"]
    check(set(names)=={"CE_HEAD_ONLY_FP16","EOS_MARGIN_HEAD_ONLY_FP16"},
          "two original fully trained physical A/B FP16 children required")
    prior=previous_report["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"]
    parent=checked_safetensors(previous_dir,KL_FP16,
                               prior["SOURCE135M_KL_neural"])
    ce=checked_safetensors(artifact_dir,CE_FP16,names["CE_HEAD_ONLY_FP16"])
    margin=checked_safetensors(artifact_dir,MARGIN_FP16,names["EOS_MARGIN_HEAD_ONLY_FP16"])
    tensors={"PARENT_KL":parent,"CE_ONLY":ce,"EOS_MARGIN":margin}
    physical={k:head_numeric_state(v) for k,v in tensors.items()}
    diff={}
    for a,b in (("PARENT_KL","CE_ONLY"),("PARENT_KL","EOS_MARGIN"),
                ("CE_ONLY","EOS_MARGIN")):
        z=quantized_diff(tensors[a],tensors[b])
        h=z["regions"].get("native_49152_readout")
        check(z["all_saved_params"]==CHILD and h is not None
              and h["total_fp16_parameters"]==HEAD
              and h["parameters_different"]>0,
              "all physical FP16 changes must originate in the existing 247040-param rank4 head")
        check(sum(t["parameters_different"] for name,t in z["regions"].items()
                  if name!="native_49152_readout")==0
              and z["changed_fp16_params"]==h["parameters_different"],
              "ABSOLUTELY NO change Core12 GT12 NRA3/NPA1 prior neural after head training")
        diff[a+"__VS__"+b]=z
    observations=report["fresh_AUTHOR14_DEV_plus_old6_actual_OWN_native_self_generated"]
    check(set(observations)==set(SAMPLES),"all four physical original decoder arms")
    scores={}
    for arm in SAMPLES:
        scores[arm]={}
        for split,n in SPLITS:
            x=observations[arm][split]["scores"]
            check(x["DEV_n"]==n and
                  x["correct_full_native_ANSWER_and_EOS"]==0
                  and x["real_original43M_OWN_forwards"]>=n,
                  "no invented quality gains, preserve completed R physical source")
            scores[arm][split]=x
    pairs={
        split:{
            "CEvsEOS_MARGIN_gain1":
                paired_by_source(observations["CE_ONLY_FITTED_GAIN_1p0"],
                                 observations["SUPERVISED_EOS_MARGIN_GAIN_1p0"],split,n),
            "historical_KL_gain0p25_vs_CE_gain1":
                paired_by_source(observations["OLD_KL_PREVIOUS_GAIN_0p25"],
                                 observations["CE_ONLY_FITTED_GAIN_1p0"],split,n),
        }
        for split,n in SPLITS
    }
    return {
        "stage":STAGE,
        "complete_real_actions_source_S119q2i8r":R_STAGE,
        "previous_source_S119q2i8k_KL_FP16_SHA256":sha(previous_dir/KL_FP16),
        "trained_S119q2i8r_original43M_headOnly_FP16s":
            {k:v["sha256"] for k,v in names.items()},
        "physical_true_FP16_421793_saved_neural_values_comparisons":diff,
        "genuine_saved_original_rank4_49152_head_EOS_bias_and_U_norm":physical,
        "matched_34TRAIN_human_gold_CE_only_vs_EOS_margin":{
            a:{"online_running_TRAIN_mean_native_CE_NOT_final_validation":
                    z["mean_full_native_human_gold_CE"],
               "online_running_TRAIN_mean_true_terminal_EOS_CE_NOT_holdout":
                    z["mean_true_terminal_EOS_CE"],
               "TRAIN_only_margin_loss_calls":z["real_additional_train_only_margin_loss_calls"],
               "physical_native49k_forwards":z["physical_43M_full_native49152_teacher_forced_TRAIN_forwards"]}
            for a,z in stage.items()},
        "actual_own_BPE_nonblind_DEV_scores_readonly":scores,
        "per_source_paired_own_native_dev_NOT_NEW_BLIND":pairs,
        "original43M_native_forwards_in_THIS_forensic":0,
        "NO_pretrained_models_no_optimizer_no_forward_no_616J7_EVAL128":True,
        "all_original_core_NRA3_NPA1_GT12_and_S115_protected_unchanged":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"EOS_MARGIN_34TRAIN_RANK4_ONLY_NOT_VALIDATED_NO_PROMOTION"
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--parent-dir",type=Path,required=True)
    p.add_argument("--trained-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    check(not a.out.exists() and a.parent_dir.is_dir()
          and a.trained_dir.is_dir(),"exact source artifacts & no overwrite")
    previous=json.loads((a.parent_dir/K_REPORT).read_text(encoding="utf8"))
    report=json.loads((a.trained_dir/R_REPORT).read_text(encoding="utf8"))
    result=run(previous,a.parent_dir,report,a.trained_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8S_HEAD_ONLY_PHYSICAL_FP16_EOS_PAIRED "+json.dumps({
        "saved_head_EOS_stats":result["genuine_saved_original_rank4_49152_head_EOS_bias_and_U_norm"],
        "number_changed_FP16":{k:v["changed_fp16_params"] for k,v in
            result["physical_true_FP16_421793_saved_neural_values_comparisons"].items()},
        "nonhead_FP16_absolute_immutability":True,
        "paired_own_native":result["per_source_paired_own_native_dev_NOT_NEW_BLIND"],
        "candidate_accepted":False,
    },sort_keys=True,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
