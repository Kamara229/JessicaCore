"""S119q2i8p: read-only *physical FP16* learned native-head EOS bias forensics.

Previous genuine original43M gain ablation #951 demonstrated 1.0 head gain
immediately produces EOS as FIRST token on all 14/14 spent small visible
synthetic DEV, while 0 gain never reaches EOS in 40 steps. This script
inspects EXACT SHA-verified prior #943 and #947 saved FP16 sidecars.
No original 43M/135M weights, model forwards, input prompts, DEV gold,
optimizer, protected J7, production services or auto-acceptance.

A fixed output_bias vector is only ONE PART of the learned correction:
learned low-rank U gelu(D layernorm(h)) is CONTEXT-DEPENDENT. Bias stats
cannot attribute exact 1.0-generation collapse without hidden-state
component attribution. This diagnosis neither tunes head nor ranks models.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

STAGE="CALIBRI_S119Q2I8P_TRUE_FP16_RANK4_NATIVE_HEAD_EOS_BIAS_FORENSIC"
G_STAGE="CALIBRI_S119Q2I8G_REAL43M_21NEW_MULTIWORD_TRAIN_CE_VS_135M_KL_EXISTING_NRA3_NPA1_GT12"
K_STAGE="CALIBRI_S119Q2I8K_REAL43M_FRESH_NEXT21_MULTIWORD_CE_VS_SOURCE135M_KL_2ND3RD_BPE"
VOCAB=49152
EOS=2
RANK=4
HIDDEN=320
FP16_CHILD=421793
PREFIX="model.nra3_calibri_joint.s119q2b."
SIDE={
 "Q2I8g_prior_CE":("g","CE_ONLY_neural","S119Q2I8G_CE_ONLY_ORIGINAL43M_JOINT_NEURAL_FP16_G0.safetensors"),
 "Q2I8k_CE":("k","CE_ONLY_neural","S119Q2I8K_CE_ONLY_ORIGINAL43M_JOINT_NEURAL_FP16_G0.safetensors"),
 "Q2I8k_early23_135M_KL":("k","SOURCE135M_KL_neural","S119Q2I8K_SOFT135M_EARLY23KL_ORIGINAL43M_JOINT_NEURAL_FP16_G0.safetensors"),
}
SUFFIX={
 "bias":".language_head.output_bias",
 "up":".language_head.up.weight",
 "down":".language_head.down.weight",
}


def guard(condition,msg):
    if not condition:
        raise ValueError("S119q2i8p fail-closed: "+msg)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def real_head_only(d):
    """Extract exactly complete genuine native head parameters by suffix."""
    guard(d and all(k.startswith(PREFIX) for k in d),
          "wrong physically persisted original research sidecar namespace")
    heads={}
    for name,suffix in SUFFIX.items():
        keys=[k for k in d if k.endswith(suffix)]
        guard(len(keys)==1,"exact native rank4 head tensor missing/ambiguous "+name)
        v=d[keys[0]]
        guard(v.dtype==np.float16 and bool(np.isfinite(v).all()),
              "only physical finite saved FP16 language head tensors")
        heads[name]=v.astype(np.float32)
    guard(heads["bias"].shape==(VOCAB,)
          and heads["up"].shape==(VOCAB,RANK)
          and heads["down"].shape==(RANK,HIDDEN),
          "true original trained 49152-vocab 4-rank 320-native head only")
    return heads


def head_report(d):
    """No real original model forward. This is WEIGHT-only, not logits."""
    h=real_head_only(d)
    bias=h["bias"]
    u=h["up"]
    eos=float(bias[EOS])
    other=bias[np.arange(VOCAB)!=EOS]
    norm=np.linalg.norm(u,axis=1)
    e_norm=float(norm[EOS])
    # Fixed-bias statistics are descriptive, not next-token ranks.
    return {
        "physical_real_output_bias_EOS_ID":EOS,
        "physical_native_vocab_size":VOCAB,
        "native_lm_head_rank":RANK,
        "bias_EOS_value_fp16_cast_fp32":round(eos,8),
        "bias_EOS_rank_among_49152_descending":1+int((bias>bias[EOS]).sum()),
        "bias_EOS_minus_max_nonEOS":round(eos-float(other.max()),8),
        "bias_EOS_minus_nonEOS_median":round(eos-float(np.median(other)),8),
        "bias_EOS_minus_nonEOS_mean":round(eos-float(other.mean()),8),
        "bias_nonEOS_std":round(float(other.std()),8),
        "bias_EOS_upper_percentile_of_all":round(100.*float((bias<=eos).mean()),7),
        "bias_nonEOS_absolute_max":round(float(np.abs(other).max()),8),
        "bias_EOS_effect_gain_0":0.,
        "bias_EOS_effect_gain_0p25":round(.25*eos,8),
        "bias_EOS_effect_gain_1":round(eos,8),
        "bias_EOS_effect_extra_gain1_minus_gain0p25":round(.75*eos,8),
        "native_up_vector_EOS_l2":round(e_norm,8),
        "native_up_vector_EOS_rank_by_l2_descending":
            1+int((norm>norm[EOS]).sum()),
        "native_up_l2_median_nonEOS":round(float(np.median(
            norm[np.arange(VOCAB)!=EOS])),8),
        "native_up_EOS_minus_median_nonEOS_l2":round(
            e_norm-float(np.median(norm[np.arange(VOCAB)!=EOS])),8),
        "native_down_weight_frobenius_l2":round(float(np.linalg.norm(h["down"])),8),
        "native_up_weight_frobenius_l2":round(float(np.linalg.norm(h["up"])),8),
        "output_bias_global_native_zero_output_bias":bool(np.all(bias==0)),
        "weight_only_CANNOT_EXPLAIN_full_hidden_dependent_logits_alone":True,
    }


def checked_full_fp16(path,metadata):
    guard(path.is_file() and path.stat().st_size==metadata["file_bytes"]
          and sha(path)==metadata["sha256"],
          "exact physical 421793 neural FP16 saved source SHA and byte-size")
    d=load_file(str(path))
    guard(sum(v.size for v in d.values())==
          metadata["saved_total_child_FP16_params"]==FP16_CHILD
          and all(v.dtype==np.float16 and np.isfinite(v).all()
                  for v in d.values()),
          "all original physical 421793 neural/region+native-head FP16")
    return d


def analyse(g_dir,k_dir):
    g_path=g_dir/"S119Q2I8G_REAL43M_NATIVE_BPE_TEACHERKL_UTF8_VS_CEONLY_REPORT.json"
    k_path=k_dir/"S119Q2I8K_REAL43M_EARLY23BPE_TEACHERKL_VS_CEONLY_REPORT.json"
    guard(g_path.is_file() and k_path.is_file(),"real completed physical research reports")
    g=json.loads(g_path.read_text(encoding="utf8"))
    k=json.loads(k_path.read_text(encoding="utf8"))
    guard(g.get("stage")==G_STAGE and k.get("stage")==K_STAGE
          and g.get("output_gate")==k.get("output_gate")==0
          and g.get("candidate_accepted") is False
          and k.get("candidate_accepted") is False
          and k["previous_physical_Q2I8g_parent_report_SHA256"]==sha(g_path)
          and k["original_S142_43M_sha256"]==g["original_S142_43M_sha256"]
          and k["original_S93_GT_sha256"]==g["original_S93_GT_sha256"],
          "matched genuine physically pretrained original 43M + previous CE baseline")
    meta={
        "g":g["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"],
        "k":k["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"],
    }
    results={}
    for name,(source,key,filename) in SIDE.items():
        path={"g":g_dir,"k":k_dir}[source]/filename
        d=checked_full_fp16(path,meta[source][key])
        results[name]={
            "physical_full_neural_fp16_sha256":sha(path),
            "physical_full_neural_saved_parameters":FP16_CHILD,
            **head_report(d),
        }
    return {
        "stage":STAGE,
        "physical_saved_original43M_research_FP16_head_only":True,
        "completed_Q2I8g_SHA256":sha(g_path),
        "completed_Q2I8k_SHA256":sha(k_path),
        "actual_named_physical_FP16_heads":results,
        "known_prior_physical_gain1_all_14_native_EOS_on_first_BPE":True,
        "known_prior_physical_gain0_NO_native_EOS_in_40_BPE":True,
        "known_prior_gain0p25_native_FIRST_BPE_matches_target_14_of14":True,
        "both_synthetic_14_DEV_and_616J7_gold_NOT_OPENED":True,
        "bias_only_does_NOT_measure_contextual_lowrank_path":True,
        "no_real_model_forward_no_optimizer_no_training_no_sealed_eval":True,
        "original_43M_and_prior_FP16_weights_not_modified":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"CONTINUE_HEAD_COMPONENT_CAUSAL_ATTRIBUTION_RESEARCH_ONLY"
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--q2i8g-dir",type=Path,required=True)
    p.add_argument("--q2i8k-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    guard(a.q2i8g_dir.is_dir() and a.q2i8k_dir.is_dir()
          and not a.out.exists(),"existing training artifacts; no overwrite")
    out=analyse(a.q2i8g_dir,a.q2i8k_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                     encoding="utf8")
    print("S119Q2I8P_REAL_FP16_NATIVE_HEAD_EOS_BIAS "+json.dumps({
        "true_existing_original_fp16":out["actual_named_physical_FP16_heads"],
        "candidate_accepted":False,
    },ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
