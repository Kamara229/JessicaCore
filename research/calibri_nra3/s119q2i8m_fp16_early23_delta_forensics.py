"""S119q2i8m: physically inspect S119q2i8k saved trained FP16 A/B delta.

Read-only original completed GitHub Actions artifact. No original 43M/135M
pretrained weights or model forwards. Detect whether teacher KL changes
actually persisted to physical fitted FP16 neural/GT sidecars even though
the 14 source-disjoint (now spent) DEV greedy native sequences are identical.
TRAIN and protected EVAL are not accessed; no promotion possible.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

STAGE="CALIBRI_S119Q2I8M_REAL_SAVED_FP16_EARLY23KL_VS_CE_WITH_NATIVE_GREEDY_AUDIT"
PREV_STAGE="CALIBRI_S119Q2I8K_REAL43M_21NEW_MULTIWORD_TRAIN_CE_VS_135M_KL_EXISTING_NRA3_NPA1_GT12"
CE="CE_ONLY"
KL="SOURCE135M_SECONDTHIRD_BPE_KL"
EOS="research.s119q2i2.fullGT12_gates_fp16"
PREFIX="model.nra3_calibri_joint.s119q2b."

def check(ok,msg):
    if not ok:raise ValueError("S119q2i8m hard gate: "+msg)

def group(name):
    if ".language_head." in name:
        return "native_49152_readout"
    if "planning_action_area" in name:
        return "NPA1_planning"
    if "structured_interface" in name:
        return "NRA3_structured"
    if "core_growth" in name:
        return "Core_growth"
    if "nra3_neural_weights" in name:
        return "S115_protected_neural_state"
    if "projection" in name or ".up.weight" in name:
        return "other_projection"
    return "other_native_child"

def quantized_diff(a,b):
    check(set(a)==set(b) and bool(a),"FP16 A/B tensor key parity")
    blocks={};changed=0;total=0;total_squared=0.;max_delta=0.;top=[]
    for k in sorted(a):
        x=a[k];y=b[k]
        check(isinstance(x,np.ndarray) and isinstance(y,np.ndarray)
              and x.dtype==np.float16 and y.dtype==np.float16
              and x.shape==y.shape and np.isfinite(x).all()
              and np.isfinite(y).all(),"only exact finite physical FP16 paired arrays")
        diff=x.astype(np.float32)-y.astype(np.float32)
        nonzero=int(np.count_nonzero(diff))
        sq=float(np.dot(diff.reshape(-1),diff.reshape(-1)))
        mx=float(np.max(np.abs(diff))) if diff.size else 0.
        area=group(k)
        slot=blocks.setdefault(area,{"total_fp16_parameters":0,
                                     "parameters_different":0,
                                     "L2_squared":0.})
        slot["total_fp16_parameters"]+=int(x.size)
        slot["parameters_different"]+=nonzero
        slot["L2_squared"]+=sq
        total+=int(x.size);changed+=nonzero;total_squared+=sq
        max_delta=max(max_delta,mx)
        if nonzero:top.append({"tensor":k,"changed_fp16":nonzero,
                               "L2_distance":round(sq**0.5,8)})
    for x in blocks.values():
        x["L2_distance"]=round(x.pop("L2_squared")**0.5,8)
    return {"all_saved_params":total,"changed_fp16_params":changed,
            "fraction_saved_fp16_changed":round(changed/total,8),
            "delta_L2":round(total_squared**0.5,8),
            "max_absolute_fp16_difference":round(max_delta,8),
            "regions":blocks,
            "top_8_most_changed_saved_tensors":sorted(
                top,key=lambda z:z["L2_distance"],reverse=True)[:8]}

def checked_safetensors(base,name,meta,*,gate=False):
    path=base/name
    check(path.exists() and path.stat().st_size==meta["file_bytes"]
          and hashlib.sha256(path.read_bytes()).hexdigest()==meta["sha256"],
          "exact completed original43M trained FP16 SHA and byte count")
    d=load_file(str(path))
    if gate:
        check(set(d)=={EOS} and d[EOS].shape==(12,)
              and d[EOS].dtype==np.float16,"only 12 physical GT12 gate scalars")
    else:
        check(sum(x.size for x in d.values())==
              meta["saved_total_child_FP16_params"]==421793
              and all(k.startswith(PREFIX) for k in d),
              "421793 genuine fitted child neural saved FP16 not proxy")
    return d

def run(report,artifact_dir):
    check(report["stage"]==PREV_STAGE and
          report.get("candidate_accepted") is False
          and report.get("output_gate")==0,
          "only completed research 43M gate0 artifacts, not production")
    train=report["physical_genuine_training_A_B"]
    check(set(train)=={CE,KL}
          and train[CE]["real_all_neural_and_gt_Adam_updates"]==
              train[KL]["real_all_neural_and_gt_Adam_updates"]==27
          and train[CE]["physical_SOURCE135M_native_forward_calls_at_gold_secondthird_BPE_ONLY"]==0
          and train[KL]["physical_SOURCE135M_native_forward_calls_at_gold_secondthird_BPE_ONLY"]==53,
          "exact same-budget 27 steps + 53 teacher source native positions")
    saved=report["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"]
    expect={"CE_ONLY_neural","SOURCE135M_KL_neural","CE_ONLY_GT12","SOURCE135M_KL_GT12"}
    check(set(saved)==expect,"original A/B neural and GT12 physical sidecars required")
    neural_ce=checked_safetensors(artifact_dir,
        "S119Q2I8K_CE_ONLY_ORIGINAL43M_JOINT_NEURAL_FP16_G0.safetensors",
        saved["CE_ONLY_neural"])
    neural_kl=checked_safetensors(artifact_dir,
        "S119Q2I8K_SOFT135M_EARLY23KL_ORIGINAL43M_JOINT_NEURAL_FP16_G0.safetensors",
        saved["SOURCE135M_KL_neural"])
    gt_ce=checked_safetensors(artifact_dir,
        "S119Q2I8K_CE_ONLY_ORIGINAL43M_GT12_FP16_G0.safetensors",
        saved["CE_ONLY_GT12"],gate=True)
    gt_kl=checked_safetensors(artifact_dir,
        "S119Q2I8K_SOFT135M_EARLY23KL_ORIGINAL43M_GT12_FP16_G0.safetensors",
        saved["SOURCE135M_KL_GT12"],gate=True)
    neural=quantized_diff(neural_ce,neural_kl)
    gates=quantized_diff(gt_ce,gt_kl)
    check(neural["regions"].get("S115_protected_neural_state",{}).get(
          "parameters_different",0)==0,
          "original S115 protected 53751 neural FP16 immutable across A/B")
    check(neural["all_saved_params"]==421793
          and gates["all_saved_params"]==12,
          "full original physical research neural FP16/12 gates actually compared")
    dev=report["physical_OWN_sixdomain_new_and_old_DEV"]
    identity={}
    for split,needed in (("NEW_DEV",14),("OLD_DEV",6)):
        a=dev[CE][split];b=dev[KL][split]
        check(len(a)==len(b)==needed and
              {x["source_id"] for x in a}=={x["source_id"] for x in b},
              "actual paired DEV generated native source IDs")
        index={x["source_id"]:x for x in b}
        identical=sum(x["generated_own_native_BPE_ids"]==
                      index[x["source_id"]]["generated_own_native_BPE_ids"]
                      for x in a)
        identity[split]={"identical_CE_KL_own_full_native_BPE_traces":identical,
                         "different_CE_KL_own_full_native_BPE_traces":needed-identical,
                         "source_count":needed}
    return {
        "stage":STAGE,"source":PREV_STAGE,
        "actual_original_original43M_unmodified":True,
        "genuine_FP16_neural_CE_vs_teacherKL":neural,
        "genuine_FP16_GT12_gates_CE_vs_teacherKL":gates,
        "true_native_49152_SELF_DEV_CE_vs_KL":identity,
        "teacher_source135M_actual_training_logit_calls":53,
        "interpretation":"Actual saved FP16 numeric effect of early-2nd3rd source135M KL, not a quality gain; new 14 author synthetic DEV already spent",
        "no_future_sealed_or_protected_gold_read":True,
        "no_model_predict_or_backward_recalculated":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"REAL_KL_WEIGHT_CHANGE_NOT_QUALITY_VALIDATION_BLOCK_ANY_PROMOTION"
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--artifact-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    check(a.input.exists() and a.artifact_dir.is_dir() and
          not a.output.exists(),"existing exact physical report, no output overwrite")
    report=json.loads(a.input.read_text(encoding="utf8"))
    out=run(report,a.artifact_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,ensure_ascii=False,indent=2)+"\n")
    print("S119Q2I8M_PHYSICAL_SAVED_FP16_DELTA "+json.dumps({
        "neural_changes":out["genuine_FP16_neural_CE_vs_teacherKL"]["changed_fp16_params"],
        "neural_total":out["genuine_FP16_neural_CE_vs_teacherKL"]["all_saved_params"],
        "neural_norm":out["genuine_FP16_neural_CE_vs_teacherKL"]["delta_L2"],
        "GT12_changes":out["genuine_FP16_GT12_gates_CE_vs_teacherKL"]["changed_fp16_params"],
        "native_own_traces":out["true_native_49152_SELF_DEV_CE_vs_KL"],
        "candidate_accepted":False
    },ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
