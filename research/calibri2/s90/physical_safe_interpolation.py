"""CALIBRI S90: bounded, research-only physically paid S87->S89 safety homotopy.

S89's actual MPS full-policy training made new strict full/EOS TRAIN answers
but lost already-correct TRAIN examples. Determine whether a *single*
global partial update of BOTH existing compressed dictionaries and internal
Green Tunnel can retain genuinely new complete answers without forgetting.

No new samples, no QA/TEST, no prompt/answer lookup, no model release or LIVE.
All candidates physically repacked and independently replayed on CPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import MODEL_SPEC, SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s87 import pilot_joint_core_tunnel as s87pilot
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s89 import train_joint_full_policy_quality as s89
from v185.codec import replace_discrete

ALPHAS=(0.125,0.25,0.375,0.50,0.75,1.0)
OPERATORS=(
    "model.layers.3.mlp.down_proj.weight",
    "model.layers.9.mlp.down_proj.weight"
)
CAP=0.7


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ensure_source_differences_are_paid(old_raw,new_raw):
    old,new=CB59(old_raw),CB59(new_raw)
    assert old.meta==new.meta and len(old_raw)==len(new_raw)
    changed=[]
    for name,desc in old.meta["tensor_descriptors"].items():
        if desc["dtype"]=="CB59_K32V8":
            a,b=old.handles[name],new.handles[name]
            if (not np.array_equal(a.packed,b.packed)
                    or not np.array_equal(a.row_scales,b.row_scales)):
                raise AssertionError("S89 source changed unpaid codes/scales: "+name)
            if not np.array_equal(a.centroids,b.centroids):
                changed.append(name)
        elif not np.array_equal(old.raw[name],new.raw[name]):
            raise AssertionError("S89 source changed frozen original 1D: "+name)
    if set(changed)!=set(OPERATORS):
        raise AssertionError("S90 expects exactly two changed paid core dictionaries")
    return old,new


def mix_sides(original,trained,alpha):
    if original.keys()!=trained.keys():
        raise AssertionError("S89 sidecar unexpectedly added or removed tensors")
    result={}
    changed=0
    for name,base in original.items():
        after=trained[name]
        if (base.shape!=after.shape or base.dtype!=after.dtype
                or not torch.isfinite(after).all()):
            raise AssertionError("S89 sidecar incompatible tensor "+name)
        if name.startswith("internal_tunnel."):
            blended=(base.float()+(after.float()-base.float())*alpha).to(base.dtype)
            changed+=int(not torch.equal(base,blended))
        else:
            if not torch.equal(base,after):
                raise AssertionError("S89 modified frozen S82 controller "+name)
            blended=base
        result[name]=blended.contiguous()
    if not changed:
        raise AssertionError("S90 alpha had no trainable tunnel delta")
    for name in result:
        if name.endswith(".gate.weight") or name.endswith(".gate.bias"):
            if not torch.count_nonzero(result[name])==0:
                raise AssertionError("S90 zero-gate S88 speed fastpath invalid")
    return result


def write_candidate(old_archive,new_archive,sources,old_side,trained_side,
                    alpha,out):
    for name in OPERATORS:
        before=old_archive.handles[name].centroids
        trained=new_archive.handles[name].centroids
        mixed=(before+(trained-before)*alpha).astype("<f2").astype(np.float32)
        with torch.no_grad():
            sources[name].centroids.copy_(
                torch.from_numpy(mixed).to(
                    device=sources[name].centroids.device,
                    dtype=sources[name].centroids.dtype))
    raw,mutation=replace_discrete(old_archive,sources,strict=False)
    if raw==old_archive.blob:
        raise AssertionError("S90 has no physical core delta at alpha "+str(alpha))
    packed=s87pilot.lossless_cb61(raw)
    side=mix_sides(old_side,trained_side,alpha)
    key=str(alpha).replace(".","p")
    source_path=out/("S90_ALPHA_"+key+"_CB61.s61")
    side_path=out/("S90_ALPHA_"+key+"_JOINT.safetensors")
    source_path.write_bytes(packed)
    save_file(side,str(side_path))
    if s87.open_envelope(source_path.read_bytes())!=raw:
        raise AssertionError("S90 physical source roundtrip failed")
    if set(side)!=set(load_file(str(side_path))):
        raise AssertionError("S90 physical tunnel roundtrip failed")
    return raw,packed,side,source_path,side_path,mutation


def entire_456(original_cb59,s82_side,cb61,side,manifest,out,old_456):
    # Reuse the existing independent S87 CPU checker (actual compact
    # reconstruction and 48-prompt speed), with mathematically exact
    # zero-gate S88 fastpath only on new model.
    original_builder=s87.build_cpu
    visited=[]
    def verified_builder(raw,weights,use_internal):
        state=original_builder(raw,weights,use_internal)
        if use_internal:
            visited.extend(install_verified_fastpath(state[-1]))
        return state
    s87.build_cpu=verified_builder
    argv=sys.argv
    thrown=None
    try:
        sys.argv=[argv[0],str(original_cb59),str(s82_side),
                  str(cb61),str(side),str(manifest),str(out)]
        try:
            s87.main()
        except SystemExit as exc:
            thrown=str(exc)
    finally:
        sys.argv=argv
        s87.build_cpu=original_builder
    file=out/"S87_ALL_456_TRAIN_GUARD_REPORT.json"
    if not file.exists():
        raise AssertionError("S90 did not produce full 456 independent CPU audit")
    report=json.loads(file.read_text())
    if visited!=["3","9"] or len(report["all_456_per_identity_results"])!=456:
        raise AssertionError("S90 fastpath/456 completeness violated")
    baseline={x["id"]:x for x in old_456["all_456_per_identity_results"]}
    seen=set()
    gains=[];losses=[];lost_first=[];untrained=[]
    target_set=set(json.loads(manifest.read_text())["S89_targeted_wrong_TRAIN_ids"])
    for new in report["all_456_per_identity_results"]:
        k=new["id"]
        before=baseline[k]
        if (k in seen or new["question_sha256"]!=before["question_sha256"]
                or new["target_ids"]!=before["target_ids"]
                or new["S82_ids"]!=before["S82_ids"]):
            raise AssertionError("S90 frozen TRAIN identity drift")
        seen.add(k)
        target=new["target_ids"]
        old_correct=before["S87_ids"]==target
        now_correct=new["S87_ids"]==target
        old_first=bool(before["S87_ids"]) and before["S87_ids"][0]==target[0]
        now_first=bool(new["S87_ids"]) and new["S87_ids"][0]==target[0]
        if old_correct and not now_correct: losses.append(k)
        if old_first and not now_first: lost_first.append(k)
        if not old_correct and now_correct:
            gains.append(k)
            if k not in target_set:untrained.append(k)
    if len(seen)!=456:raise AssertionError("S90 missing rows")
    summary={
        "stage":"S90_456_RECONSTRUCTED_PHYSICAL_CPU_STRICT_FULL_EOS_TRAIN",
        "S87_old_full":sum(x["S87_ids"]==x["target_ids"] for x in baseline.values()),
        "S90_new_full":sum(x["new_full"] for x in report["all_456_per_identity_results"]),
        "strict_new_full_and_EOS_gains":gains,
        "new_full_on_untrained_existing_TRAIN":untrained,
        "lost_old_full_ids":losses,
        "lost_old_first_ids":lost_first,
        "physically_paid_bpp":report["all_inclusive_bpp"],
        "reconstructed_pytorch_CPU_speed_ratio_vs_S82":
            report["cpu_speed_ratio_48_prompt_median_S82_over_S87"],
        "S88_exact_zero_gate_fastpath_verified":True,
        "all_456_no_forgetting":not losses and not lost_first,
        "acceptance_candidate_for_more_testing":not thrown and not losses
            and not lost_first and len(gains)>0 and report["cpu_speed_90pct_guard"]
            and report["all_inclusive_bpp"]<=CAP,
        "not_native_C_or_GPU_CPU_parity_or_independent_QA":True,
        "model_promoted":False,"S87_guard_exception":thrown
    }
    (out/"S90_ALL_456_FULL_EOS_RESULT.json").write_text(json.dumps(
        summary,indent=2))
    print(json.dumps({"event":"S90_FULL_456_RESULT",
        "gains":len(gains),"forgotten_full":len(losses),
        "forgotten_first":len(lost_first),
        "bpp":summary["physically_paid_bpp"],
        "CPU_ratio":summary["reconstructed_pytorch_CPU_speed_ratio_vs_S82"],
        "eligible_for_further_tests":summary["acceptance_candidate_for_more_testing"],
        "model_promoted":False}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    for arg in ("S87_CB61","S87_sidecar","S87_manifest",
                "S89_CB61","S89_sidecar","S89_manifest",
                "original_S45_CB59","original_S82_sidecar",
                "S87_frozen_456","out"):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    old_packed=a.S87_CB61.read_bytes()
    trained_packed=a.S89_CB61.read_bytes()
    old_meta=json.loads(a.S87_manifest.read_text())
    new_meta=json.loads(a.S89_manifest.read_text())
    reference=json.loads(a.S87_frozen_456.read_text())
    if (sha(old_packed)!=old_meta["source_CB61_physical_sha256"]
            or sha(trained_packed)!=new_meta["source_CB61_physical_sha256"]
            or sha(a.S87_sidecar.read_bytes())!=old_meta["prototype_sidecar_sha256"]
            or sha(a.S89_sidecar.read_bytes())!=new_meta["prototype_sidecar_sha256"]
            or sha(a.original_S82_sidecar.read_bytes())!=old_meta["old_S82_sidecar_sha256"]
            or not reference["full_safety_guard"]
            or reference["source_new_cb59_sha256"]!=old_meta["source_sha256"]
            or new_meta["candidate_accepted"] is not False):
        raise AssertionError("S90 immutable S82/S87/S89 provenance mismatch")
    old_raw=s87.open_envelope(old_packed)
    trained_raw=s87.open_envelope(trained_packed)
    old_archive,new_archive=ensure_source_differences_are_paid(old_raw,trained_raw)
    old_side=load_file(str(a.S87_sidecar))
    trained_side=load_file(str(a.S89_sidecar))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    rows,targets,protected=s89.references(tok,reference)
    student,sources=create_small_paid_model(old_archive,build_config(MODEL_SPEC))
    student.eval().requires_grad_(False)
    trial_results=[]
    winner=None
    # Alpha choices are fixed BEFORE any 456 evaluation. The only
    # selection here is on PRE-EXISTING 42 authorized TRAIN examples.
    for alpha in ALPHAS:
        raw,packed,side,source_path,side_path,mut=write_candidate(
            old_archive,new_archive,sources,old_side,trained_side,alpha,a.out)
        pre=s89.sample_guard(raw,side,tok,targets,protected,reference)
        bpp_upper=8*(len(packed)+side_path.stat().st_size+7000)/SOURCE_PARAMS
        info={"alpha":alpha,"mini":pre,"physical_bpp_upper":bpp_upper,
              "changed_FP16_paid_centroid_cells":
                mut["actually_changed_FP16_codebook_cells"]}
        trial_results.append(info)
        print(json.dumps({"event":"S90_TRAIN_MINIGUARD",**info}),flush=True)
        if (not pre["lost_full"] and not pre["lost_first"] and pre["gained"]
                and bpp_upper<=CAP):
            key=(len(pre["gained"]),-alpha)
            if winner is None or key>winner[0]:
                winner=(key,alpha,raw,packed,side,source_path,side_path,mut,pre)
    sweep={"stage":"S90_PRECOMMITTED_PHYSICAL_TRAIN_ONLY_PARETO_SWEEP",
           "alphas":list(ALPHAS),"trials":trial_results,
           "all_prior_QA_TEST_permanently_embargoed":True,
           "S87_S89_frozen":True,
           "all_inclusive_bpp_ceiling":CAP,
           "selected_on_mini_TRAIN_only":True,
           "no_NEW_training_or_Jessica_tests":True,
           "model_promoted":False,
           "selected_alpha":winner[1] if winner else None}
    (a.out/"S90_PARETO_SWEEP_REPORT.json").write_text(json.dumps(sweep,indent=2))
    if winner is None:
        print(json.dumps({"event":"S90_SCIENTIFIC_NEGATIVE",
                          "reason":"No physically paid interpolation yielded fresh strict full/EOS TRAIN gains without protected mini forgetting",
                          "candidate_accepted":False}),flush=True)
        return

    _,alpha,raw,packed,side,source_path,side_path,mutation,mini=winner
    selected_cb=a.out/"S90_UNACCEPTED_SELECTED_CB61.s61"
    selected_side=a.out/"S90_UNACCEPTED_SELECTED_JOINT.safetensors"
    selected_cb.write_bytes(source_path.read_bytes())
    selected_side.write_bytes(side_path.read_bytes())
    meta={
        "stage":"S90_RESEARCH_ONLY_TRAIN_CONSTRAINED_INTERPOLATION_UNACCEPTED",
        "base_source_S87_sha256":sha(old_packed),
        "original_S89_prototype_sha256":sha(trained_packed),
        "source_sha256":sha(raw),
        "source_CB61_physical_sha256":sha(packed),
        "old_S82_sidecar_sha256":sha(a.original_S82_sidecar.read_bytes()),
        "prototype_sidecar_sha256":sha(selected_side.read_bytes()),
        "S89_targeted_wrong_TRAIN_ids":new_meta["targeted_wrong_TRAIN_ids"],
        "S90_selected_interpolation_alpha":alpha,
        "preflight_mini_TRAIN":mini,
        "repacked_mutation":mutation,
        "candidate_accepted":False,
        "all_previous_inspected_QA_embargoed":True,
        "no_LIVE_main_Render_Supabase_APK_Jessica_changes":True,
        "native_C_or_MPS_CPU_parity_not_proven":True
    }
    manifest=a.out/"S90_UNACCEPTED_PHYSICAL_MANIFEST.json"
    manifest.write_bytes(s89.stable_manifest(meta,packed,selected_side.read_bytes()))
    result_dir=a.out/"S90_FULL456"
    result_dir.mkdir(exist_ok=True)
    entire_456(a.original_S45_CB59,a.original_S82_sidecar,selected_cb,
               selected_side,manifest,result_dir,reference)


if __name__=="__main__":
    main()
