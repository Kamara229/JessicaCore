"""NRA3 S14: hard-accuracy constrained rank8 TRAIN-only independent head path.

S13 rank8 feedback architecture and its 1,266 physically paid FP16
source stay unchanged. S13 weighted cross-entropy on 72 TRAIN reduced
soft loss but caused a 37/408 (-9.07 pp) UPOS hard top-class regression.

Precommitted S14: only S9 original 72 TRAIN, split 8 lexical families
as optimizer TRAIN and 4 lexical families as INTERNAL development
for a safer objective. The original S11 router used ALL 72 examples,
so internal development is NOT independent generalization/EVAL.

Two predeclared methods:
* 'joint_margin': ordinary CE + class margin + protected-base
  distillation on already-correct baseline predictions, with original
  1,266-param rank8 readout.
* 'upos_locked': SAME operator with UPOS residual last-row weights
  and bias IDENTICALLY ZERO after FP16 physical export, to preserve
  all baseline UPOS predictions *by construction*, then train only
  semantic roles and causally observed dependency heads.

Only SHA-verified physical FP16 reopened weights may be scored.
Development data is used for selection, hence it is NOT an
independent heldout. No S12c consumed EVAL16 or old five EVAL128
or sixth reserved S3CI is read or scored; original 43M+GT+CB59
or 39parameter S11 router never trained or modified.
"""
from __future__ import annotations

from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config,UPOS,SEMANTIC_ROLES
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s11.paid_router import PaidThreeWayRouter
from research.nra3_s11.real_ru72_source import (
    checked_real_train72,S10_FEATURE_SHA,
)
from research.nra3_s13.paid_relation_feedback import (
    PaidRelationFeedback,PHYSICAL_BYTES,PARAMETERS,UPOS_DIM,ROLE_DIM,
)
from research.nra3_s13.real43m_train72_readouts import (
    S8B_43_CB59_SHA,S11_39_FP16_SHA,
    FixedNative72,gather_native_fixed_training,
)
from research.nra3_s9.new_ru_train72 import TEMPLATES

FAMILY_TRAIN=tuple(range(8))
FAMILY_INTERNAL=tuple(range(8,12))
STEPS=160
SNAP_EVERY=20
LR=.018
MARGIN=.25
MODES=("joint_margin","upos_locked")
S14_PHYSICAL_OPERATOR_BYTES=PHYSICAL_BYTES
S14_OPERATOR_PARAM_COUNT=PARAMETERS


def guard(ok,why):
    if not bool(ok):
        raise ValueError("NRA3 S14 causal rank8 hard-accuracy control "+why)


def split72(samples:tuple,physical:FixedNative72):
    ids_train=tuple(i for i,s in enumerate(samples)
                    if s.row.lexical_family in FAMILY_TRAIN)
    ids_internal=tuple(i for i,s in enumerate(samples)
                       if s.row.lexical_family in FAMILY_INTERNAL)
    guard(len(ids_train)==48 and len(ids_internal)==24
          and set(ids_train).isdisjoint(ids_internal)
          and set(ids_train)|set(ids_internal)==set(range(72))
          and len({x.row.lexical_family for x in samples})==12,
          "fixed 8/4 precommitted family-based split changed")
    def subset(indices):
        i=torch.tensor(indices,dtype=torch.long)
        n=int((physical.upos_gold[i]>=0).sum())
        h=int((physical.dep_gold[i]>=0).sum())
        return FixedNative72(
            *[t.index_select(0,i).contiguous() for t in (
                physical.relation,physical.upos,physical.role,physical.dep,
                physical.valid,physical.upos_gold,physical.role_gold,
                physical.dep_gold,physical.candidate_heads)],
            num_words=n,num_past_head_arcs=h)
    return subset(ids_train),subset(ids_internal)


def _ce(logits,labels):
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                           labels.reshape(-1),ignore_index=-100)


def _margin(logits,labels,margin=MARGIN):
    mask=labels>=0
    x=logits[mask];gold=labels[mask]
    guard(len(gold)>0,"empty margins")
    true=x.gather(1,gold[:,None]).squeeze(1)
    alt=x.scatter(1,gold[:,None],-1e4).max(1).values
    return F.softplus((alt-true+margin)*1.5).mean()/1.5


def _retention_kl(new,orig,labels):
    active=labels>=0
    correct=active&(orig.argmax(-1)==labels)
    if not bool(correct.any()):
        return new.sum()*0.
    # Penalizes confidence shifts for already-correct BASE hypotheses.
    return F.kl_div(F.log_softmax(new[correct],dim=-1),
                    F.softmax(orig[correct].detach(),dim=-1),
                    reduction="batchmean")


def task_aligned_objective(model:PaidRelationFeedback,
                           x:FixedNative72,mode:str):
    guard(mode in MODES,"unknown precommitted objective")
    z=model(x.relation,x.upos,x.role,x.dep,x.valid)
    pu=_ce(z.upos_logits,x.upos_gold)
    pr=_ce(z.semantic_role_logits,x.role_gold)
    d=z.dependency_logits.masked_fill(~x.candidate_heads,-1e4)
    pd=_ce(d,x.dep_gold)
    rolemargin=_margin(z.semantic_role_logits,x.role_gold)
    if mode=="joint_margin":
        upmargin=_margin(z.upos_logits,x.upos_gold)
        keepupos=_retention_kl(z.upos_logits,x.upos,x.upos_gold)
        keeproles=_retention_kl(z.semantic_role_logits,x.role,x.role_gold)
        loss=pu+pr+.5*pd+.6*upmargin+.4*rolemargin+ \
             .50*keepupos+.30*keeproles
    else:
        upmargin=pu.detach()*0.
        keepupos=pu.detach()*0.
        keeproles=_retention_kl(z.semantic_role_logits,x.role,x.role_gold)
        loss=pr+.65*pd+.35*rolemargin+.10*keeproles
    loss+=.00008*model.output_weight.square().sum()+ \
          .0002*model.dependency_diagonal.square().sum()
    guard(bool(torch.isfinite(loss)),"nonfinite training loss")
    return loss


def metrics(model:PaidRelationFeedback,x:FixedNative72):
    with torch.no_grad():
        out=model(x.relation,x.upos,x.role,x.dep,x.valid)
        fields={}
        for key,logits,target in (
            ("upos",out.upos_logits,x.upos_gold),
            ("roles",out.semantic_role_logits,x.role_gold),
            ("past_heads",out.dependency_logits.masked_fill(
                ~x.candidate_heads,-1e4),x.dep_gold),
        ):
            mask=target>=0
            pred=logits.argmax(-1)
            fields[key]={
                "correct":int((pred[mask]==target[mask]).sum()),
                "total":int(mask.sum()),
                "hard_accuracy":float((pred[mask]==target[mask]).float().mean()),
                "unweighted_CE":float(_ce(logits,target))
            }
        return fields


def baseline_safe_comparison(cand:dict,base:dict, *,
                             require_semantic_nonregression:bool=True):
    for head in ("upos","roles","past_heads"):
        if (cand[head]["total"]!=base[head]["total"] or
            cand[head]["total"]<=0):
            raise ValueError("S14 mismatched baseline head count")
    upok=cand["upos"]["correct"]>=base["upos"]["correct"]
    rolesok=cand["roles"]["correct"]>=base["roles"]["correct"]
    depok=cand["past_heads"]["correct"]>=base["past_heads"]["correct"]
    return upok and rolesok and depok


def _lock_upos_(model:PaidRelationFeedback):
    # No extra inference mode flags; all safety state is paid physically.
    # Zeros are physically FP16-serialized in the existing 1,266 values.
    with torch.no_grad():
        model.output_weight[:UPOS_DIM,:].zero_()
        model.output_bias[:UPOS_DIM].zero_()


def learn(xtrain:FixedNative72,xdev:FixedNative72,mode:str):
    guard(mode in MODES,"precommitted two training regimes only")
    torch.manual_seed(314001)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model=PaidRelationFeedback().eval()
    if mode=="upos_locked":
        _lock_upos_(model)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    snap=[]
    for i in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss=task_aligned_objective(model,xtrain,mode)
        loss.backward()
        guard(all(p.grad is not None for p in model.parameters()),
              "all explicit physical fields must be part of training")
        if mode=="upos_locked":
            with torch.no_grad():
                model.output_weight.grad[:UPOS_DIM].zero_()
                model.output_bias.grad[:UPOS_DIM].zero_()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if mode=="upos_locked":
            _lock_upos_(model)
        if (i+1)%SNAP_EVERY==0:
            with torch.no_grad():
                physical=model.physical_bytes()
                loaded=PaidRelationFeedback.from_physical(physical)
                guard(loaded.physical_bytes()==physical,
                      "physical FP16 roundtrip changed")
                if mode=="upos_locked":
                    guard(bool((loaded.output_weight[:UPOS_DIM]==0).all())
                          and bool((loaded.output_bias[:UPOS_DIM]==0).all()),
                          "physically serialized UPOS must remain exact base")
                snap.append({
                    "mode":mode,
                    "step":i+1,
                    "source":physical,
                    "train":metrics(loaded,xtrain),
                    "internal_development":metrics(loaded,xdev),
                    "physical_sha":hashlib.sha256(physical).hexdigest(),
                })
    guard(len(snap)==STEPS//SNAP_EVERY and
          all(k["mode"]==mode for k in snap),"optimizer snapshot contract")
    return snap


def _select(records,train_base,dev_base):
    # Stage is explicitly TRAIN+INTERNAL selection only, not EVAL.
    # Main goals lexicographic in the precommitted INTERNAL group:
    # 1. no hard regression in any linguistic head,
    # 2. better roles and/or dependency while preserving UPOS,
    # 3. minimize sum of unweighted task cross-entropy.
    eligible=[]
    for r in records:
        dev=r["internal_development"]
        safe=baseline_safe_comparison(dev,dev_base)
        gain_roles=dev["roles"]["correct"]-dev_base["roles"]["correct"]
        gain_dep=dev["past_heads"]["correct"]-dev_base["past_heads"]["correct"]
        gain_upos=dev["upos"]["correct"]-dev_base["upos"]["correct"]
        counts=(gain_roles,gain_dep,gain_upos)
        ce=sum(dev[k]["unweighted_CE"] for k in ("upos","roles","past_heads"))
        r["S14_INTERNAL_only_no_regression_guard"]=safe
        r["S14_INTERNAL_only_hard_gains_roles_dep_upos"]=counts
        r["S14_INTERNAL_only_unweighted_three_task_CE"]=ce
        if safe and (gain_roles>0 or gain_dep>0 or gain_upos>0):
            eligible.append(r)
    if not eligible:
        return None
    def rank(r):
        gain=r["S14_INTERNAL_only_hard_gains_roles_dep_upos"]
        return (gain[0]+gain[1]+gain[2], gain[0],gain[2],
                -r["S14_INTERNAL_only_unweighted_three_task_CE"])
    return max(eligible,key=rank)


def main():
    ap=argparse.ArgumentParser()
    for name in ("s10-hidden","s10-report","s8-cb59","s11-router","out"):
        ap.add_argument("--"+name,required=True,type=Path)
    args=ap.parse_args()
    guard(not args.out.exists() or not any(args.out.iterdir()),
          "do not overwrite original S14 physical optimizer run")
    samples,original_report=checked_real_train72(
        args.s10_hidden,args.s10_report)
    source=args.s8_cb59.read_bytes()
    tiny=args.s11_router.read_bytes()
    guard(len(source)==653798 and
          hashlib.sha256(source).hexdigest()==S8B_43_CB59_SHA
          and CB59(source).meta["original_unique_parameter_count"]==6_106_154
          and len(CB59(source).handles)==43
          and len(tiny)==123 and hashlib.sha256(tiny).hexdigest()==S11_39_FP16_SHA
          and PaidThreeWayRouter.from_physical(tiny).physical_bytes()==tiny,
          "original 43 native C packed 5bit/39 FP16 selector source drift")
    args.out.mkdir(parents=True,exist_ok=True)
    lib=args.out/"original_s14_43_CB59_C.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                    "research/calibri/direct_kernel_v59.c","-o",str(lib)],check=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parent=NRA3NativePackedCPU(
        source,NRA3Config(parallel_hypotheses=3,parallel_selector_version=3),
        lib,paid_router_physical=tiny).eval()
    guard(parent.source_sha256==S8B_43_CB59_SHA and
          parent.paid_router.physical_bytes()==tiny and
          parent.output_gate==0 and parent.candidate_accepted is False and
          not parent.native_cpu_weights_dense_resident,
          "S14 incorrectly modified original 6M CB59 or S11")
    x=gather_native_fixed_training(samples,parent)
    train,internal=split72(samples,x)
    init=PaidRelationFeedback().eval()
    basetrain=metrics(init,train)
    basedev=metrics(init,internal)
    all_snaps=[]
    for mode in MODES:
        all_snaps.extend(learn(train,internal,mode))
    selected=_select(all_snaps,basetrain,basedev)
    # Always save one representative per prereg training regime, so
    # evidence exists even if nothing passes INTERNAL selection.
    for mode in MODES:
        candidate=max([q for q in all_snaps if q["mode"]==mode],
            key=lambda q:(sum(q["internal_development"][h]["correct"]
                           for h in ("upos","roles","past_heads")),
                          -q["S14_INTERNAL_only_unweighted_three_task_CE"]))
        target=args.out/f"NRA3_S14_{mode}_PAID_FP16_CANDIDATE_INTERNAL_ONLY.n3s13"
        target.write_bytes(candidate["source"])
        guard(PaidRelationFeedback.from_physical(target.read_bytes()).physical_bytes()
              ==candidate["source"],"S14 physically saved candidate corruption")
    if selected is not None:
        selected_path=args.out/"NRA3_S14_INTERNAL_DEVELOPMENT_SELECTED_NOT_INDEPENDENT_FP16.n3s13"
        selected_path.write_bytes(selected["source"])
        guard(selected_path.read_bytes()==selected["source"],"physical select mismatch")
    physical_nra=653798+123+PHYSICAL_BYTES
    physical_params=6_106_154+39+PARAMETERS
    summary={
      "stage":"NRA3_S14_TRUE43M_RU72_CAUSAL_HARD_HEAD_RETENTION_INTERNAL_DEV_ONLY",
      "original_43M_S10_post_GT_real_train72_sha256":S10_FEATURE_SHA,
      "original_S8B_43_C_native_CB59_sha256":hashlib.sha256(source).hexdigest(),
      "original_S11_39_FP16_six_cycle_router_sha256":hashlib.sha256(tiny).hexdigest(),
      "S14_physical_model_did_not_change_parent_or_GT_or_S11":True,
      "S14_uses_original_72_previously_seen_S9_TRAIN_only":True,
      "S14_lexical_families_optimizer_train":list(FAMILY_TRAIN),
      "S14_lexical_families_INTERNAL_development":list(FAMILY_INTERNAL),
      "S14_source_train_rows":48,
      "S14_internal_development_rows":24,
      "S14_internal_development_is_NOT_independent_EVAL":True,
      "S14_internal_development_partially_contaminated_by_pretrained_S11":True,
      "S14_inference_uses_only_causal_relation_and_original_logits_not_gold":True,
      "S14_PRECOMMITTED_train_regimes":list(MODES),
      "S14_PRECOMMITTED_steps":STEPS,
      "S14_PRECOMMITTED_checkpoints":list(range(SNAP_EVERY,STEPS+1,SNAP_EVERY)),
      "S14_zero_UPOS_path_in_upos_locked_physically_ensures_base_parity":True,
      "S14_base_train_hard":basetrain,
      "S14_base_INTERNAL_development_hard":basedev,
      "S14_each_precommitted_candidate":{
        k: [
           {"step":r["step"],"SHA":r["physical_sha"],
            "TRAIN_hard":r["train"],
            "INTERNAL_hard":r["internal_development"],
            "INTERNAL_guard":r["S14_INTERNAL_only_no_regression_guard"]}
           for r in all_snaps if r["mode"]==k
        ] for k in MODES},
      "S14_internal_training_only_eligible_candidate":selected is not None,
      "S14_internal_selected_mode_step_sha":(
         None if selected is None else
         {"mode":selected["mode"],"step":selected["step"],
          "SHA":selected["physical_sha"],
          "TRAIN_only_scores":selected["train"],
          "INTERNAL_used_FOR_MODEL_SELECTION":selected["internal_development"]}),
      "S14_new_1266_Fp16_paid_physical_bytes":PHYSICAL_BYTES,
      "S14_new_1266_FP16_parameters":PARAMETERS,
      "S14_raw_nra_selector_readout_bytes":physical_nra,
      "S14_raw_nra_selector_readout_bpp":8*physical_nra/physical_params,
      "S14_old_consumed_S12c_eval16_NEVER_loaded":True,
      "S14_old_five_EVAL128_never_loaded":True,
      "S14_reserved_sixth_S3CI_EVAL128_never_loaded":True,
      "S14_new_unseen_independent_EVAL_quality_measured":False,
      "S14_actual_active_616_J7_or_fullEOS_measured":False,
      "S14_full_system_43M_GT_NRA_inclusive_bpp_at_most_point7_verified":False,
      "S14_full_system_CPU_at_least90pct_CB59_verified":False,
      "S14_REAL_quality_gain_certified":False,
      "S14_research_acceptance":False,
      "S14_candidate_accepted":False,
      "S14_output_gate":0,
      "S14_activation_allowed":False,
      "S14_no_LIVE_Jessica_chat_Render_Supabase_APK":True,
    }
    p=args.out/"NRA3_S14_CAUSAL_HARD_HEAD_INTERNAL_DEVELOPMENT_NO_PROMOTION.json"
    p.write_text(json.dumps(summary,sort_keys=True,ensure_ascii=False,
                            indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items()
                      if k!="S14_each_precommitted_candidate"},
                      sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
