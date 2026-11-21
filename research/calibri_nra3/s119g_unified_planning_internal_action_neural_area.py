"""S119g: neural latent Planning-and-Internal-Action Area (NPA1).

ONE Calibri 2.0 research body contains:
  original true-12 CB59+GT frozen source -> trainable rank4 shared core
  -> FULL old 53,751 NRA3-R -> NPA1 multistep neural computation.
Only the NEW 12-layer rank4 core, the NEW S119f graph gate, and NEW NPA1
weights receive a SINGLE joint loss. No separate external planner model.
S119g reuses actual physical S119f checkpoint and genuine old S114
original12 CPU TRAIN-A representations. The original Calibri decoder and
GT are NOT attached to this offline source experiment, so it does NOT
demonstrate actual autoregressive planning, improved language responses,
external tool execution, or full J7/CPU/bpp acceptance.

NPA1 "actions" are latent learned alternatives for INTERNAL state transitions
(attend, relate, compose, check); these are NOT real external actions or
guaranteed learned planning semantics. A causal original-source prefix is
processed without using the next token; only TRAIN-side next-position
NRA3 state probabilities supervise a narrow next-state prediction pretext.
Source-only test is NOT an independent language-quality benchmark.

No protected S118 SynTagRus-B GOLD, spent eval, Jessica, Render, Supabase,
LIVE or production writes. Source license CC BY-NC-SA 4.0 research only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119d_reuse_old37087_true12_inweight_eventgraph import (
    PARENT_H, PARENT_H_SHA, PARENT_REPORT,
    two_hop_parameter_free_event_relation_graph,
)
from research.calibri_nra3.s119f_unified_core_gt_reasoning_joint import (
    UnifiedCalibri, STAGE as PARENT_STAGE,
    CHECKPOINT as PARENT_CHECKPOINT, REPORT as PARENT_REPORT_NAME,
    sha as physical_sha,
)

STAGE="CALIBRI_S119G_UNIFIED_CORE_GT_NRA3R_NPA1_LATENT_INTERNAL_ACTIONS_GATE0"
CHECKPOINT="CALIBRI_S119G_CORE_NRA3_NPA1_UNIFIED_TRAINONLY_FP16_SHADOW.safetensors"
REPORT="CALIBRI_S119G_TRAIN_A_REAL12_NEW_NPA1_SINGLE_JOINT_LOSS_NO_GOLD.json"
PREFIX="model.nra3_calibri_joint."
LATENT_ACTIONS=("attend","relate","compose","check")
ACTION_WIDTH=32
ACTION_STEPS=3


@dataclass(frozen=True)
class NeuralInternalPlan:
    action_logits:torch.Tensor      # [B,3,4] 3 recurrent internal steps
    state_trace:torch.Tensor        # [B,3,32] differentiated neural states
    next_event_logits:torch.Tensor  # [B,6] predicted next-position NRA3 state


class NeuralPlanningActionArea(nn.Module):
    """Persistent in-model trainable internal computation, never a tool executor.

    The real work here is learned action mixing + GRU neural state transitions
    conditioned jointly on original12 core vectors and NRA3 neural roles,
    mini areas, morphology-associated events, and learned memory attention.
    """
    def __init__(self):
        super().__init__()
        # two 320D Calibri states + six event + eight original mini area
        # + sixteen original neural memory features = 670
        self.input_projection=nn.Linear(670,ACTION_WIDTH)
        self.state_update=nn.GRUCell(ACTION_WIDTH,ACTION_WIDTH)
        self.action_selector=nn.Linear(ACTION_WIDTH,len(LATENT_ACTIONS))
        self.internal_action_matrices=nn.Parameter(torch.randn(
            len(LATENT_ACTIONS),ACTION_WIDTH,ACTION_WIDTH)*.02)
        self.state_norm=nn.LayerNorm(ACTION_WIDTH)
        self.pred_next_event=nn.Linear(ACTION_WIDTH,6)
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False

    def forward(self,core12_hidden,graph,valid):
        check(self.output_gate==0 and not self.candidate_accepted
              and not self.activation_allowed,
              "NPA1 cannot execute external actions or affect original decoder")
        check(core12_hidden.ndim==4 and core12_hidden.shape[1]==12
              and core12_hidden.shape[-1]==320
              and valid.ndim==2 and valid.dtype==torch.bool
              and valid.shape==(
                  core12_hidden.shape[0],core12_hidden.shape[2])
              and valid.shape[1]>=2 and bool(valid.any(-1).all())
              and bool(torch.isfinite(core12_hidden).all()),
              "NPA1 requires original 12-layer Calibri prefix h320")
        b,t=valid.shape
        counts=valid.long().sum(-1)
        check(bool((valid==(
                  torch.arange(t,device=valid.device)[None,:]<
                  counts[:,None])).all()),
              "native prefix must be contiguous, no future or pad leakage")
        batch=torch.arange(b,device=valid.device)
        last=counts-1
        # Decoder-side prefix tensors are causal source: never open NEXT
        # position in forward. NRA3 graph must be computed on same prefix.
        mixed=core12_hidden.mean(1)             # [B,T,320]
        selected=mixed[batch,last]               # [B,320]
        arcs=graph.role_arc_logits[batch,:,last,:].float()
        weights=torch.softmax(arcs,dim=-1)
        weights=weights*valid[:,None,:]
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1.e-9)
        roles=torch.einsum("brt,bth->brh",weights,mixed).mean(1)
        event=graph.event_state_logits[batch,last].float().softmax(-1)
        minis=graph.mini_area_logits[batch,last].float().sigmoid()
        mem=graph.memory_slot_attention[batch,last].float()
        feats=torch.cat((selected,roles,event,minis,mem),dim=-1)
        check(feats.shape==(b,670),"NPA1 must use real core AND old NRA3 memory")
        embedded=F.gelu(self.input_projection(feats))
        state=torch.zeros_like(embedded)
        actions=[]
        states=[]
        for _ in range(ACTION_STEPS):
            proposed=self.state_update(embedded,state)
            choice=self.action_selector(proposed)
            mix=torch.einsum("ba,aij,bj->bi",
                choice.softmax(-1),self.internal_action_matrices,proposed)
            state=self.state_norm(proposed+0.20*torch.tanh(mix))
            actions.append(choice)
            states.append(state)
        p=NeuralInternalPlan(
            torch.stack(actions,dim=1),
            torch.stack(states,dim=1),
            self.pred_next_event(state))
        check(p.action_logits.shape==(b,ACTION_STEPS,len(LATENT_ACTIONS))
              and p.state_trace.shape==(b,ACTION_STEPS,ACTION_WIDTH)
              and p.next_event_logits.shape==(b,6)
              and bool(torch.isfinite(p.next_event_logits).all()),
              "NPA1 recurrent action execution not well formed")
        return p


class UnifiedCalibriWithNPA1(UnifiedCalibri):
    """NPA1 is a real nn.Module child of S119f unified Calibri state_dict."""
    def __init__(self):
        super().__init__()
        self.planning_action_area=NeuralPlanningActionArea()

    def forward(self,h,valid):
        graph,relational=super().forward(h,valid)
        # Representation proxy ONLY; real Calibri decoder hookup is next
        # stage. Keep the complete original S115 neural child untouched.
        planning=self.planning_action_area(
            self.core_growth(h),graph,valid)
        return graph,relational,planning

    def new_joint_trainables(self):
        return (super().new_joint_trainables()
                +list(self.planning_action_area.parameters()))

    def load_physically_trained_s119f(self,checkpoint:dict):
        raw={}
        for name,value in checkpoint.items():
            check(name.startswith(PREFIX) and
                  value.dtype==torch.float16,
                  "only physically trained unified S119f checkpoint")
            k=name[len(PREFIX):]
            raw[k]=value.float()
        expected=super().state_dict()
        check(set(raw)==set(expected)
              and sum(v.numel() for v in raw.values())==84_479,
              "parent must include trained core + full old NRA3 + graph gate")
        # Do not materialize a separate second model; load S119f directly
        # into current instance's inherited param namespaces.
        self.load_state_dict(raw,strict=False)
        for p in self.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        check(not any(p.requires_grad for p in
                      self.nra3_neural_weights.parameters()),
              "all original 53751 NRA3 parameters must remain frozen")


def teacher_targets(child,src,valid,prefix:int):
    """Only TRAIN-A original next-token representation; never S118 GOLD.

    Teacher is frozen S115 itself, NOT an external model/label. Prefix
    context and next-position representation are explicitly separated.
    """
    check(2<=prefix<src.shape[2] and bool(valid[:,:prefix+1].all()),
          "real source TRAIN-only next native BPE position required")
    old=child.nra3_neural_weights
    with torch.no_grad():
        prior=old(src[:,:,:prefix],valid[:,:prefix])
        next_full=old(src[:,:,:prefix+1],valid[:,:prefix+1])
        next_event=next_full.event_state_logits[:,-1].float().softmax(-1)
        rel=two_hop_parameter_free_event_relation_graph(
            prior,valid[:,:prefix]).float()
        rel=rel/rel.sum(-1,keepdim=True).clamp_min(1e-9)
        minis=prior.mini_area_logits.detach()
    return next_event.detach(),rel.detach(),minis


def single_joint_loss_step(child,src,valid,prefix,optimizer):
    optimizer.zero_grad(set_to_none=True)
    target_next,target_rel,old_minis=teacher_targets(
        child,src,valid,prefix)
    graph,rel,plan=child(src[:,:,:prefix],valid[:,:prefix])
    plan_loss=-(target_next*F.log_softmax(
        plan.next_event_logits,-1)).sum(-1).mean()
    rel_loss=-(target_rel*F.log_softmax(
        rel,-1)).sum(-1)[valid[:,:prefix]].mean()
    keep_minis=(
        graph.mini_area_logits[valid[:,:prefix]]-
        old_minis[valid[:,:prefix]]).square().mean()
    loss=plan_loss+0.25*rel_loss+0.01*keep_minis
    check(bool(torch.isfinite(loss)),
          "one shared core+NRA3R+NPA1 train loss must remain finite")
    loss.backward()
    grads={
        "new_core_up":child.core_growth.up.grad,
        "new_graph_gate":child.reasoning_role_gate.grad,
        "new_planning_action_policy":child.planning_action_area.action_selector.weight.grad,
        "new_planning_action_execution":child.planning_action_area.internal_action_matrices.grad,
        "new_planning_next_event_head":child.planning_action_area.pred_next_event.weight.grad,
    }
    for name,v in grads.items():
        check(v is not None and bool(torch.isfinite(v).all())
              and float(v.abs().sum())>1.e-10,
              "ONE SAME supervised loss must train "+name)
    trace={
        "only_trainA_source":True,
        "one_joint_objective":float(loss.detach()),
        "next_event_self_distill_ce":float(plan_loss.detach()),
        "relational_self_distill_ce":float(rel_loss.detach()),
        "mini8_retention_penalty":float(keep_minis.detach()),
        "grads_l1":{k:float(v.abs().sum()) for k,v in grads.items()},
        "soft_latent_computation_only":True,
    }
    optimizer.step()
    return trace


def run(*,prior_s119f,prior_report,source,source_report,out):
    check(not out.exists()
          and prior_s119f.name==PARENT_CHECKPOINT
          and prior_report.name==PARENT_REPORT_NAME
          and source.name==PARENT_H
          and source_report.name==PARENT_REPORT
          and physical_sha(source)==PARENT_H_SHA,
          "only original S114 TRAIN-A plus one pinned S119f unified model")
    old_json=json.loads(prior_report.read_text(encoding="utf8"))
    check(old_json["stage"]==PARENT_STAGE
          and old_json["checkpoint_SHA256"]==physical_sha(prior_s119f)
          and old_json["checkpoint_bytes"]==prior_s119f.stat().st_size
          and old_json["old_trained_S115_full53751_SHA256"]
            =="42e0bb4e81cb4ea199e4ee8b5fccd939ab4550b7c6269af44ede6a77462d795c"
          and old_json["new_trainable_core_rank4_parameters"]==30720
          and old_json["new_graph_twohop_role_parameters"]==8
          and old_json["output_gate"]==0
          and old_json["candidate_accepted"] is False,
          "S119f unified 84479 source provenance and physical SHA wrong")
    provenance=json.loads(source_report.read_text())
    check(provenance["ALL12_CPU_SAFETENSORS_SHA256"]==PARENT_H_SHA
          and provenance["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"] is True
          and provenance["candidate_accepted"] is False,
          "actual original Calibri+GT old 616 source witness absent")
    torch.manual_seed(119119)
    torch.set_num_threads(2)
    old_weights=load_file(str(prior_s119f),device="cpu")
    model=UnifiedCalibriWithNPA1()
    model.load_physically_trained_s119f(old_weights)
    frozen={k:v.detach().half().clone()
            for k,v in model.nra3_neural_weights.state_dict().items()}
    new0={k:v.detach().clone() for k,v in model.state_dict().items()
          if not k.startswith("nra3_neural_weights.")}
    a=load_file(str(source),device="cpu")
    h=a["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    mask=a["native_BPE_source_valid_mask"]
    check(h.ndim==4 and h.shape[0]==256 and h.shape[1]==12
          and h.shape[-1]==320 and h.dtype==torch.float16
          and mask.shape==(256,h.shape[2]) and mask.dtype==torch.bool,
          "only true original12 source TRAIN-A tensors permitted")
    candidates=[(i,int(mask[i].sum())) for i in range(1,33)
                if int(mask[i].sum())>=20]
    check(len(candidates)>=2,
          "genuine distinct TRAIN-A rows for train and unused probe")
    row_train,n=candidates[0]
    row_probe,pn=candidates[1]
    check(bool(mask[row_train,:n].all()) and
          bool(mask[row_probe,:pn].all()) and
          not bool(mask[row_train,n:].any()) and
          not bool(mask[row_probe,pn:].any()),
          "native BPE source rows contiguous")
    src=h[row_train:row_train+1,:,:n].detach().float()
    valid=mask[row_train:row_train+1,:n]
    probe=h[row_probe:row_probe+1,:,:pn].detach().float()
    pvalid=mask[row_probe:row_probe+1,:pn]
    # On TRAIN-A row unseen by S119f prior training. The second row
    # stays probe-only; neither is independent pre-registered GOLD.
    ptr=min(16,n-1)
    ppr=min(16,pn-1)
    def probe_ce():
        with torch.no_grad():
            teacher,_,_=teacher_targets(model,probe,pvalid,ppr)
            _,_,plan=model(probe[:,:,:ppr],pvalid[:,:ppr])
            ce=-(teacher*F.log_softmax(
                plan.next_event_logits,-1)).sum(-1).mean()
            return float(ce)
    probe_before=probe_ce()
    optimizer=torch.optim.Adam(model.new_joint_trainables(),lr=.0008)
    trace=[single_joint_loss_step(model,src,valid,ptr,optimizer)
           for _ in range(2)]
    probe_after=probe_ce()
    for name,t in frozen.items():
        check(torch.equal(t,model.nra3_neural_weights.state_dict()[name].detach().half()),
              "S119g cannot forget ANY of 53751 S115 trained original weights")
    check(bool((model.core_growth.up.detach()-
                new0["core_growth.up"]).abs().max()>0)
          and bool((model.reasoning_role_gate.detach()-
                new0["reasoning_role_gate"]).abs().max()>0)
          and bool((model.planning_action_area.action_selector.weight.detach()-
                new0["planning_action_area.action_selector.weight"]).abs().max()>0)
          and bool((model.planning_action_area.internal_action_matrices.detach()-
                new0["planning_action_area.internal_action_matrices"]).abs().max()>0),
          "core, reasoning and planning neuron must ALL actually update")
    physical={PREFIX+k:v.detach().half().cpu().contiguous()
              for k,v in model.state_dict().items()}
    check(sum(v.numel() for v in physical.values())==
          84_479+sum(p.numel() for p in model.planning_action_area.parameters()),
          "new area must physically live within single Calibri child")
    for k,v in old_weights.items():
        if ".nra3_neural_weights." in k:
            check(torch.equal(v,physical[k]),
                  "old 53751 FP16 words must remain exactly identical")
    out.mkdir(parents=True)
    save_file(physical,str(out/CHECKPOINT))
    reloaded=load_file(str(out/CHECKPOINT),device="cpu")
    check(set(reloaded)==set(physical)
          and all(torch.equal(reloaded[k],v) for k,v in physical.items()),
          "S119g full neural child physically saved / reloaded exact")
    report={
        "stage":STAGE,
        "single_Calibri_neural_module":PREFIX.removesuffix("."),
        "S119f_parent_checkpoint_SHA256":physical_sha(prior_s119f),
        "true_original12_S114_TRAIN_A_source_SHA256":PARENT_H_SHA,
        "all_53751_original_NRA2_NRA3_params_preserved_byte_exact":True,
        "new_shared_trainable_core_parameters":30720,
        "new_shared_graph_parameters":8,
        "new_NPA1_neural_parameters":sum(p.numel() for p in
                                         model.planning_action_area.parameters()),
        "NPA1_latent_neural_action_names_not_real_external_actions":
            list(LATENT_ACTIONS),
        "NPA1_recurrent_internal_computation_steps":ACTION_STEPS,
        "all_components_one_joint_optimizer_and_loss":True,
        "distinct_old_source_TRAIN_A_row_for_new_training":row_train,
        "distinct_TRAIN_A_row_for_source_probe_only":row_probe,
        "train_source_native_BPE_count":n,
        "probe_source_native_BPE_count":pn,
        "joint_gradient_training_trace":trace,
        "probe_SOURCE_TRAIN_A_only_cross_entropy_before":probe_before,
        "probe_SOURCE_TRAIN_A_only_cross_entropy_after":probe_after,
        "S119g_parent_candidate_J7_616_NOT_RUN":True,
        "NPA1_quality_or_actual_planning_NOT_PROVEN":True,
        "complete_decoder_or_GT_joint_forward_NOT_RUN":True,
        "external_tools_NEVER_EXECUTED":True,
        "independent_unseen_generation_QUALITY_NOT_MEASURED":True,
        "whole_model_CPU_and_all_in_bpp_NOT_MEASURED":True,
        "SynTagRusB_GOLD_still_unopened":True,
        "noncommercial_CC_BY_NC_SA_4_0_only":True,
        "physical_single_child_all_FP16_parameters":sum(v.numel() for v in
                                                        physical.values()),
        "physical_single_child_checkpoint_bytes":(out/CHECKPOINT).stat().st_size,
        "physical_single_child_SHA256":physical_sha(out/CHECKPOINT),
        "candidate_accepted":False,
        "activation_allowed":False,
        "output_gate":0,
        "decision":"ONE_NEURAL_MODEL_CORE_NRA3_NPA1_REAL12_SOURCE_GRADIENT_PROOF_ONLY",
    }
    (out/REPORT).write_text(json.dumps(report,indent=2,sort_keys=True,
                                      ensure_ascii=False)+"\n",encoding="utf8")
    print("S119G_UNIFIED_NEURAL_PLANNING_ACTION_AREA_SOURCE_ONLY "+
          json.dumps({
              "pretrained_old_parameters":53751,
              "new_core_parameters":30720,
              "new_NRA3R_parameters":8,
              "new_neural_planning_area_parameters":
                  report["new_NPA1_neural_parameters"],
              "source_only_train_BPE":n,
              "single_model_parameters":
                  report["physical_single_child_all_FP16_parameters"],
              "checkpoint_bytes":
                  report["physical_single_child_checkpoint_bytes"],
              "core_nra_plan_single_gradients":True,
              "old_pretrained_exact":True,
              "candidate_accepted":False,
              "independent_quality_proven":False,
          },sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ("prior_s119f","prior_report","source","source_report","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(prior_s119f=a.prior_s119f,prior_report=a.prior_report,
        source=a.source,source_report=a.source_report,out=a.out)

if __name__=="__main__":
    main()
