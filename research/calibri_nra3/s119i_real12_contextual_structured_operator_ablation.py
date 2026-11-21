"""Calibri S119i: contextual structured interface into a unified core.

This stage tests the *useful* substance of the user's representation idea:
learned input representations should encode causal event/participant relations,
rather than simply apply token-wise learned mathematical projections.

Rather than forcing two large R/T islands from the S119h MOCK exercise,
this child extends the genuine physically trained S119g unified Calibri
core-growth+NRA3-R+NPA1 checkpoint, keeping ALL old 53,751 S115 weights.

New ContextualStructuredInterface mixes 8 old in-neural learned role arcs,
event state, 8 NRA2 mini areas and old neural memory with 320D original
CB59+Green Tunnel true 12-layer source. Its attention *only attends backward*,
and only the last prefix position is updated. A zero-initialized residual
guarantees exact old unified-model behavior at step zero.

Genuine S114 original12 TRAIN-A SOURCE-only h320 and native next-position
h320 train a tiny contrastive next-state PROXY, with a matched-parameter
token-local ablation, pre-selected disjoint TRAIN-A source-row probes.
Neither path reads gold grammar or protected independent data. The probe
was already a TRAIN corpus used in prior history; this is NOT an untouched
independent language or universal-answer validation.

IMPORTANT: S114 captures are POST original layer outputs. The interface is
tested on the earliest captured layer0 representation; it is NOT yet placed
in front of actual original CB59 embedding and does not run actual 43M
decoder or live Green Tunnel with the new area. Do not call it an
in-production "pre-core" area, or an improved generative language model.
No original 616 J7, actual all-in bpp, whole-model CPU or external QA claims.
No Jessica, Render, Supabase, LIVE, merge, S118-B GOLD or spent EVAL.
"""
from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119g_unified_planning_internal_action_neural_area import (
    UnifiedCalibriWithNPA1, STAGE as PARENT_STAGE,
    PREFIX as PARENT_PREFIX, CHECKPOINT as PARENT_CHECKPOINT,
    REPORT as PARENT_REPORT, teacher_targets,
)
from research.calibri_nra3.s119d_reuse_old37087_true12_inweight_eventgraph import (
    PARENT_H_SHA, PARENT_H, PARENT_REPORT as SOURCE_REPORT,
)

STAGE="CALIBRI_S119I_REAL_ORIGINAL12_CONTEXTUAL_RELATION_STRUCTURED_INPUT_ABLATION_GATE0"
CHECKPOINT="CALIBRI_S119I_CONTEXTUAL_RELATIONS_CORE_NRA3_NPA1_FP16_SHADOW.safetensors"
REPORT="CALIBRI_S119I_REAL12_TRAIN_A_MATCHED_LOCAL_VS_CONTEXTUAL_ABLATION.json"
HIDDEN=320
STRUCT_WIDTH=32
FROZEN_PRETRAINED=53751
PARENT_PARAMETERS=116777
EXPECTED_S114_ROWS=256

def sha(p:Path)->str:
    return sha256(p.read_bytes()).hexdigest()

class ContextualStructuredInterface(nn.Module):
    """Learned causal structure from real pretrained NRA3 relations, no oracle.

    For current position t: causal attention over 0..t and role arcs,
    plus old 6 event, 8 original mini-area, 16 memory components.
    No future state, no token surface or hardcoded role rule.
    SAME parameters under local/contextual ablation.
    """
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(HIDDEN)
        self.query=nn.Linear(HIDDEN,STRUCT_WIDTH,bias=False)
        self.key=nn.Linear(HIDDEN,STRUCT_WIDTH,bias=False)
        self.value=nn.Linear(HIDDEN,STRUCT_WIDTH,bias=False)
        self.semantic=nn.Linear(6+8+16,STRUCT_WIDTH,bias=False)
        self.role_mix=nn.Parameter(torch.zeros(8))
        self.fuse=nn.Linear(STRUCT_WIDTH,HIDDEN,bias=False)
        nn.init.zeros_(self.fuse.weight)
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False

    def forward(self,earliest_layer_h,graph,valid,*,contextual:bool):
        check(self.output_gate==0 and self.candidate_accepted is False
              and self.activation_allowed is False
              and isinstance(contextual,bool)
              and earliest_layer_h.ndim==3
              and earliest_layer_h.shape[-1]==HIDDEN
              and valid.dtype==torch.bool
              and valid.shape==earliest_layer_h.shape[:2]
              and bool(valid.any(-1).all())
              and bool(torch.isfinite(earliest_layer_h).all()),
              "only original layer0 causal native BPE real h320 permitted")
        b,t,_=earliest_layer_h.shape
        count=valid.long().sum(-1)
        check(bool((valid==(
                  torch.arange(t,device=valid.device)[None,:]<
                  count[:,None])).all())
              and bool((count>=2).all()),
              "only contiguous causal source prefix, cannot use future/padding")
        # Only last available position in each row can be modified.
        # Previous original causal hidden states are always byte-identical.
        row=torch.arange(b,device=valid.device)
        last=count-1
        norm=self.norm(earliest_layer_h)
        v=self.value(norm)
        direct=v[row,last].unsqueeze(1)
        if contextual:
            q=self.query(norm[row,last]).unsqueeze(1)
            k=self.key(norm)
            pair=(q*k).sum(-1)/math.sqrt(STRUCT_WIDTH)
            pretrained_arcs=graph.role_arc_logits.float()
            check(pretrained_arcs.shape==(b,8,t,t)
                  and bool(torch.isfinite(pretrained_arcs).all()),
                  "must reuse full 8 genuine learned OLD NRA3 graph arcs")
            role_at_last=pretrained_arcs[row,:,last,:]
            causal_valid=(torch.arange(t,device=valid.device)[None,:]
                          <=last[:,None])&valid
            attention=role_at_last.masked_fill(
                ~causal_valid[:,None,:],-1e4).softmax(-1)
            prior=torch.einsum("r,brt->bt",
                self.role_mix.softmax(-1),attention)
            logits=pair+0.5*prior.clamp_min(1e-8).log()
            logits=logits.masked_fill(~causal_valid,-1e4)
            weights=logits.softmax(-1)
            contextual_value=torch.einsum("bt,btd->bd",weights,v).unsqueeze(1)
        else:
            # Exact SAME parameter budget, data, train steps and optimizer.
            # Context/relations are ablated; all other paths identical.
            contextual_value=direct
        g=graph
        event=g.event_state_logits[row,last].float().softmax(-1)
        mini=g.mini_area_logits[row,last].float().sigmoid()
        memory=g.memory_slot_attention[row,last].float()
        features=torch.cat((event,mini,memory),dim=-1)
        check(features.shape==(b,30),"learned old event/mini8/memory required")
        delta=self.fuse(F.gelu(
            contextual_value+self.semantic(features).unsqueeze(1)))
        out=earliest_layer_h.clone()
        out[row,last]+=delta[:,0]
        check(bool(torch.isfinite(out).all())
              and torch.equal(out[:,:-1],
                              earliest_layer_h[:,:-1])
              if bool((count==t).all()) else bool(torch.isfinite(out).all()),
              "original prefix previous positions must stay unchanged")
        return out

class UnifiedCalibriWithStructuredInterface(UnifiedCalibriWithNPA1):
    """One S119g Calibri in-weight research child, adds a causal RS operator."""
    def __init__(self):
        super().__init__()
        self.structured_interface=ContextualStructuredInterface()
        # Auxiliary source-only next-state contrastive prediction, not
        # the real original native BPE vocabulary / generation head.
        self.next_state_projector=nn.Linear(32,HIDDEN,bias=False)

    def load_exact_s119g(self,original:dict):
        old={}
        for k,v in original.items():
            check(k.startswith(PARENT_PREFIX) and
                  v.dtype==torch.float16,"only S119g full original FP16 body")
            old[k[len(PARENT_PREFIX):]]=v.float()
        current=self.state_dict()
        expected={k for k in current
                  if not k.startswith(("structured_interface.",
                                        "next_state_projector."))}
        check(set(old)==expected
              and sum(v.numel() for v in old.values())==PARENT_PARAMETERS,
              "full original S119g must be loaded without shrinking brain")
        result=self.load_state_dict(old,strict=False)
        check(not result.unexpected_keys
              and set(result.missing_keys)==set(current)-expected,
              "only NEW contextual RS + contrastive probe head may initialize")
        for p in self.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        check(sum(p.numel()
                  for p in self.nra3_neural_weights.parameters())==FROZEN_PRETRAINED
              and not any(p.requires_grad for p in
                          self.nra3_neural_weights.parameters()),
              "old in-weight NRA3 S115 53751 incl eight micro-area must remain")

    def forward(self,h,valid,*,contextual:bool):
        check(h.ndim==4 and h.shape[1]==12 and h.shape[-1]==HIDDEN
              and h.shape[0]==valid.shape[0] and h.shape[2]==valid.shape[1]
              and valid.dtype==torch.bool
              and bool((valid==torch.ones_like(valid)).all()),
              "joint RESEARCH interface only full native BPE prefix")
        # Original 12-layer outputs are physically frozen. The old graph
        # informs the new structured interface, not an external parser.
        old_graph=self.nra3_neural_weights(h,valid)
        adapted=self.structured_interface(
            h[:,0],old_graph,valid,contextual=contextual)
        enhanced=torch.cat((adapted.unsqueeze(1),h[:,1:]),dim=1)
        graph,rel,plan=super().forward(enhanced,valid)
        projected=self.next_state_projector(plan.state_trace[:,-1])
        return graph,rel,plan,projected

    def new_joint_trainables(self):
        return (super().new_joint_trainables()
                +list(self.structured_interface.parameters())
                +list(self.next_state_projector.parameters()))


def calculate_example(model,h,valid,prefix:int,*,contextual:bool):
    """True original NEXT hidden at prefix is target only, NEVER model input."""
    check(h.shape[0]==1 and valid.shape==(1,h.shape[2])
          and prefix>=2 and h.shape[2]>=prefix+4
          and bool(valid.all()),
          "need strictly real original12 TRAIN-A prefix and 4 future targets")
    prefix_h=h[:,:,:prefix]
    mask=valid[:,:prefix]
    # Full original next hidden (post-layer11) only as target.
    candidates=h[:,-1,prefix:prefix+4].detach().clone()[0]
    with torch.no_grad():
        target_event,target_relation,old_mini=teacher_targets(
            model,h,valid,prefix)
    graph,rel,plan,pred=model(prefix_h,mask,contextual=contextual)
    # Real next-state discrimination: actual position prefix against three
    # other native BPE positions. Not a generated original-vocab task.
    scores=12.0*F.normalize(pred,dim=-1)@F.normalize(candidates,dim=-1).T
    classification=F.cross_entropy(scores,torch.zeros(1,dtype=torch.long))
    event_loss=-(target_event*F.log_softmax(
        plan.next_event_logits,-1)).sum(-1).mean()
    rel_loss=-(target_relation*F.log_softmax(
        rel,-1)).sum(-1)[mask].mean()
    mini_loss=(graph.mini_area_logits[mask]-
               old_mini[mask]).square().mean()
    joint_loss=classification+0.20*event_loss+0.04*rel_loss+0.01*mini_loss
    check(bool(torch.isfinite(joint_loss)),
          "true original12 matched real next representation proxy finite")
    return joint_loss,{
        "next_original_hidden_contrastive_ce":float(classification.detach()),
        "next_original_hidden_top1_of_four":int(scores.argmax(-1).item()==0),
        "source_only_event_soft_ce":float(event_loss.detach()),
        "source_only_role_relational_ce":float(rel_loss.detach()),
        "mini8_retention_MSE":float(mini_loss.detach()),
    }


def select_rows(mask,segment,n=3):
    eligible=[]
    for row in segment:
        size=int(mask[row].sum().item())
        if size>=28:
            check(bool(mask[row,:size].all())
                  and not bool(mask[row,size:].any()),
                  "actual original native BPE row must be contiguous")
            eligible.append((row,size))
            if len(eligible)==n:
                break
    check(len(eligible)==n,"frozen source must yield n precommitted rows")
    return eligible


def evaluate(model,rows,h,valid,contextual):
    model.eval()
    examples=[]
    with torch.no_grad():
        for idx,n in rows:
            source=h[idx:idx+1,:,:n].float()
            mask=valid[idx:idx+1,:n]
            limit=min(16,n-4)
            loss,details=calculate_example(
                model,source,mask,limit,contextual=contextual)
            examples.append({"source_row":idx,
                             "native_BPE_count":n,
                             "loss":float(loss),
                             **details})
    return {
        "mean_joint_source_proxy_loss":sum(x["loss"] for x in examples)/len(examples),
        "next_hidden_top1_at_four":sum(x["next_original_hidden_top1_of_four"]
                                       for x in examples),
        "next_hidden_total":len(examples),
        "rows":examples,
    }


def train_variant(model,rows,h,valid,contextual):
    model.train()
    opt=torch.optim.Adam(model.new_joint_trainables(),lr=0.0006)
    trace=[]
    original_s115={k:v.detach().half().clone()
                   for k,v in model.nra3_neural_weights.state_dict().items()}
    count=0
    for epoch in range(2):
        for row,n in rows:
            source=h[row:row+1,:,:n].float()
            mask=valid[row:row+1,:n]
            prefix=min(16,n-4)
            opt.zero_grad(set_to_none=True)
            loss,details=calculate_example(
                model,source,mask,prefix,contextual=contextual)
            loss.backward()
            # After first two steps, must jointly reach all mechanisms.
            probes={
                "structured_to_core":
                    model.structured_interface.fuse.weight.grad,
                "new_rank4_core":
                    model.core_growth.up.grad,
                "NRA3R_reasoning_gate":
                    model.reasoning_role_gate.grad,
                "NPA1_recurrent_plan":
                    model.planning_action_area.pred_next_event.weight.grad,
                "next_actual_original_hidden_prediction":
                    model.next_state_projector.weight.grad,
            }
            if count>=1:
                for name,grad in probes.items():
                    check(grad is not None
                          and bool(torch.isfinite(grad).all())
                          and float(grad.abs().sum())>1e-11,
                          "single shared loss must train "+name)
            trace.append({"epoch":epoch+1,"row":row,
                          "objective":float(loss.detach()),
                          **details})
            torch.nn.utils.clip_grad_norm_(
                model.new_joint_trainables(),max_norm=1.0)
            opt.step()
            count+=1
    check(all(torch.equal(original_s115[k],v.detach().half())
              for k,v in model.nra3_neural_weights.state_dict().items()),
          "all exact old NRA3 FP16 53751 MUST be retained")
    check(bool(torch.count_nonzero(model.structured_interface.fuse.weight))
          and bool(torch.count_nonzero(model.core_growth.up))
          and bool(torch.count_nonzero(model.reasoning_role_gate)),
          "new structural, rank4 core and graph must participate")
    return trace


def run(prior:Path, prior_report:Path, source:Path, source_report:Path,
        out:Path):
    check(not out.exists()
          and prior.name==PARENT_CHECKPOINT
          and prior_report.name==PARENT_REPORT
          and source.name==PARENT_H
          and source_report.name==SOURCE_REPORT
          and sha(source)==PARENT_H_SHA,
          "only genuine S119g one-module and exact SHA-pinned real S114 TRAIN-A")
    parent=json.loads(prior_report.read_text(encoding="utf8"))
    provenance=json.loads(source_report.read_text(encoding="utf8"))
    check(parent["stage"]==PARENT_STAGE
          and parent["physical_single_child_SHA256"]==sha(prior)
          and parent["physical_single_child_all_FP16_parameters"]==116777
          and parent["all_53751_original_NRA2_NRA3_params_preserved_byte_exact"]
          and parent["candidate_accepted"] is False
          and parent["output_gate"]==0
          and provenance["ALL12_CPU_SAFETENSORS_SHA256"]==PARENT_H_SHA
          and provenance["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"] is True
          and provenance["candidate_accepted"] is False,
          "physical real original12 S114 / exact S119g neural heritage required")
    torch.set_num_threads(2)
    torch.manual_seed(119121)
    f=load_file(str(source),device="cpu")
    hidden=f["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    mask=f["native_BPE_source_valid_mask"]
    ids=f["original_native_BPE_token_ids"]
    check(hidden.shape[0]==EXPECTED_S114_ROWS
          and hidden.shape[1]==12 and hidden.shape[-1]==HIDDEN
          and hidden.dtype==torch.float16
          and mask.dtype==torch.bool
          and mask.shape==ids.shape==(
              EXPECTED_S114_ROWS,hidden.shape[2])
          and ids.dtype==torch.int32
          and torch.equal(ids[~mask],torch.full_like(ids[~mask],-1)),
          "genuine original CB59-GT all12 native token ids and labels aligned")
    train_rows=select_rows(mask,range(64,96),3)
    probe_rows=select_rows(mask,range(128,160),3)
    check(not (set(i for i,_ in train_rows)&
               set(i for i,_ in probe_rows)),
          "STRICT source-TRAIN cohort separation for narrow internal control")
    original_fp16=load_file(str(prior),device="cpu")
    model=UnifiedCalibriWithStructuredInterface()
    model.load_exact_s119g(original_fp16)
    # Fail closed: zero residual must preserve full prior S119g source
    # joint graph + planner identically, not just a cosmetic tensor output.
    src0=hidden[train_rows[0][0]:train_rows[0][0]+1,:,:8].float()
    valid0=torch.ones((1,8),dtype=torch.bool)
    model.eval()
    with torch.inference_mode():
        g0,r0,p0=UnifiedCalibriWithNPA1.forward(model,src0,valid0)
        g1,r1,p1,_=model(src0,valid0,contextual=True)
        check(torch.equal(g0.mini_area_logits,g1.mini_area_logits)
              and torch.equal(g0.role_arc_logits,g1.role_arc_logits)
              and torch.equal(r0,r1)
              and torch.equal(p0.next_event_logits,p1.next_event_logits),
              "before training new contextual structure MUST preserve all old signals")
    local=copy.deepcopy(model)
    original_probe=evaluate(model,probe_rows,hidden,mask,True)
    check(original_probe==evaluate(local,probe_rows,hidden,mask,False),
          "matched equal initialization must yield identical probe-before data")
    start=time.perf_counter()
    contextual_trace=train_variant(model,train_rows,hidden,mask,True)
    contextual_elapsed=time.perf_counter()-start
    start=time.perf_counter()
    local_trace=train_variant(local,train_rows,hidden,mask,False)
    local_elapsed=time.perf_counter()-start
    evaluated_context=evaluate(model,probe_rows,hidden,mask,True)
    evaluated_local=evaluate(local,probe_rows,hidden,mask,False)
    for k,value in original_fp16.items():
        if ".nra3_neural_weights." in k:
            key=k[len(PARENT_PREFIX):]
            check(torch.equal(model.state_dict()[key].detach().half(),value)
                  and torch.equal(local.state_dict()[key].detach().half(),value),
                  "full original old 53751 trained neural weights must remain")
    physical={PARENT_PREFIX+k:v.detach().half().cpu().contiguous()
              for k,v in model.state_dict().items()}
    local_physical={PARENT_PREFIX+k:v.detach().half().cpu().contiguous()
                    for k,v in local.state_dict().items()}
    check(sum(x.numel() for x in physical.values())==
          sum(x.numel() for x in local_physical.values())
          and set(physical)==set(local_physical)
          and sum(v.numel() for v in original_fp16.values())==PARENT_PARAMETERS,
          "MATCHED PARAMETER COUNT in structure vs local counterfactual")
    out.mkdir(parents=True)
    checkpoint=out/CHECKPOINT
    save_file(physical,str(checkpoint))
    recovered=load_file(str(checkpoint),device="cpu")
    check(set(recovered)==set(physical)
          and all(torch.equal(recovered[k],v) for k,v in physical.items()),
          "physical trained contextual model must reload exact FP16")
    report={
        "stage":STAGE,
        "original_S114_real12_SHA256":PARENT_H_SHA,
        "inherited_real_S119g_full_FP16_SHA256":sha(prior),
        "model_full_earlier_NRA3_53751_FP16_exact":True,
        "new_structured_contextual_area_parameter_count":
            sum(p.numel() for p in model.structured_interface.parameters()),
        "new_source_next_state_probe_linear_parameter_count":
            sum(p.numel() for p in model.next_state_projector.parameters()),
        "both_ablation_models_have_exactly_matched_parameters":True,
        "train_ROWS_S114_TRAIN_ONLY":train_rows,
        "probe_ROWS_S114_TRAIN_ONLY_not_an_independent_heldout_GOLD":probe_rows,
        "untrained_source_probe":original_probe,
        "trained_contextual_source_probe":evaluated_context,
        "trained_token_local_source_probe":evaluated_local,
        "train_structured_trace":contextual_trace,
        "train_token_local_trace":local_trace,
        "contextual_vs_local_source_proxy_objective_delta":
            evaluated_local["mean_joint_source_proxy_loss"]-
            evaluated_context["mean_joint_source_proxy_loss"],
        "contextual_WINS_on_narrow_seen_corpus_probe_only":
            evaluated_context["mean_joint_source_proxy_loss"]<
            evaluated_local["mean_joint_source_proxy_loss"],
        "contextual_train_CPU_seconds_not_decoder_latency":contextual_elapsed,
        "local_train_CPU_seconds_not_decoder_latency":local_elapsed,
        "actual_original_CB59_GT_full_new_forward_NOT_DONE":True,
        "early_real_layer0_capture_is_NOT_true_pre_embedding":True,
        "new_real_next_BPE_softmax_generation_NOT_TRAINED":True,
        "real_fully_generated_universal_answers_NOT_TESTED":True,
        "independent_S118_B_GOLD_never_opened":True,
        "all_in_packed_bpp_NOT_MEASURED":True,
        "paired_original_CPU_decode_NOT_MEASURED":True,
        "candidate_exact616_J7_not_REPLAYED":True,
        "physical_neural_child_params":sum(v.numel() for v in physical.values()),
        "physical_neural_child_bytes":checkpoint.stat().st_size,
        "physical_neural_child_SHA256":sha(checkpoint),
        "output_gate":0,"activation_allowed":False,
        "candidate_accepted":False,
        "decision":"ACTUAL_ORIGINAL12_SOURCE_STRUCTURED_RELATIONS_VS_EQUAL_PARAM_LOCAL_PROXY_ONLY",
    }
    (out/REPORT).write_text(json.dumps(report,indent=2,sort_keys=True,
                    ensure_ascii=False)+"\n",encoding="utf8")
    print("S119I_REAL12_STRUCTURED_VS_LOCAL_ABLATION "+
          json.dumps({
             "real_original12_train_rows":train_rows,
             "real_original12_source_probe_rows":probe_rows,
             "contextual_source_proxy_loss":
                 evaluated_context["mean_joint_source_proxy_loss"],
             "matched_local_source_proxy_loss":
                 evaluated_local["mean_joint_source_proxy_loss"],
             "contextual_proxy_better":report[
                 "contextual_WINS_on_narrow_seen_corpus_probe_only"],
             "full_old_53751_FP16_exact":True,
             "total_child_parameters":report["physical_neural_child_params"],
             "physical_child_bytes":checkpoint.stat().st_size,
             "candidate_accepted":False},sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    for name in ("prior","prior_report","source","source_report","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    z=p.parse_args()
    run(z.prior,z.prior_report,z.source,z.source_report,z.out)

if __name__=="__main__":
    main()
