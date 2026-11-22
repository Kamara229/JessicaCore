"""S119q2f: trainable CAUSAL early-layer learned projection into NRA3/NPA1.

Actual ORIGINAL frozen S142 43M, Core12+GreenTunnel, full 49,152 native
LM-head, and a physically trained q2d contextual NRA3/NPA1/Core child.

The learned low-rank 8-dimensional representation uses ONLY h0..h5 that
actually exist after original decoder layer5. The projected features live
in virtual slots6..11 of the existing learned NRA3 joint region's 12-slot
interface. NEVER claim to have physically seen original future h6..h11.

Controlled A/B: duplicate the SAME physically trained q2d contextual FP16
child and same new random low-rank projection parameters, then train
both on 96 previous (authored synthetic) TRAIN records for two epochs:
  CONTEXTUAL: early-source trainable layer mixture h0..h5 and contextual
    structured NRA3 and NPA1.
  TOKEN_LOCAL: exactly same parameters, trainable low-rank projection
    operates on six replicas of h5; structured module contextual=False.
96x2 original native forward + backward each, 4 gradient accumulated
steps, 48 optimizer updates per variant, same optimizer order/seeds.
Source-only diagnostic uses 18 old q2a CALIBRATION and 12 old q2c1
SOURCE_PROBE records and is NOT independent quality testing or blind QA.
No sealed S118 B GOLD or spent EVAL128, no Jessica chat, LIVE, Render,
Supabase, main, APK, CPU/bpp/J7 promotion. ZERO activation gate always.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    persist_half,verify_q2a_artifact,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    source_inventory,pack_sources,scheduled_position,native_step,
    VOCAB_SIZE,PROBE_COUNTS,CONTEXT_CKPT as PARENT_CONTEXT_CKPT,
    LOCAL_CKPT as PARENT_LOCAL_CKPT,REPORT as PARENT_REPORT,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)

STAGE="CALIBRI_S119Q2F_REAL43M_GT_LEARNED_CAUSAL_H0H5_REPRESENTATION"
REPORT="S119Q2F_LEARNED_CAUSAL_PROJECTOR_NATIVE_JOINT_A_B_REPORT.json"
CONTEXT_CKPT="S119Q2F_CONTEXTUAL_CAUSAL_PROJECTOR_FP16_UNACCEPTED.safetensors"
LOCAL_CKPT="S119Q2F_LOCAL_MATCHED_CAUSAL_PROJECTOR_FP16_UNACCEPTED.safetensors"
SEED=119136
RANK=8
TRAIN_ROWS=96
EPOCHS=2
ACCUM=4
LR=0.0005
SCALE=0.05
SOURCE_PROBES=30
EXPECTED_NEW_PARAMETERS=320*RANK*2+6*6+6
EXPECTED_OLD_NEURAL_WEIGHTS=53751
EXPECTED_TRAIN_DECODER_FORWARDS=TRAIN_ROWS*EPOCHS*2
EXPECTED_DIAGNOSTIC_FORWARDS=SOURCE_PROBES*4
EXPECTED_ACTUAL_DECODER_FORWARDS=(EXPECTED_TRAIN_DECODER_FORWARDS+
                                  EXPECTED_DIAGNOSTIC_FORWARDS)


class LearnedCausalEarlyProjector(nn.Module):
    """Trainable low-rank per-slot causal h0..5 projector on q2d child.

    The first six slots remain EXACT actual early states, and original
    frozen layers6..11 still execute after the layer5 modification.
    Repeated h5 pseudo-slots remain identity baseline plus projected
    h0..5-only information. This is one ORIGINAL decoder forward.
    """
    def __init__(self, child):
        super().__init__()
        self.child=child
        self.down=nn.Linear(320,RANK,bias=False)
        self.up=nn.Linear(RANK,320,bias=False)
        self.mix_logits=nn.Parameter(torch.eye(6)*2.0)
        self.slot_gain=nn.Parameter(torch.full((6,),0.2))
        nn.init.normal_(self.down.weight,mean=0.,std=0.02)
        nn.init.normal_(self.up.weight,mean=0.,std=0.02)
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.last_projected_H320_norm=None
        check(not self.child.candidate_accepted and
              not self.child.activation_allowed and
              self.child.output_gate==0,
              "only physical research-only unaccepted original q2d child")

    def representation(self,physical12,*,contextual):
        check(physical12.shape[0]==1 and physical12.shape[1]==12
              and physical12.shape[-1]==320 and physical12.shape[2]>=2
              and bool(torch.isfinite(physical12).all()),
              "true early original 12slot h320 ABI only")
        early=physical12[:,:6]
        h5=physical12[:,5:6]
        check(torch.equal(physical12[:,6:],h5.expand_as(physical12[:,6:])),
              "NO hidden future h6-11 or replacing physical causal h5 prior")
        source=(early if contextual else h5.expand_as(early))
        # Learnable layer mixing on original early h0..h5 for six virtual
        # source slots. Token-local control only uses copies of actual h5.
        mixing=torch.softmax(self.mix_logits,dim=-1)
        mixed=torch.einsum("ij,bjth->bith",mixing,source)
        compressed=F.gelu(self.down(F.layer_norm(mixed,(320,))))
        projected=self.up(compressed)
        scaled=SCALE*torch.tanh(self.slot_gain).view(1,6,1,1)*projected
        reconstructed=torch.cat((
            early,physical12[:,6:]+scaled),dim=1)
        check(reconstructed.shape==physical12.shape and
              torch.equal(reconstructed[:,:6],early) and
              bool(torch.isfinite(reconstructed).all()),
              "trainable virtual early-only h6..11 without modifying real h0..5")
        self.last_projected_H320_norm=float(
            torch.linalg.vector_norm(scaled[:,:,-1,:].detach()).item())
        return reconstructed

    def forward(self,physical12,*,contextual):
        check(self.output_gate==0 and
              not self.candidate_accepted and
              not self.activation_allowed and
              not self.child.candidate_accepted,
              "research output gate0 and unaccepted child")
        x=self.representation(physical12,contextual=contextual)
        return self.child(x,contextual=contextual)

    def trainables(self):
        params=list(self.child.trainables())+[
            self.down.weight,self.up.weight,
            self.mix_logits,self.slot_gain,
        ]
        check(len(params)==len({id(p) for p in params}) and
              all(p.requires_grad for p in params),
              "no repeated original protected weights or frozen parameters")
        return params


def physical_train_one(model,sources,bridge,full,expansion,adapter,train,
                       *,contextual):
    check(len(train)==TRAIN_ROWS and
          all(x["source_split"]=="TRAIN" for x in train),
          "only 96 original-authored q2a and q2c1 TRAIN examples")
    params=adapter.trainables()
    all_count=sum(x.numel() for x in params)
    check(sum(x.numel() for x in (
        adapter.down.weight,adapter.up.weight,
        adapter.mix_logits,adapter.slot_gain))==EXPECTED_NEW_PARAMETERS,
        "same exact small causal rank8 layer-mixer parameter overhead")
    old_neural={
        k:v.detach().half().cpu().clone()
        for k,v in
        adapter.child.shadow.unified.nra3_neural_weights.state_dict().items()
    }
    check(sum(v.numel() for v in old_neural.values())==
          EXPECTED_OLD_NEURAL_WEIGHTS,
          "protected original S115 53751 FP16")
    init_projected={k:v.detach().cpu().clone()
                    for k,v in adapter.state_dict().items()}
    optimizer=torch.optim.Adam(params,lr=LR)
    report_epochs=[];n_forwards=0;updates=0
    learned_gradient=None
    for epoch in range(EPOCHS):
        ce_array=[];greedy=0;contrasts=0
        for i,entry in enumerate(train):
            pos=scheduled_position(entry,epoch)
            loss,stats=native_step(
                model,sources,bridge,full,expansion,adapter,entry,pos,
                contextual=contextual,gradient=True)
            (loss/ACCUM).backward()
            n_forwards+=1
            ce_array.append(stats["next_native_CE"])
            greedy+=int(stats["correct_native_greedy_token"])
            contrasts+=int(stats["contrast_applied"])
            if learned_gradient is None and i>=ACCUM:
                assert_list={
                    "causal_projection_down":adapter.down.weight.grad,
                    "causal_projection_up":adapter.up.weight.grad,
                    "causal_projection_slot_gate":adapter.slot_gain.grad,
                    "rank4_core":adapter.child.shadow.unified.core_growth.up.grad,
                    "structured_NRA3":
                        adapter.child.shadow.unified.structured_interface.fuse.weight.grad,
                    "planning_NPA1":
                        adapter.child.shadow.unified.planning_action_area.state_update.weight_ih.grad,
                }
                learned_gradient={}
                for key,g in assert_list.items():
                    check(g is not None and bool(torch.isfinite(g).all())
                          and float(g.abs().sum())>1.e-20,
                          "real original native full 49k backward reaches "+key)
                    learned_gradient[key]=float(
                        torch.linalg.vector_norm(g).item())
                learned_gradient["h0h5_learnable_mixer_grad_norm"]=float(
                    torch.linalg.vector_norm(
                        adapter.mix_logits.grad).item())
            if (i+1)%ACCUM==0:
                torch.nn.utils.clip_grad_norm_(params,1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates+=1
                check(all(bool(torch.isfinite(p).all()) for p in params),
                      "no NaN/Inf in original 12layer native learned projection")
        report_epochs.append({
            "epoch":epoch+1,
            "actual_original_full_native_decoder_training_calls":TRAIN_ROWS,
            "optimizer_updates":TRAIN_ROWS//ACCUM,
            "source_train_conditional_mean_next_BPE_CE":
                sum(ce_array)/len(ce_array),
            "source_train_target_exact_greedy_count":greedy,
            "first_counterfactual_answer_divergent_BPE_losses":contrasts,
            "answer_positions_rotated_so_train_epoch_CE_not_comparable":True,
        })
    check(updates==TRAIN_ROWS//ACCUM*EPOCHS and
          n_forwards==TRAIN_ROWS*EPOCHS and
          learned_gradient is not None,
          "real original decoder forward backprop and paired updates complete")
    check(all(torch.equal(old_neural[k],v.detach().half().cpu())
              for k,v in
              adapter.child.shadow.unified.nra3_neural_weights.state_dict().items()),
          "old original 53751 pretrained FP16 neural S115 unchanged")
    n_changed=sum(
        not torch.equal(init_projected[k],v.detach().cpu())
        for k,v in adapter.state_dict().items()
    )
    check(n_changed>0 and
          not torch.equal(init_projected["up.weight"],
                          adapter.up.weight.detach().cpu()),
          "learned causal projector up weights actually got optimizer updates")
    return {
        "epochs":report_epochs,"original_decoder_forward_calls":n_forwards,
        "optimizer_updates":updates,"trainable_parameters":all_count,
        "new_projector_parameters":EXPECTED_NEW_PARAMETERS,
        "actual_changed_module_tensors":n_changed,
        "joint_gradient_norms":learned_gradient,
        "final_slot_gains":adapter.slot_gain.detach().cpu().tolist(),
        "trained_on_synthetic_original_author_SOURCE_TRAIN_only":True,
    }


def source_diagnostic_readonly(model,sources,bridge,full,expansion,
                               adapter,probes,*,contextual):
    check(len(probes)==SOURCE_PROBES and
          Counter(x["source_split"] for x in probes)==PROBE_COUNTS,
          "only old nonblind 18 CAL +12 original-authored SOURCE_PROBE")
    domain_ce={};trace=[]
    for e in probes:
        p=e["first_distinct_position"] if e["source_family"]=="q2c1" else 0
        _,stat=native_step(model,sources,bridge,full,expansion,adapter,
                           e,p,contextual=contextual,gradient=False)
        trace.append(stat)
        domain_ce.setdefault(stat["kind"],[]).append(stat["next_native_CE"])
    return {
        "n_source_only_probes":len(trace),
        "mean_conditional_native_CE":sum(x["next_native_CE"]
                                        for x in trace)/len(trace),
        "exact_greedy_native_token_targets":
            sum(int(x["correct_native_greedy_token"]) for x in trace),
        "same_30_old_source_probes_not_independent":True,
        "source_only_per_kind_CE":{k:sum(v)/len(v)
                                   for k,v in sorted(domain_ce.items())},
        "unchanged_prior_conditioned_answer_prefix_target_semantics":True,
    }


def run(s93,side,s142,q2a_dir,q2d_dir,reference,out):
    check(not out.exists(),"new isolated research-only output, no overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    parent=load_q2d_proof(
        q2d_dir/PARENT_REPORT,
        q2d_dir/PARENT_CONTEXT_CKPT,
        q2d_dir/PARENT_LOCAL_CKPT,s93,s142)
    source_rows,source_manifest=verify_q2a_artifact(q2a_dir)
    check(parent["q2a_source_SHA256"]==
          source_manifest["synthetic_rows_SHA256"],
          "original authored q2a fixed SHA provenance")
    reference_data=json.loads(reference.read_text(encoding="utf8"))
    check(reference_data["q2d_real_research_source_report_SHA256"]==
          sha(q2d_dir/PARENT_REPORT)
          and reference_data["candidate_accepted"] is False
          and reference_data["summary"]["all_results_only_previously_used_same_author_probes"],
          "original frozen real source baseline report exact parent SHA")
    q2d_parent_context,_=load_q2d_trained_child(
        q2d_dir/PARENT_CONTEXT_CKPT,
        parent["new_research_checkpoint_contextual"])
    torch.manual_seed(SEED+1)
    model_ctx=LearnedCausalEarlyProjector(q2d_parent_context)
    model_local=copy.deepcopy(model_ctx)
    check(all(torch.equal(a.detach(),b.detach()) for a,b
              in zip(model_ctx.state_dict().values(),
                     model_local.state_dict().values())),
          "SAME q2d actual trained contextual initialization PLUS same projector")
    check(sum(p.numel() for p in model_ctx.trainables())==
          sum(p.numel() for p in model_local.trainables()) and
          len(model_ctx.trainables())==len(model_local.trainables()),
          "exact match trainable budget A/B even local unused layer mixer")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE,"full original native 49152 head only")
    train_raw,probes_raw=source_inventory(source_rows)
    train,probes=pack_sources(tok,train_raw,probes_raw)
    check(len(train)==TRAIN_ROWS and len(probes)==SOURCE_PROBES
          and {x["prompt_sha256"] for x in train}.isdisjoint(
              x["prompt_sha256"] for x in probes),
          "96 actual original SOURCE_TRAIN and 30 previously exposed CAL/probes")
    original,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    original.model.add_module("nra3_s119m_onpath",model_ctx)
    original.eval()
    original_sha=old_original_param_digest(original)
    check(all(not p.requires_grad for n,p in original.named_parameters()
              if "nra3_s119m_onpath" not in n),
          "ALL S142 12layer original Core+GT original weights frozen")
    before_ctx=source_diagnostic_readonly(
        original,sources,bridge,full,expansion,model_ctx,probes,
        contextual=True)
    before_local=source_diagnostic_readonly(
        original,sources,bridge,full,expansion,model_local,probes,
        contextual=False)
    result_ctx=physical_train_one(
        original,sources,bridge,full,expansion,model_ctx,train,
        contextual=True)
    result_local=physical_train_one(
        original,sources,bridge,full,expansion,model_local,train,
        contextual=False)
    after_ctx=source_diagnostic_readonly(
        original,sources,bridge,full,expansion,model_ctx,probes,
        contextual=True)
    after_local=source_diagnostic_readonly(
        original,sources,bridge,full,expansion,model_local,probes,
        contextual=False)
    check(result_ctx["trainable_parameters"]==
          result_local["trainable_parameters"] and
          result_ctx["optimizer_updates"]==
          result_local["optimizer_updates"]==
          (TRAIN_ROWS//ACCUM)*EPOCHS and
          result_ctx["original_decoder_forward_calls"]==
          result_local["original_decoder_forward_calls"]==
          TRAIN_ROWS*EPOCHS and
          old_original_param_digest(original)==original_sha,
          "real native paired A/B matched TRAIN and protected ORIGINAL exact")
    for mdl in (model_ctx,model_local):
        check(mdl.output_gate==0 and not mdl.candidate_accepted
              and not mdl.activation_allowed and
              mdl.child.output_gate==0
              and not mdl.child.candidate_accepted,
              "no research model ever activated or deployed")
    out.mkdir(parents=True)
    checkpoint_ctx=persist_half(model_ctx,out/CONTEXT_CKPT)
    checkpoint_local=persist_half(model_local,out/LOCAL_CKPT)
    result={
        "stage":STAGE,
        "original_S93_SHA256":sha(s93),
        "original_S142_GT_SHA256":sha(s142),
        "parent_q2d_training_report_SHA256":sha(q2d_dir/PARENT_REPORT),
        "parent_q2d_contextual_FP16_SHA256":
            parent["new_research_checkpoint_contextual"]["sha256"],
        "original_frozen_native_source_baseline_report_SHA256":sha(reference),
        "original_q2a_synthetic_source_SHA256":
            source_manifest["synthetic_rows_SHA256"],
        "full_original_native_BPE_vocab":VOCAB_SIZE,
        "original_12layer_GreenTunnel_original_core_weights_unchanged":True,
        "old_S115_53751_FP16_neural_weights_unchanged":True,
        "exact_same_q2d_TRained_contextual_initial_weights_for_both_A_B":True,
        "all_early_state_sources_only_REAL_h0_h5_after_layer5":True,
        "virtual_late_slots6_11_are_trainable_early_lowrank_not_real_future_layers":True,
        "learned_rank_limited_projector_rank":RANK,
        "new_neural_projector_parameter_overhead":EXPECTED_NEW_PARAMETERS,
        "same_trainable_parameters_and_optimizer_A_B":True,
        "both_train_original_source_96x2_only":True,
        "teacher_forced_positions_rotate_and_are_not_free_running":True,
        "no_new_blind_test_no_sealed_gold_spent_EVAL128_J7":True,
        "source_only_30_previously_exposed_not_independent":True,
        "source_probe_used_for_REPORT_only_no_hyperparam_selection":True,
        "physically_verified_original_single_pass_per_example":True,
        "before_training_contextual_source_probe":before_ctx,
        "before_training_local_source_probe":before_local,
        "trained_contextual":result_ctx,
        "trained_tokenlocal_matched":result_local,
        "after_training_contextual_source_probe":after_ctx,
        "after_training_local_source_probe":after_local,
        "contextual_minus_local_source_only_CE":
            after_ctx["mean_conditional_native_CE"]-
            after_local["mean_conditional_native_CE"],
        "contextual_improvement_vs_itself_before_CE":
            after_ctx["mean_conditional_native_CE"]-
            before_ctx["mean_conditional_native_CE"],
        "local_improvement_vs_itself_before_CE":
            after_local["mean_conditional_native_CE"]-
            before_local["mean_conditional_native_CE"],
        "projector_contextual_child_checkpoint":checkpoint_ctx,
        "projector_local_matched_checkpoint":checkpoint_local,
        "actual_original_decoder_forward_calls":
            EXPECTED_ACTUAL_DECODER_FORWARDS,
        "no_616_fullJ7_EOS_BPP_KVCPU_or_independent_quality_cert":True,
        "no_Jessica_chat_LIVE_Render_Supabase_main_APK":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"LEARNED_CAUSAL_PROJECTOR_SOURCE_TRAIN_MECHANICAL_ONLY_NOT_ACCEPTED",
    }
    (out/REPORT).write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2F_ORIGINAL43M_GT_LEARNED_CAUSAL_PROJECTOR "+
          json.dumps({
            "actual_original_decoder_calls":EXPECTED_ACTUAL_DECODER_FORWARDS,
            "new_projector_parameters":EXPECTED_NEW_PARAMETERS,
            "trainable_parameters_each":result_ctx["trainable_parameters"],
            "actual_optimizer_updates_each":result_ctx["optimizer_updates"],
            "contextual_before_CE":
                before_ctx["mean_conditional_native_CE"],
            "local_before_CE":before_local["mean_conditional_native_CE"],
            "contextual_after_CE":after_ctx["mean_conditional_native_CE"],
            "local_after_CE":after_local["mean_conditional_native_CE"],
            "contextual_wins_source_only":
                after_ctx["mean_conditional_native_CE"]<
                after_local["mean_conditional_native_CE"],
            "original_weights_intact":True,
            "accepted":False,
          },sort_keys=True),flush=True)


def main():
    a=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2a_dir","q2d_dir","reference","out"):
        a.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    x=a.parse_args()
    run(x.s93,x.side,x.s142,x.q2a_dir,x.q2d_dir,x.reference,x.out)


if __name__=="__main__":
    main()
