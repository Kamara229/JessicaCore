"""S119q2i6 TRUE 43M+GT12/Core/NRA3/NPA1/full-native head UTF8 TRAIN.

Experimental language continuation curriculum. Actual original frozen S142
12-layer 43M+GreenTunnel full 49,152-native-vocabulary forward and backward.
Both A/B begin from EXACT SAME previously trained Q2I3 RECURRENT-GT full
neural FP16 + same 12 learned GT gates. RECUR carries GT12 across layers;
LOCAL uses same physical GT12 with no cross-layer carry. Exactly identical
trainable 368,051 params, 3 epochs, 18 NEW authored TRAIN plus SIX older
q2i1 TRAIN (retention rehearsal) examples and full native BPE+EOS.

Novel treatment: if actual byte-level native BPE *gold preceding prefix*
ends mid-UTF8 character, teacher-forced native continuation-target CE is
weighted 3.0x. No inferred gold from DEV, no token masking, no forced
ID155/ID30 bonuses, no naive string byte fragmentation, no grade hacking.
All other native targets 1.0x. This train-only objective exploits the
actual ByteLevel tokenizer, not artificial Unicode transformations.
Rehearsal uses ONLY previously designated TRAIN, never old DEV.

Validation: 6 newly authored DEV + six old q2i1 already scored DEV,
freely-generated original native own greedy BPE, real EOS, no gold prefix.
Small authored and visible DEV != truly independent blind evaluation.
Protected original 616 J7+EOS, packed physical bpp, KV CPU speed gates
remain closed; output_gate=0 candidate_accepted=false. All original weights
unchanged and research sidecars isolated, no production.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
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
    persist_half,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CTX,LOCAL_CKPT as Q2D_LOCAL,
    REPORT as Q2D_REPORT,VOCAB_SIZE,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CTX,REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as old_i1_rows,provenance as old_i1_provenance,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    STAGE as Q2I1_STAGE,CONTEXT_CKPT as Q2I1_CTX,
    REPORT as Q2I1_REPORT,verified_native_answer,real_joint_native,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    load_q2i1_physical_context,native_full_gt_gate_contract,set_gt_route,
    own_token_dev,safe_gt_gate_sidecar,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    STAGE as Q2I3_STAGE,REPORT as Q2I3_REPORT,
    RECURRENT_CKPT,RECURRENT_GATES,HEAD_GAIN,
    scoped_reduced_native_head_gain,make_joint_trainable,
    TOTAL_EXPECTED_TRAINABLE,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import (
    check_trained_joint,verified_gt12_gates,
)
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import (
    native_bytelevel_full_utf8_status,
)
from research.calibri_nra3.s119q2i6_fresh_ru_utf8_curriculum import (
    STAGE as SOURCE_STAGE,
    rows as new_ru_rows,verify as verify_curriculum,
    source_sha as source_sha,
    KINDS,AUTHOR_ORIGIN,
)

STAGE="CALIBRI_S119Q2I6_TRUE43M_GT_NRA3_NPA1_UTF8_MULTIBPE_CONTINUATION_TRAIN"
REPORT="S119Q2I6_REAL43M_FULL_NATIVE_UTF8_3ROUNDS_NEWDEV_OLDDEV_G0.json"
RECURRENT_OUT="S119Q2I6_GT12_RECURRENT_JOINT_UTF8_FP16_G0.safetensors"
LOCAL_OUT="S119Q2I6_GT12_LOCAL_JOINT_UTF8_FP16_G0.safetensors"
RECURRENT_GATE_OUT="S119Q2I6_GT12_RECURRENT_UTF8_GATES_FP16_G0.safetensors"
LOCAL_GATE_OUT="S119Q2I6_GT12_LOCAL_UTF8_GATES_FP16_G0.safetensors"
SEED=119145
ROUNDS=3
NEW_TRAIN=18
REHEARSAL_TRAIN=6
TOTAL_TRAIN=NEW_TRAIN+REHEARSAL_TRAIN
NEW_DEV=6
OLD_DEV=6
DEV_OWN_CAP=12
CONTINUATION_WEIGHT=3.0
GAIN=HEAD_GAIN
REGION_LR=0.00017
GT_GATE_LR=0.003
PROJECTION_LR=0.0008
HEAD_BIAS_LR=0.000025
CLIP=0.6
CONDITIONS=("GT_RECURRENT","GT_NOCARRY_LOCAL")
GATES=12


def prepare_new(tok,row):
    """Exact original BPE chat prefix and complete UTF8 target plus EOS."""
    check(row["split"] in ("TRAIN","DEV")
          and row["origin"]==AUTHOR_ORIGIN
          and row["kind"] in KINDS,
          "only pristine registered author-owned UTF8 TRAIN or DEV")
    # The original q2i1 translator has the exact native prompt-boundary,
    # completion decode and EOS fail-closed checks. Change ONLY its
    # provenance namespace on a local shallow copy, never train on labels
    # from different sources or overwrite the original source.
    translated=dict(row)
    translated["origin"]="original_newly_authored_examples_no_third_party_corpus"
    result=verified_native_answer(tok,translated)
    check(result["id"]==row["id"] and result["kind"]==row["kind"]
          and result["split"]==row["split"] and
          tok.decode(result["answer_native_without_eos"],
                     skip_special_tokens=True)==row["completion"],
          "full source-native BPE target preserved exactly")
    return result


def native_train_weights(tok,answer_native_with_eos,*,continuation_weight=CONTINUATION_WEIGHT):
    """Upweight ONLY the gold *next* BPE after an incomplete UTF8 prefix."""
    check(type(answer_native_with_eos) is list
          and len(answer_native_with_eos)>=2
          and answer_native_with_eos[-1]==tok.eos_token_id,
          "complete correct original native BPE including terminal EOS")
    check(continuation_weight in (1.0,3.0),
          "controlled q2i7 continuation weight 1x/3x only")
    weights=[];statuses=[]
    for t in range(len(answer_native_with_eos)):
        prior=answer_native_with_eos[:t]  # past gold ONLY
        state=native_bytelevel_full_utf8_status(
            tok,prior)["entire_original_native_BPE_prefix_UTF8_status"]
        check(state in (
            "VALID_COMPLETE_UTF8",
            "INCOMPLETE_UTF8_NEEDS_CONTINUATION",
        ),"correct target BPE must never have INVALID UTF8 earlier prefix")
        weight=(continuation_weight
                if state=="INCOMPLETE_UTF8_NEEDS_CONTINUATION" else 1.0)
        if t==len(answer_native_with_eos)-1:
            check(state=="VALID_COMPLETE_UTF8",
                  "native EOS cannot follow incomplete Russian multi-byte character")
        weights.append(weight)
        statuses.append(state)
    check(all(x in (1.,continuation_weight) for x in weights)
          and len(weights)==len(answer_native_with_eos),
          "fixed predeclared weighting on previous gold native UTF8 only")
    return weights,statuses


def train_true_joint(model,sources,bridge,full,expansion,child,train,tok,
                     *,condition,continuation_weight=CONTINUATION_WEIGHT,
                     expected_total=TOTAL_TRAIN,expected_new=NEW_TRAIN,
                     new_id_prefix="S119Q2I6_",rounds=ROUNDS,
                     teacher_135m=None,teacher_KL_weight=0.0):
    check(continuation_weight in (1.0,3.0) and condition in CONDITIONS and
          ((teacher_135m is None and teacher_KL_weight==0.0) or
           (teacher_135m is not None and teacher_KL_weight==0.25 and
            all(not p.requires_grad for p in teacher_135m.parameters()))) and
          rounds in (1,2,3) and expected_new>=NEW_TRAIN and
          expected_total>=expected_new and
          len(train)==expected_total and
          Counter(x["split"] for x in train)=={"TRAIN":expected_total}
          and sum(x["id"].startswith(new_id_prefix) for x in train)==expected_new,
          "ONLY registered fresh new and old TRAIN, never scored DEV labels")
    selected=make_joint_trainable(child)
    check(sum(p.numel() for p in selected)==TOTAL_EXPECTED_TRAINABLE-GATES,
          "original q2i3 region+head full 368039 trainables")
    for name,p in full.named_parameters():
        p.requires_grad_(name=="gates")
    check(full.gates.requires_grad and full.gates.numel()==GATES
          and len(selected)==len({id(p) for p in selected}),
          "only 12 GT12 scalar gates train, original GT donor projections fixed")
    head=child.language_head
    special=[head.down.weight,head.up.weight,head.output_bias]
    nongate=[p for p in selected if id(p) not in {id(q) for q in special}]
    check(sum(p.numel() for p in nongate)+
          sum(p.numel() for p in special)+GATES==TOTAL_EXPECTED_TRAINABLE,
          "strict matched 368051 trainable region+readout+GT12 params")
    opt=torch.optim.Adam([
        {"params":[full.gates],"lr":GT_GATE_LR},
        {"params":nongate,"lr":REGION_LR},
        {"params":[head.down.weight,head.up.weight],"lr":PROJECTION_LR},
        {"params":[head.output_bias],"lr":HEAD_BIAS_LR},
    ])
    initial={
        "GT12":full.gates.detach().clone(),
        "Core":child.child.child.shadow.unified.core_growth.up.detach().clone(),
        "NRA3":child.child.child.shadow.unified.structured_interface.fuse.weight.detach().clone(),
        "NPA1":child.child.child.shadow.unified.planning_action_area.state_update.weight_ih.detach().clone(),
        "rank8_projection":child.child.up.weight.detach().clone(),
        "language_head":head.up.weight.detach().clone(),
    }
    protected={k:v.detach().cpu().half().clone() for k,v in
        child.child.child.shadow.unified.nra3_neural_weights.state_dict().items()}
    epoch_reports=[];steps=0;forwards=0;continuation_targets=0
    physical_teacher_forwards=0;teacher_KL_values=[]
    for epoch in range(rounds):
        all_ce=[];all_cont_ce=[];all_eos_ce=[];epoch_forwards=0
        for row in train:
            gold=row["answer_native_with_eos"]
            weights,statuses=native_train_weights(tok,gold,
                continuation_weight=continuation_weight)
            opt.zero_grad(set_to_none=True)
            with scoped_reduced_native_head_gain(head) as witness:
                for t,target in enumerate(gold):
                    prefix=row["prefix"]+gold[:t]
                    check(prefix[len(row["prefix"]):]==gold[:t]
                          and prefix[:len(row["prefix"])]==row["prefix"]
                          and len(prefix)<160,
                          "causal ORIGINAL full-native decoder uses PREVIOUS gold only")
                    logits=real_joint_native(
                        model,sources,bridge,full,expansion,
                        child,prefix,row["prompt_features"],contextual=True)
                    check(full.last_forward_coverage==tuple(range(12))
                          and not full._active and full._carry is None
                          and logits.shape==(1,VOCAB_SIZE),
                          "REAL original 43M Core12 GT12 native BPE every position")
                    loss=F.cross_entropy(
                        logits,torch.tensor([int(target)],dtype=torch.long))
                    weighted=weights[t]*loss/sum(weights)
                    # PRE-DECLARED optional S119q2i8d SAME physical 49k native
                    # teacher-logit distillation ONLY following an INCOMPLETE
                    # gold UTF8 prefix. EOS stays human-target CE-only.
                    if (teacher_135m is not None and
                        statuses[t]=="INCOMPLETE_UTF8_NEEDS_CONTINUATION"):
                        check(int(target)!=tok.eos_token_id,
                              "NEVER teacher-distill EOS or teacher-generated QA")
                        from research.calibri_nra3.s119q2i8b_source135m_vs_original43m_native import source_teacher_step
                        soft=source_teacher_step(teacher_135m,prefix)
                        check(soft.shape==logits.shape==(1,VOCAB_SIZE),
                              "source135m and student43m actual native 49152 logits")
                        teacher_p=F.softmax(soft.detach().float()/2.0,dim=-1)
                        student_log=F.log_softmax(logits/2.0,dim=-1)
                        KL=F.kl_div(student_log,teacher_p,reduction="sum")*(2.0**2)
                        check(bool(torch.isfinite(KL)) and float(KL.detach())>=-0.00005,
                              "T2 full original native softteacher KL nonnegative")
                        weighted=weighted+teacher_KL_weight*KL/sum(weights)
                        physical_teacher_forwards+=1
                        teacher_KL_values.append(float(KL.detach()))
                    check(bool(torch.isfinite(weighted)),
                          "finite continuation-weighted real original native loss")
                    weighted.backward()
                    ce=float(loss.detach())
                    all_ce.append(ce)
                    if statuses[t]=="INCOMPLETE_UTF8_NEEDS_CONTINUATION":
                        all_cont_ce.append(ce)
                        continuation_targets+=1
                    if t==len(gold)-1:all_eos_ce.append(ce)
                    forwards+=1;epoch_forwards+=1
                    del logits,loss,weighted
            check(witness[0]==len(gold),
                  "identical predeclared readout scale for EVERY native BPE")
            torch.nn.utils.clip_grad_norm_(selected+[full.gates],CLIP)
            opt.step();steps+=1
            check(bool(torch.isfinite(full.gates).all())
                  and all(bool(torch.isfinite(x).all()) for x in selected)
                  and float(full.gates.detach().abs().max())<.5,
                  "true GT12 and full neural weights remain finite/contained")
        check(all_cont_ce and len(all_eos_ce)==expected_total,
              "actual trained Russian multibyte continuation and EOS targets")
        epoch_reports.append({
            "round":epoch+1,
            "actual_original_full_native_43M_decoder_calls":epoch_forwards,
            "optimizer_updates":expected_total,
            "train_mean_native_CE_ALL_target_BPE_EOS":
                sum(all_ce)/len(all_ce),
            "train_mean_native_CE_UTF8_continuation_target":
                sum(all_cont_ce)/len(all_cont_ce),
            "train_mean_native_EOS_CE":
                sum(all_eos_ce)/len(all_eos_ce),
            "continuation_target_count":len(all_cont_ce),
            "identical_full_native_49152_head_and_3x_CE_weights":True,
        })
    changed={}
    after={
        "GT12":full.gates,
        "Core":child.child.child.shadow.unified.core_growth.up,
        "NRA3":child.child.child.shadow.unified.structured_interface.fuse.weight,
        "NPA1":child.child.child.shadow.unified.planning_action_area.state_update.weight_ih,
        "rank8_projection":child.child.up.weight,
        "language_head":head.up.weight,
    }
    for key,v in after.items():
        delta=float(torch.linalg.vector_norm(v.detach()-initial[key]))
        changed[key]=delta
        check(delta>1e-10,
              "ALL GreenTunnel/NRA3/NPA1/CORE/rank8/head must learn: "+key)
    check(all(torch.equal(v.detach().cpu().half(),protected[k])
              for k,v in child.child.child.shadow.unified.nra3_neural_weights.state_dict().items())
          and steps==expected_total*rounds,
          "prior S115 53,751 original neural FP16 unchanged, 3 true rounds")
    return {
        "condition":condition,
        "rounds":epoch_reports,
        "trained_original_decoder_calls":forwards,
        "native_byte_continuation_weight":continuation_weight,
        "count_weighted_native_byte_continuation_targets_across_all_epochs":
            continuation_targets,
        "real_all_neural_and_gt_Adam_updates":steps,
        "actual_gt12_core_NRA3_NPA1_native_head_change_l2":changed,
        "trainable_parameter_count":TOTAL_EXPECTED_TRAINABLE,
        "source135m_teacher_KL_fixed_T2_continuation_only":teacher_KL_weight,
        "physical_SOURCE135M_native_forward_calls_at_incomplete_prefix_only":physical_teacher_forwards,
        "mean_source135m_T2_4xKL_at_train_continuation":(
            sum(teacher_KL_values)/len(teacher_KL_values) if teacher_KL_values else None),
        "all_teacher_logits_softlabels_only_and_never_EOS":True,
        "actual_registered_new_TRAIN_rows":expected_new,
        "actual_registered_total_TRAIN_rows":expected_total,
        "actual_joint_epochs":rounds,
        "no_original_or_protected_weights_modified":True,
    }


def eval_new_and_old(tok,model,sources,bridge,full,expansion,child,dev,*,variant):
    results=[]
    for e in dev:
        with scoped_reduced_native_head_gain(child.language_head) as proof:
            got=own_token_dev(model,sources,bridge,full,expansion,
                              child,e,condition=variant)
        check(proof[0]==got["original_full_decoder_calls"],
              "original full-native own-token generation only with 0.25 correction")
        results.append(got)
    return results


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,out):
    check(not out.exists(),"never overwrite fitted native research artifacts")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=verify_curriculum()
    check(source["stage"]==SOURCE_STAGE and
          source["fresh_original_author_TRAIN"]==NEW_TRAIN and
          source["fresh_original_author_DEV"]==NEW_DEV,
          "register NEW author source before any pretrained weight load")
    oldmeta=old_i1_provenance()
    oldrehearsal=[v for v in old_i1_rows("TRAIN")
                  if int(v["id"].rsplit("_",1)[-1])==0]
    old_dev=[v for v in old_i1_rows("DEV")
             if int(v["id"].rsplit("_",1)[-1])==6]
    check(len(oldrehearsal)==REHEARSAL_TRAIN and
          len(old_dev)==OLD_DEV and
          not {v["prompt"] for v in oldrehearsal}&
                  {v["prompt"] for v in old_dev},
          "rehearse only 6 previous TRAIN and never prior-scored DEV")
    q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    p=q2i3_dir/Q2I3_REPORT
    q2i3=json.loads(p.read_text(encoding="utf8"))
    check(q2i1["stage"]==Q2I1_STAGE and
          q2i3["stage"]==Q2I3_STAGE and
          q2i3["original_S93_sha256"]==sha(s93) and
          q2i3["original_S142_43M_GT_sha256"]==sha(s142) and
          q2i3["prior_q2i1_real_joint_report_sha256"]==
              sha(q2i1_dir/Q2I1_REPORT) and
          q2i3["source_prior_scored_ru_DEV_sha256"]==oldmeta["DEV_sha256"]
          and q2i3["same_FULL_neural_param_budget_A_B"]==
              TOTAL_EXPECTED_TRAINABLE and
          q2i3["candidate_accepted"] is False,
          "load ONLY completed original physical joint GT+NRA3/NPA1 parent")
    oldq2d,_=load_q2d_trained_child(
        q2d_dir/Q2D_CTX,q2d["new_research_checkpoint_contextual"])
    oldq2f=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,
        q2f["projector_contextual_child_checkpoint"],oldq2d)
    oldq2i1=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],oldq2f)
    fitted_recurrent=check_trained_joint(
        q2i3_dir/RECURRENT_CKPT,
        q2i3["research_only_updated_FULL_neural_child_FP16"][
            "GT_RECURRENT"],oldq2i1)
    fitted_gt=verified_gt12_gates(
        q2i3_dir/RECURRENT_GATES,
        q2i3["research_only_updated_GT12_gate_FP16_sidecar"][
            "GT_RECURRENT"])
    # EXACT same Q2I3 recurrent neural checkpoint and gate vector A/B.
    baseline=copy.deepcopy(fitted_recurrent)
    recurrent=copy.deepcopy(fitted_recurrent)
    local=copy.deepcopy(fitted_recurrent)
    check(all(torch.equal(v,w) and torch.equal(v,z)
              for v,w,z in zip(baseline.state_dict().values(),
                               recurrent.state_dict().values(),
                               local.state_dict().values())),
          "both new joint training conditions start identical current fitted state")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE and tok.eos_token_id is not None,
          "ORIGINAL 49152 native ByteLevel BPE tokenizer")
    native_new_train=[prepare_new(tok,v) for v in new_ru_rows("TRAIN")]
    native_new_dev=[prepare_new(tok,v) for v in new_ru_rows("DEV")]
    native_rehearsal=[verified_native_answer(tok,v) for v in oldrehearsal]
    native_old_dev=[verified_native_answer(tok,v) for v in old_dev]
    train=native_new_train+native_rehearsal
    dev=native_new_dev+native_old_dev
    check(len(train)==TOTAL_TRAIN and len(dev)==NEW_DEV+OLD_DEV
          and {v["prompt_sha256"] for v in train}.isdisjoint(
              v["prompt_sha256"] for v in dev)
          and all(x["split"]=="TRAIN" for x in train)
          and all(x["split"]=="DEV" for x in dev),
          "fresh new native UTF8 split, original TRAIN-only rehearsal, no DEV gradient")
    for e in dev:
        e["eos_native_id"]=tok.eos_token_id
        e["tokenizer"]=tok
    model,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",recurrent)
    model.eval()
    orig_digest=old_original_param_digest(model)
    orig_GT={k:v.detach().cpu().clone() for k,v in full.state_dict().items()}
    original_gate=full.gates.detach().clone()
    original_provenance=native_full_gt_gate_contract(full,bridge)
    baseline_neural={k:v.detach().cpu().clone() for k,v in baseline.state_dict().items()}
    check(all(not par.requires_grad for n,par in model.named_parameters()
              if "nra3_s119m_onpath" not in n),
          "all ORIGINAL S142 43M Core12+GT donors+native head frozen")
    training={};outcomes={};trained_gates={};artifact={}
    out.mkdir(parents=True)
    try:
        for mode,child in (
            ("GT_RECURRENT",recurrent),("GT_NOCARRY_LOCAL",local)
        ):
            set_gt_route(full,mode,fitted_gt)
            training[mode]=train_true_joint(
                model,sources,bridge,full,expansion,child,train,tok,
                condition=mode)
            trained_gates[mode]=full.gates.detach().clone()
            check(old_original_param_digest(model)==orig_digest,
                  "original S142 pretrained 43M/Core/GT/49152 head untouched")
        check(training["GT_RECURRENT"]["trainable_parameter_count"]==
              training["GT_NOCARRY_LOCAL"]["trainable_parameter_count"]==
              TOTAL_EXPECTED_TRAINABLE
              and training["GT_RECURRENT"]["real_all_neural_and_gt_Adam_updates"]==
                  training["GT_NOCARRY_LOCAL"]["real_all_neural_and_gt_Adam_updates"]==
                  ROUNDS*TOTAL_TRAIN
              and training["GT_RECURRENT"]["trained_original_decoder_calls"]==
                  training["GT_NOCARRY_LOCAL"]["trained_original_decoder_calls"],
              "matched starting FP16 state and equal true native original 43M train budget")
        artifacts={
            "recurrent_child":persist_half(recurrent,out/RECURRENT_OUT),
            "local_child":persist_half(local,out/LOCAL_OUT),
            "recurrent_GT12":safe_gt_gate_sidecar(
                out/RECURRENT_GATE_OUT,trained_gates["GT_RECURRENT"],"GT_RECURRENT"),
            "local_GT12":safe_gt_gate_sidecar(
                out/LOCAL_GATE_OUT,trained_gates["GT_NOCARRY_LOCAL"],"GT_NOCARRY_LOCAL"),
        }
        for name,child,gate,gate_type in (
            ("PARENT_RECURRENT",baseline,fitted_gt,"GT_RECURRENT"),
            ("GT_RECURRENT",recurrent,trained_gates["GT_RECURRENT"],"GT_RECURRENT"),
            ("GT_NOCARRY_LOCAL",local,trained_gates["GT_NOCARRY_LOCAL"],"GT_NOCARRY_LOCAL"),
        ):
            set_gt_route(full,gate_type,gate)
            outcomes[name]={
                "NEWDEV":eval_new_and_old(
                    tok,model,sources,bridge,full,expansion,child,
                    native_new_dev,variant=gate_type),
                "OLDDEV":eval_new_and_old(
                    tok,model,sources,bridge,full,expansion,child,
                    native_old_dev,variant=gate_type),
            }
    finally:
        with torch.no_grad():
            full.gates.copy_(orig_GT["gates"])
            full.mix_logits.copy_(orig_GT["mix_logits"])
        for par in full.parameters():par.requires_grad_(False)
    check(all(torch.equal(v,full.state_dict()[k].detach().cpu())
              for k,v in orig_GT.items()) and
          old_original_param_digest(model)==orig_digest
          and all(torch.equal(v,baseline.state_dict()[k].detach().cpu())
                  for k,v in baseline_neural.items()),
          "S142 43M + GreenTunnel exactly restored, previous q2i3 true parent read-only")
    def score(report):
        check(len(report)==6,"six native own-token source cases each split")
        return {
            "strict_full_answer_with_NATIVE_EOS":sum(
                z["strict_finished_full_answer_exact"] for z in report),
            "exact_text_even_without_EOS":sum(
                z["text_exact_even_if_not_finished"] for z in report),
            "native_EOS_finished":sum(z["native_EOS"] for z in report),
            "real_original_decoder_forward_calls":sum(
                z["original_full_decoder_calls"] for z in report),
        }
    scored={k:{split:score(records) for split,records in d.items()}
            for k,d in outcomes.items()}
    train_calls=sum(t["trained_original_decoder_calls"]
                    for t in training.values())
    dev_calls=sum(scored[v][split]["real_original_decoder_forward_calls"]
                  for v in scored for split in ("NEWDEV","OLDDEV"))
    payload={
        "stage":STAGE,
        "original_S93_SHA256":sha(s93),
        "original_S142_Core_GT12_43M_SHA256":sha(s142),
        "parent_q2i3_full_988_physical_run_SHA256":sha(p),
        "fresh_original_author_bytelevel_SOURCE":source,
        "rehearsal_old_q2i1_TRAIN_count":REHEARSAL_TRAIN,
        "no_old_q2i1_DEV_or_old_q2g_labels_in_optimizer":True,
        "same_initial_existing_physical_q2i3_recurrent_model_AND_GT12_gates":True,
        "true_native_russian_bytes_incomplete_prefix_CE_weight":CONTINUATION_WEIGHT,
        "three_joint_epochs_with_rehearsal":ROUNDS,
        "matched_368051_actual_optimizer_params":TOTAL_EXPECTED_TRAINABLE,
        "actual_original_43M_native_decoder_calls_TRAIN":train_calls,
        "actual_original_43M_native_decoder_calls_own_BPE_DEV":dev_calls,
        "actual_original_43M_native_decoder_calls_TOTAL":train_calls+dev_calls,
        "training_real_fullBPE_EOS":training,
        "new_training_FP16_sidecar_provenance":artifacts,
        "new_original_author_DEV_and_spent_OLDDEV_own_native_scoring":scored,
        "every_own_native_DEV_full_answer_trace":outcomes,
        "original_S142_43M_GT_all_pretrained_weights_UNCHANGED":True,
        "prior_q2i3_fitted_neural_parent_FP16_UNCHANGED":True,
        "old_S115_53751_FP16_neural_weights_UNCHANGED":True,
        "no_independent_BLIND_eval_small_authored_DEV_only":True,
        "protected_FULL_616_J7_EOS_not_retested":True,
        "physical_allin_packed_bpp_and_matched_KV_CPU_not_validated":True,
        "NO_S118_GOLD_spent_EVAL128_J7_answers_or_Jessica_chat":True,
        "no_main_LIVE_Render_Supabase_APK_secrets":True,
        "output_gate":0,"candidate_accepted":False,
        "decision":"FRESH_UTF8_CONTINUATION_JOINT_RESEARCH_NEVER_PROMOTE",
    }
    (out/REPORT).write_text(json.dumps(
        payload,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119Q2I6_REAL43M_UTF8_MULTIBPE_JOINT_TRAIN "+
          json.dumps({
            "physical_original_decoder_calls":train_calls+dev_calls,
            "new_source_TRAIN_DEV":[NEW_TRAIN,NEW_DEV],
            "rehearsal_old_TRAIN_only":REHEARSAL_TRAIN,
            "real_full_native_EOS_Adam_updates_per_arm":ROUNDS*TOTAL_TRAIN,
            "continuation_weight":CONTINUATION_WEIGHT,
            "training_first_and_last_CE":{
                k:[v["rounds"][0]["train_mean_native_CE_ALL_target_BPE_EOS"],
                   v["rounds"][-1]["train_mean_native_CE_ALL_target_BPE_EOS"]]
                for k,v in training.items()},
            "native_DEV_self_complete_correct_answers":{
                name:{d:s["strict_full_answer_with_NATIVE_EOS"]
                      for d,s in v.items()}
                for name,v in scored.items()},
            "original_pretrained_Core_GT12_FROZEN":True,
            "accepted":False,
          },sort_keys=True,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir","q2i3_dir","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,a.q2i3_dir,a.out)


if __name__=="__main__":
    main()
