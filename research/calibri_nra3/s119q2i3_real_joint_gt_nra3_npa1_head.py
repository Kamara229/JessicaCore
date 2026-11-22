"""S119q2i3: TRUE joint learning of Green Tunnel + NRA3/NPA1/Core and native head.

Physical original S142 43M Core12+GT rank4 original 12layer recurrent state.
Start from EXACT prior physically trained S119q2i1 contextual 368,039
joint neural parameters; add only 12 new trainable GT12 gates. Unlike the
S119q2i2 gate-only connectivity test, optimizer here updates joint
rank4 core, structured NRA3, NPA1 planning, early H0..5 projection,
original-trained neural native full 49152 head, AND Green Tunnel 12 gates,
all via the SAME actual 12layer model one-pass original decoder.

Two strictly matched children, same q2i1 FP16 neural initialization,
same training examples, EOS target, optimizer, learning-rate scheme,
trainable parameter budget, native full causal forward:
 A: recurrent GT12 with original frozen mix_logits==0 (carry sigmoid .5).
 B: local no-carry GT12 with same rank4 encoder/decoder/12 gates, frozen
    mix_logits==-20 (carry approximately zero).
Train 3 rounds over exactly 12 already-authored TRAIN questions (2/domain)
from S119q2i1. 12 OTHER old already-scored DEV questions: independent
of gradients but NOT independently blind or new benchmark. The q2i1
originally-trained language-head correction dominated invalid byte-token
generation; this controlled research study applies the SAME FIXED x0.25
head-delta gain in baseline and BOTH candidate variants to limit output
collapse without adding any learned param or hardcoding Russian words.
No optimization of that gain on DEV, no language-token filtering.

Original pretrained 43M, original FP16 53,751 NRA3 and S93/GT donor
weights frozen; S142 GT12 original ZERO gate and mix restored at the end,
new trained states saved only as isolated research FP16 checkpoints.
No GOLD/EVAL128/J7 or q2g/q2h spent labels used for gradients. No
main/LIVE/Render/Supabase/Jessica chat/APK. output_gate=0; no promotion.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path

import torch
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
    CONTEXT_CKPT as Q2D_CONTEXT, LOCAL_CKPT as Q2D_LOCAL,
    REPORT as Q2D_REPORT,VOCAB_SIZE,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CONTEXT,REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as ru_rows,provenance as ru_provenance,KINDS,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    STAGE as Q2I1_STAGE,CONTEXT_CKPT as Q2I1_CTX,
    REPORT as Q2I1_REPORT,verified_native_answer,real_joint_native,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    native_full_gt_gate_contract,load_q2i1_physical_context,
    set_gt_route,own_token_dev,safe_gt_gate_sidecar,
    GT_DEPTH,GT_RANK,GT_WIDTH,CONDITIONS,
)

STAGE="CALIBRI_S119Q2I3_ORIGINAL43M_GT12_NRA3_NPA1_HEAD_ALL_JOINT_3ROUNDS"
REPORT="S119Q2I3_GT12_NRA3_NPA1_NATIVEHEAD_3ROUNDS_OLDDEV_REPORT.json"
RECURRENT_CKPT="S119Q2I3_RECURRENT_GT12_CORE_NRA3_NPA1_NATIVE_HEAD_FP16_G0.safetensors"
LOCAL_CKPT="S119Q2I3_LOCAL_GT12_CORE_NRA3_NPA1_NATIVE_HEAD_FP16_G0.safetensors"
RECURRENT_GATES="S119Q2I3_RECURRENT_GT12_12GATES_FP16_G0.safetensors"
LOCAL_GATES="S119Q2I3_LOCAL_GT12_12GATES_FP16_G0.safetensors"
SEED=119142
TRAIN_ROWS=12
DEV_ROWS=12
ROUNDS=3
HEAD_GAIN=0.25
GATE_LR=0.004
CORE_REGION_LR=0.00012
HEAD_PROJECTION_LR=0.00035
HEAD_BIAS_LR=0.00002
GRAD_CLIP=0.50
GT_CONDITIONS=("GT_RECURRENT","GT_NOCARRY_LOCAL")
LEARNED_NEW_REGIONS_EXPECTED=368039
TOTAL_EXPECTED_TRAINABLE=LEARNED_NEW_REGIONS_EXPECTED+GT_DEPTH


@contextmanager
def scoped_reduced_native_head_gain(language_head):
    """Same x0.25 learned original-head correction for ALL three arms."""
    check(language_head.output_gate==0
          and not language_head.candidate_accepted
          and HEAD_GAIN==0.25,
          "fixed research head correction gain; no hand-coded BPE masking")
    prior=set(language_head._forward_hooks)
    counter=[0]
    def reduce(_module,_args,delta):
        check(delta.ndim==3 and delta.shape[-1]==VOCAB_SIZE
              and bool(torch.isfinite(delta).all()),
              "full original native LM readout correction only")
        counter[0]+=1
        return delta*HEAD_GAIN
    handle=language_head.register_forward_hook(reduce)
    try:
        yield counter
    finally:
        handle.remove()
        check(set(language_head._forward_hooks)==prior,
              "scoped native language scaling MUST remove hook")


def make_joint_trainable(child):
    """Unfreeze ONLY previously-new param set, not original S115/43M."""
    # S119q2i2 load_q2i1_physical_context deliberately freezes ALL child
    # parameters to test GT-only; this stage re-enables strictly the
    # q2i1 new joint neural parameters and head, retaining old old 53751.
    for p in child.parameters():
        p.requires_grad_(False)
    inner=child.child.child
    old_new=list(inner.trainables())
    projection=[
        child.child.down.weight,
        child.child.up.weight,
        child.child.mix_logits,
        child.child.slot_gain,
    ]
    native_head=list(child.language_head.parameters())
    selected=old_new+projection+native_head
    check(len(selected)==len({id(p) for p in selected})
          and sum(p.numel() for p in selected)==LEARNED_NEW_REGIONS_EXPECTED
          and not any(id(p) in {id(q) for q in selected}
                      for p in inner.shadow.unified.nra3_neural_weights.parameters()),
          "exact 368039 q2i1 new neural Core/NRA3/NPA1/projector/readout only")
    for p in selected:
        p.requires_grad_(True)
    check(len(child.trainables())==len(selected)
          and all(p.requires_grad for p in child.trainables())
          and all(not p.requires_grad for p in
                  inner.shadow.unified.nra3_neural_weights.parameters()),
          "old S115 53,751 pretrained FP16 remain fully immutable")
    return selected


def train_all_joint(model,sources,bridge,full,expansion,child,training,
                    *,condition):
    check(condition in GT_CONDITIONS and len(training)==TRAIN_ROWS
          and Counter(x["kind"] for x in training)=={k:2 for k in KINDS}
          and all(x["split"]=="TRAIN" for x in training),
          "all 12 authored TRAIN; no oldDEV or RESERVED EVAL gradients")
    region_params=make_joint_trainable(child)
    for name,p in full.named_parameters():
        p.requires_grad_(name=="gates")
    head_params=list(child.language_head.parameters())
    nonhead=[p for p in region_params if id(p) not in
             {id(q) for q in head_params}]
    check(sum(p.numel() for p in region_params)==LEARNED_NEW_REGIONS_EXPECTED
          and full.gates.requires_grad
          and sum(p.numel() for p in region_params)+full.gates.numel()
              ==TOTAL_EXPECTED_TRAINABLE,
          "same 368051 FULL effective trainables GT + all new native areas")
    optimizer=torch.optim.Adam([
        {"params":[full.gates],"lr":GATE_LR},
        {"params":nonhead,"lr":CORE_REGION_LR},
        {"params":[child.language_head.down.weight,
                   child.language_head.up.weight],"lr":HEAD_PROJECTION_LR},
        {"params":[child.language_head.output_bias],"lr":HEAD_BIAS_LR},
    ])
    old_weights={
        k:v.detach().cpu().half().clone()
        for k,v in child.child.child.shadow.unified.nra3_neural_weights.state_dict().items()
    }
    before_new={
        "GT":full.gates.detach().clone(),
        "NRA3":child.child.child.shadow.unified.structured_interface.fuse.weight.detach().clone(),
        "NPA1":child.child.child.shadow.unified.planning_action_area.state_update.weight_ih.detach().clone(),
        "CORE":child.child.child.shadow.unified.core_growth.up.detach().clone(),
        "HEAD":child.language_head.up.weight.detach().clone(),
        "PROJECTOR":child.child.up.weight.detach().clone(),
    }
    rounds=[];forwards=0;steps=0;first_grad=None
    for epoch in range(ROUNDS):
        losses=[];eos_ce=[];num_native=0
        for e in training:
            optimizer.zero_grad(set_to_none=True)
            gold=e["answer_native_with_eos"]
            with scoped_reduced_native_head_gain(child.language_head) as scale_witness:
                for t,target in enumerate(gold):
                    prefix=e["prefix"]+gold[:t]
                    check(prefix[:len(e["prefix"])]==e["prefix"] and
                          prefix[len(e["prefix"]):]==gold[:t] and
                          int(target)==int(gold[t]) and
                          len(prefix)<160,
                          "native causal teacher forcing only earlier BPE, never target")
                    logits=real_joint_native(
                        model,sources,bridge,full,expansion,child,
                        prefix,e["prompt_features"],contextual=True)
                    check(logits.shape==(1,VOCAB_SIZE)
                          and full.last_forward_coverage==tuple(range(GT_DEPTH))
                          and not full._active and full._carry is None,
                          "real original full 12layer GT12 native head same pass")
                    objective=F.cross_entropy(
                        logits,torch.tensor([target],dtype=torch.long))
                    check(bool(torch.isfinite(objective)),
                          "real original native full answer+EOS CE")
                    (objective/len(gold)).backward()
                    losses.append(float(objective.detach()))
                    if t==len(gold)-1:
                        eos_ce.append(float(objective.detach()))
                    num_native+=1;forwards+=1
                    del objective,logits
            check(scale_witness[0]==len(gold),
                  "fixed x0.25 full-native correction used for EVERY target token")
            if first_grad is None and steps>=1:
                grad={
                    "GT12_full_layer_recurrent_gates":full.gates.grad,
                    "NRA3_structured_interface":
                        child.child.child.shadow.unified.structured_interface.fuse.weight.grad,
                    "NPA1_planning":
                        child.child.child.shadow.unified.planning_action_area.state_update.weight_ih.grad,
                    "new_core_rank4":
                        child.child.child.shadow.unified.core_growth.up.grad,
                    "learned_causal_projector":
                        child.child.up.weight.grad,
                    "native_49152_LM_alignment":
                        child.language_head.up.weight.grad,
                }
                first_grad={}
                for name,val in grad.items():
                    check(val is not None
                          and bool(torch.isfinite(val).all())
                          and float(val.detach().abs().sum())>1e-18,
                          "ONE ORIGINAL 43M FULL NATIVE loss gradients reach "+name)
                    first_grad[name]=float(torch.linalg.vector_norm(val))
            torch.nn.utils.clip_grad_norm_(region_params+[full.gates],GRAD_CLIP)
            optimizer.step();steps+=1
            check(all(bool(torch.isfinite(p).all())
                      for p in region_params+[full.gates])
                  and bool((full.gates.detach().abs()<1.).all()),
                  "joint original-43M GT/NRA3/NPA1/LM-head finite/bounded")
        rounds.append({
            "round":epoch+1,
            "original_physical_full_native_decoder_forwards":num_native,
            "real_adam_updates":TRAIN_ROWS,
            "TRAIN_mean_full_Russian_BPE_plus_EOS_CE":
                sum(losses)/len(losses),
            "TRAIN_mean_real_native_EOS_CE":sum(eos_ce)/len(eos_ce),
            "fixed_head_correction_gain":HEAD_GAIN,
            "GT12_gate_Linf":float(full.gates.detach().abs().max()),
            "all_NRA3_NPA1_Core_GT_and_output_head_joint_trainable":True,
        })
    check(first_grad is not None and steps==TRAIN_ROWS*ROUNDS
          and rounds[0]["original_physical_full_native_decoder_forwards"]>TRAIN_ROWS
          and all(torch.equal(v.detach().cpu().half(),old_weights[k])
                  for k,v in
                  child.child.child.shadow.unified.nra3_neural_weights.state_dict().items()),
          "real three full-bpe joint rounds; original old S115 53,751 unmodified")
    modified={
        "GT":full.gates,
        "NRA3":child.child.child.shadow.unified.structured_interface.fuse.weight,
        "NPA1":child.child.child.shadow.unified.planning_action_area.state_update.weight_ih,
        "CORE":child.child.child.shadow.unified.core_growth.up,
        "HEAD":child.language_head.up.weight,
        "PROJECTOR":child.child.up.weight,
    }
    changes={}
    for key,post in modified.items():
        delta=float(torch.linalg.vector_norm(post.detach()-before_new[key]))
        changes[key]=delta
        check(delta>1e-9,"actual joint learned param changed "+key)
    return {
        "rounds":rounds,
        "original_full_native_decoder_forwards":forwards,
        "real_Adam_updates":steps,
        "trainable_joint_parameters_including_GT12":TOTAL_EXPECTED_TRAINABLE,
        "first_physical_gradient_norms":first_grad,
        "actual_changed_module_delta_norms":changes,
        "last_GT12_gates":full.gates.detach().cpu().tolist(),
        "original_old_53751_fp16_frozen":True,
    }


def dev_own_stream(model,sources,bridge,full,expansion,child,dev,condition):
    check(len(dev)==DEV_ROWS and
          all(x["split"]=="DEV" for x in dev),
          "previously SCORED q2i1 DEV only for read-only diagnosis")
    reports=[]
    for e in dev:
        with scoped_reduced_native_head_gain(child.language_head) as measured:
            got=own_token_dev(
                model,sources,bridge,full,expansion,child,e,condition=condition)
        check(measured[0]==got["original_full_decoder_calls"],
              "same fixed native-head correction gain every OWN BPE")
        reports.append(got)
    return reports


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,out):
    check(not out.exists(),"new experiment must not overwrite old research")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=ru_provenance()
    train_rows=[v for v in ru_rows("TRAIN")
                if int(v["id"].rsplit("_",1)[-1])<2]
    dev_rows=ru_rows("DEV")
    check(len(train_rows)==TRAIN_ROWS and len(dev_rows)==DEV_ROWS
          and source["TRAIN_rows"]==36 and source["DEV_rows"]==DEV_ROWS
          and source["old_sources_and_spent_q2g_disjoint"],
          "same preregistered 12 TRAIN and 12 old DEV source only")
    q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CONTEXT,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    check(q2i1["stage"]==Q2I1_STAGE
          and q2i1["original_S93_SHA256"]==sha(s93)
          and q2i1["original_S142_43M_GT_SHA256"]==sha(s142)
          and q2i1["prior_physically_trained_q2f_report_sha256"]
              ==sha(q2f_dir/Q2F_REPORT)
          and q2i1["original_author_new_Russian_source_provenance"]==source
          and q2i1["candidate_accepted"] is False
          and q2f["parent_q2d_training_report_SHA256"]
              ==sha(q2d_dir/Q2D_REPORT),
          "true prior physical trained joint child and original S142 exact SHA")
    q2d_parent,_=load_q2d_trained_child(
        q2d_dir/Q2D_CONTEXT,
        q2d["new_research_checkpoint_contextual"])
    q2f_parent=load_physical_q2f_child(
        q2f_dir/Q2F_CONTEXT,
        q2f["projector_contextual_child_checkpoint"],q2d_parent)
    parent=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],q2f_parent)
    # All three comparable baselines MUST share identical learned start,
    # including native output bias and all old neural protected parameters.
    recurrent=copy.deepcopy(parent)
    local=copy.deepcopy(parent)
    baseline=copy.deepcopy(parent)
    check(all(torch.equal(a.detach(),b.detach()) and
              torch.equal(a.detach(),c.detach())
              for a,b,c in zip(parent.state_dict().values(),
                  recurrent.state_dict().values(),local.state_dict().values()))
          and not parent.candidate_accepted,
          "identical physical q2i1 child and LM-head initialization")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE and tok.eos_token_id is not None,
          "true original native 49152 tokenizer EOS")
    train=[verified_native_answer(tok,x) for x in train_rows]
    dev=[verified_native_answer(tok,x) for x in dev_rows]
    check(len(train)==TRAIN_ROWS and len(dev)==DEV_ROWS and
          {x["prompt_sha256"] for x in train}.isdisjoint(
              x["prompt_sha256"] for x in dev),
          "strict TRAIN DEV disjoint source; EVAL not opened")
    for x in dev:
        x["eos_native_id"]=int(tok.eos_token_id)
        x["tokenizer"]=tok
    model,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",recurrent)
    model.eval()
    original_digest=old_original_param_digest(model)
    full_state={k:v.detach().cpu().clone()
                for k,v in full.state_dict().items()}
    original_zero=full.gates.detach().clone()
    original_mix=full.mix_logits.detach().clone()
    gt_provenance=native_full_gt_gate_contract(full,bridge)
    check(all(not p.requires_grad for name,p in model.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "ALL 43M original Core/GreenTunnel donors/49k LM head frozen")
    results={};trained_gates={};learned_states={}
    total_train=0;total_dev=0
    out.mkdir(parents=True)
    try:
        for condition,child in (("GT_RECURRENT",recurrent),
                                ("GT_NOCARRY_LOCAL",local)):
            set_gt_route(full,condition,original_zero)
            before_old={k:v.detach().cpu().clone() for k,v in
                        child.child.child.shadow.unified.nra3_neural_weights.state_dict().items()}
            results[condition]=train_all_joint(
                model,sources,bridge,full,expansion,child,train,
                condition=condition)
            trained_gates[condition]=full.gates.detach().clone()
            total_train+=results[condition]["original_full_native_decoder_forwards"]
            check(all(torch.equal(v,child.child.child.shadow.unified.nra3_neural_weights.state_dict()[k].detach().cpu())
                      for k,v in before_old.items()),
                  "original S115 53751 neural weights unchanged after joint GT optimization")
            learned_states[condition]={
                k:v.detach().cpu().clone()
                for k,v in child.state_dict().items()}
            check(old_original_param_digest(model)==original_digest,
                  "entire old original Core12 43M/GT3/9/nativeLM immutable")
        check(results["GT_RECURRENT"]["trainable_joint_parameters_including_GT12"]==
              results["GT_NOCARRY_LOCAL"]["trainable_joint_parameters_including_GT12"]
              ==TOTAL_EXPECTED_TRAINABLE
              and results["GT_RECURRENT"]["real_Adam_updates"]==
                  results["GT_NOCARRY_LOCAL"]["real_Adam_updates"]==
                  TRAIN_ROWS*ROUNDS
              and results["GT_RECURRENT"]["original_full_native_decoder_forwards"]==
                  results["GT_NOCARRY_LOCAL"]["original_full_native_decoder_forwards"],
              "exact matched true recurrent and no-carry FULL neural training budget")
        # Save learned GT gates separately; trained neural changes are NEW
        # research child FP16, never overwrite q2i1 original trained artifact.
        physical_gt={
            condition:safe_gt_gate_sidecar(
                out/(RECURRENT_GATES if condition=="GT_RECURRENT"
                     else LOCAL_GATES),trained_gates[condition],condition)
            for condition in GT_CONDITIONS
        }
        checkpoints={
            "GT_RECURRENT":persist_half(recurrent,out/RECURRENT_CKPT),
            "GT_NOCARRY_LOCAL":persist_half(local,out/LOCAL_CKPT),
        }
        dev_traces={}
        for condition,child in (
            ("GT_ZERO_BASELINE",baseline),
            ("GT_RECURRENT",recurrent),
            ("GT_NOCARRY_LOCAL",local),
        ):
            g=(original_zero if condition=="GT_ZERO_BASELINE"
               else trained_gates[condition])
            set_gt_route(full,condition,g)
            dev_traces[condition]=dev_own_stream(
                model,sources,bridge,full,expansion,child,dev,condition)
            total_dev+=sum(x["original_full_decoder_calls"] for x in dev_traces[condition])
    finally:
        with torch.no_grad():
            full.gates.copy_(original_zero)
            full.mix_logits.copy_(original_mix)
        for p in full.parameters(): p.requires_grad_(False)
    check(all(torch.equal(v,full.state_dict()[k].detach().cpu())
              for k,v in full_state.items()) and
          old_original_param_digest(model)==original_digest,
          "original full GT12/GreenTunnel/pretrained 43M restored exactly")
    scored={}
    for condition,trace in dev_traces.items():
        check(len(trace)==DEV_ROWS
              and Counter(x["domain"] for x in trace)=={k:2 for k in KINDS},
              "the EXACT 12 disjoint OLD q2i1 DEV native free generation")
        scored[condition]={
            "strict_real_EOS_full_answers":
                sum(x["strict_finished_full_answer_exact"] for x in trace),
            "visible_exact_text_without_EOS":
                sum(x["text_exact_even_if_not_finished"] for x in trace),
            "native_EOS_finished_count":sum(x["native_EOS"] for x in trace),
            "actual_full_original_native_decoder_forwards":
                sum(x["original_full_decoder_calls"] for x in trace),
        }
    summary={
        "stage":STAGE,
        "source_original_ru_TRAIN_sha256":source["TRAIN_sha256"],
        "source_prior_scored_ru_DEV_sha256":source["DEV_sha256"],
        "original_S93_sha256":sha(s93),
        "original_S142_43M_GT_sha256":sha(s142),
        "prior_q2i1_real_joint_report_sha256":sha(q2i1_dir/Q2I1_REPORT),
        "GT12_original_all_12_gates_zero_provenance":gt_provenance,
        "original_full_native_vocab":VOCAB_SIZE,
        "same_prior_q2i1_contextual_FP16_start_FOR_BOTH":True,
        "identical_HEAD_GAIN_baseline_recurrent_nocarry":HEAD_GAIN,
        "same_FULL_neural_param_budget_A_B":TOTAL_EXPECTED_TRAINABLE,
        "3_real_joint_GT_NRA3_NPA1_head_epochs":ROUNDS,
        "original_real_43M_full_decoder_TRAIN_forwards":total_train,
        "original_real_43M_full_decoder_OLDDEV_own_BPE_forwards":total_dev,
        "total_real_original_decoder_forwards":total_train+total_dev,
        "training":results,
        "research_only_updated_GT12_gate_FP16_sidecar":physical_gt,
        "research_only_updated_FULL_neural_child_FP16":checkpoints,
        "oldDEV_own_native_EOS_scores":scored,
        "oldDEV_own_native_complete_text":dev_traces,
        "original_S142_GT12_gates_and_mix_RESTORED":True,
        "all_original_Core12_43M_GT3_GT9_and_native_head_frozen":True,
        "old_53751_pretrained_FP16_neural_weights_UNCHANGED":True,
        "no_new_independent_BLIND_eval_or_616_J7_EOS":True,
        "no_gold_S118_B_spent_EVAL128_or_q2g_labels":True,
        "no_packed_all_in_bpp_or_matched_KV_cache_CPU":True,
        "no_Jessica_chat_LIVE_Render_Supabase_main_APK":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"PHYSICAL_JOINT_GT12_NRA3_NPA1_LANG_HEAD_RESEARCH_NOT_ACCEPTED",
    }
    (out/REPORT).write_text(
        json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I3_TRUE_ORIGINAL43M_GT_NRA3_NPA1_JOINT_3ROUNDS "+
          json.dumps({
              "true_original_decoder_forwards":total_train+total_dev,
              "learnable_params_each":TOTAL_EXPECTED_TRAINABLE,
              "trainable_GT_gates":GT_DEPTH,
              "training_recurrent_first_CE":results["GT_RECURRENT"]["rounds"][0][
                  "TRAIN_mean_full_Russian_BPE_plus_EOS_CE"],
              "training_recurrent_final_CE":results["GT_RECURRENT"]["rounds"][-1][
                  "TRAIN_mean_full_Russian_BPE_plus_EOS_CE"],
              "training_local_final_CE":results["GT_NOCARRY_LOCAL"]["rounds"][-1][
                  "TRAIN_mean_full_Russian_BPE_plus_EOS_CE"],
              "oldDEV_strict_native_answers":{
                  k:v["strict_real_EOS_full_answers"] for k,v in scored.items()},
              "oldDEV_visible_text_correct":{
                  k:v["visible_exact_text_without_EOS"] for k,v in scored.items()},
              "old_original_GT12_restored":True,
              "candidate_accepted":False,
          },sort_keys=True,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,a.out)


if __name__=="__main__":
    main()
