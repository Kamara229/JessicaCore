"""S119q2i1: true joint original43M native FULL-ANSWER language alignment TRAIN.

Original 12-layer frozen S142 Core12+GreenTunnel and full 49,152 LM head.
Train a small learned low-rank CONTEXT-CONDITIONED native output adapter
and the previously trained q2f Core rank4/NRA3/NPA1/rank8 neural child
together, using 36 NEW original-author Russian TRAIN answers. Every
correct native BPE and an actual EOS target is supervised with causal
teacher forcing: the target and future BPE are NEVER input. A subsequent
separate DEV probe uses ONLY OWN predicted native tokens with greedy
autoregression and native EOS. 12 DEV examples (2 from each six domains),
no DEV gradient, SEALED_EVAL (12) is never accessed by this runner.

Two candidates initialized from EXACT SAME trained contextual q2f FP16
child and identical new rank4 native head weights:
 CON: genuine h0..h5 causal NRA3+NPA1 context=true
 LOC: same param count, token-local reference, context=false.
Original native LM head remains frozen. We install a TEMPORARY forward
hook inside original native lm_head only for original physical prefill;
no separate model calls, no hard-coded Russian output token biases or
posthoc language masks. The language correction is fully learned via
native full-vocab gradient. Genuine FULL 49,152-way normal loss.
Outputs GATE0 research unaccepted; no promotion or production mutation.

This tiny authored synthetic dataset is not a licensed NATURAL corpus
and DEV is not independently blinded. Native-contextual vs local
quality is NOT established until genuinely separate comprehensive tests
and protected 616 J7/EOS, all-in bpp<=0.7, matched KV CPU>=90%.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter,defaultdict
from contextlib import contextmanager
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
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    first_original_pass,prompt_features,
)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    strict_chat_source_ids,
)
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    onepass_full_native_logits,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    persist_half,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CONTEXT, LOCAL_CKPT as Q2D_LOCAL,
    REPORT as Q2D_REPORT, VOCAB_SIZE,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CONTEXT, LOCAL_CKPT as Q2F_LOCAL,
    REPORT as Q2F_REPORT, STAGE as Q2F_STAGE,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as new_ru_rows, provenance as ru_provenance,
    hash_rows as ru_source_hash,KINDS,
)

STAGE="CALIBRI_S119Q2I1_REAL43M_JOINT_RUSSIAN_FULL_NATIVE_BPE_EOS_TRAIN"
REPORT="S119Q2I1_REAL43M_RU_NATIVE_EOS_TRAIN_FREE_DEV_REPORT.json"
CONTEXT_CKPT="S119Q2I1_CONTEXTUAL_CORE_NRA3_NPA1_HEAD_FP16_G0.safetensors"
LOCAL_CKPT="S119Q2I1_TOKEN_LOCAL_CORE_NRA3_NPA1_HEAD_FP16_G0.safetensors"
SEED=119140
N_TRAIN=36
N_DEV=12
EPOCHS=2
LOWRANK=4
HEAD_LEARNING_RATE=0.018
CHILD_LEARNING_RATE=0.0003
MAX_ANSWER_NATIVE=48
DEV_MAX_GEN_NATIVE=16
HEAD_PARAMETER_COUNT=320*LOWRANK+LOWRANK*VOCAB_SIZE+VOCAB_SIZE


class NativeHeadAlignment(nn.Module):
    """Contextual neural native readout: true hidden last-state + low rank."""
    def __init__(self):
        super().__init__()
        self.down=nn.Linear(320,LOWRANK,bias=False)
        self.up=nn.Linear(LOWRANK,VOCAB_SIZE,bias=False)
        self.output_bias=nn.Parameter(torch.zeros(VOCAB_SIZE))
        nn.init.normal_(self.down.weight,mean=0.,std=0.02)
        nn.init.zeros_(self.up.weight)
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False

    def forward(self,last_hidden):
        check(self.output_gate==0 and not self.candidate_accepted
              and not self.activation_allowed and
              last_hidden.ndim==3 and last_hidden.shape[-1]==320,
              "only original frozen Core12 final hidden h320 native head")
        correction=self.up(F.gelu(self.down(
            F.layer_norm(last_hidden,(320,)))))
        return correction+self.output_bias.view(1,1,-1)


class JointRuNative(nn.Module):
    """Preserve trained Core/NRA3/NPA1 and separately learn full native head."""
    def __init__(self,physical_child):
        super().__init__()
        self.child=physical_child
        self.language_head=NativeHeadAlignment()
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.projection_enabled=True
    def forward(self,physical12,*,contextual):
        check(self.output_gate==0 and not self.candidate_accepted
              and not self.activation_allowed
              and self.child.output_gate==0
              and not self.child.candidate_accepted,
              "both research original onpath and language head unaccepted")
        return self.child(physical12,contextual=contextual)
    def trainables(self):
        params=list(self.child.trainables())+list(self.language_head.parameters())
        check(len(params)==len({id(p) for p in params}) and
              all(p.requires_grad for p in params),
              "only NEW child/Core/NRA3/NPA1 + compact native head trainable")
        return params


@contextmanager
def temporary_learned_native_head(model,head):
    """Single original physical decoder native-lm-head hook, exact rollback."""
    check(hasattr(model,"lm_head") and isinstance(model.lm_head,nn.Module)
          and isinstance(head,NativeHeadAlignment)
          and head.output_gate==0 and not head.candidate_accepted,
          "real frozen ORIGINAL native lm_head required")
    count={"original_head_calls":0}
    def head_hook(_module,args,original_logits):
        check(count["original_head_calls"]==0 and
              len(args)>=1 and isinstance(args[0],torch.Tensor)
              and isinstance(original_logits,torch.Tensor)
              and args[0].ndim==3 and args[0].shape[-1]==320
              and original_logits.shape[:2]==args[0].shape[:2]
              and original_logits.shape[-1]==VOCAB_SIZE,
              "ONE real original 49152 native head hidden BPE logits")
        count["original_head_calls"]+=1
        delta=head(args[0])
        check(delta.shape==original_logits.shape and
              bool(torch.isfinite(delta).all()),
              "lowrank original frozen native head correction finite")
        return original_logits.float()+delta.float()
    handle=model.lm_head.register_forward_hook(head_hook)
    try:
        yield count
        check(count["original_head_calls"]==1,
              "ONE physical ORIGINAL LM head per original decoder forward")
    finally:
        handle.remove()


def verified_native_answer(tok,row):
    """Exact original native chat input; full answer plus EOS causal targets."""
    check(row["split"] in ("TRAIN","DEV")
          and row["origin"]==
              "original_newly_authored_examples_no_third_party_corpus",
          "original-author TRAIN/DEV only, NEVER SEALED_EVAL")
    prompt=strict_chat_source_ids(tok,row["prompt"])
    chat=tok.apply_chat_template(
        [{"role":"user","content":row["prompt"]}],
        tokenize=False,add_generation_prompt=True)
    expected_full=tok.encode(chat+row["completion"],add_special_tokens=False)
    check(expected_full[:len(prompt)]==prompt and
          len(expected_full)>len(prompt),
          "original native BPE chat header must be exact causal prefix")
    answer_native=list(expected_full[len(prompt):])
    check(1<=len(answer_native)<=MAX_ANSWER_NATIVE-1 and
          all(type(x) is int and 0<=x<VOCAB_SIZE for x in answer_native) and
          tok.eos_token_id not in answer_native and
          len(prompt)+len(answer_native)+1<160,
          "full original author native text targets, no truncated EOS")
    check(tok.decode(answer_native,skip_special_tokens=True).strip()==
          row["completion"].strip(),
          "full authored answer native BPE decodes exactly")
    return {
        "id":row["id"],"kind":row["kind"],
        "split":row["split"],
        "prompt_sha256":hashlib.sha256(
            row["prompt"].encode("utf8")).hexdigest(),
        "prefix":prompt,
        "answer_native_with_eos":answer_native+[int(tok.eos_token_id)],
        "answer_native_without_eos":answer_native,
        "prompt_features":prompt_features(tok,row["prompt"]),
        "gold_answer_for_scoring_ONLY":row["completion"],
    }


def real_joint_native(model,sources,bridge,full,expansion,adapter,
                      prefix,features,*,contextual):
    with temporary_learned_native_head(
        model,adapter.language_head) as audit:
        logits,witness=onepass_full_native_logits(
            model,sources,bridge,full,expansion,adapter,
            prefix,features,contextual=contextual)
    check(audit["original_head_calls"]==1 and
          witness["candidate_forward_calls"]==1 and
          logits.shape==(1,VOCAB_SIZE) and logits.requires_grad,
          "one full original Core12 + native Head training forward")
    return logits


def train_joint(model,sources,bridge,full,expansion,adapter,train,*,contextual):
    check(len(train)==N_TRAIN and
          all(x["split"]=="TRAIN" for x in train),
          "new author handwritten 36 TRAIN examples only, no DEV gradient")
    torch.manual_seed(SEED)
    language_params=list(adapter.language_head.parameters())
    child_params=list(adapter.child.trainables())
    check(sum(p.numel() for p in language_params)==HEAD_PARAMETER_COUNT
          and sum(p.numel() for p in child_params)>100000,
          "full 49152 head learned lowrank4+bias and true NRA3/NPA1 child")
    optimizer=torch.optim.Adam([
        {"params":language_params,"lr":HEAD_LEARNING_RATE},
        {"params":child_params,"lr":CHILD_LEARNING_RATE},
    ])
    old53751={k:v.detach().cpu().half().clone()
              for k,v in
              adapter.child.child.shadow.unified.nra3_neural_weights.state_dict().items()}
    state_before={
        "head_up":adapter.language_head.up.weight.detach().cpu().clone(),
        "head_bias":adapter.language_head.output_bias.detach().cpu().clone(),
    }
    epochs=[]
    total=0;opt_updates=0;first_joint_norms=None
    for epoch in range(EPOCHS):
        ce=[];eos_ce=[];native_targets=0
        for row in train:
            gold=row["answer_native_with_eos"]
            optimizer.zero_grad(set_to_none=True)
            for token_index,target in enumerate(gold):
                prefix=row["prefix"]+gold[:token_index]
                check(len(prefix)<160 and
                      prefix[:len(row["prefix"])]==row["prefix"] and
                      prefix[len(row["prefix"]):]==gold[:token_index],
                      "current/future target cannot leak into native decoder")
                logits=real_joint_native(
                    model,sources,bridge,full,expansion,adapter,prefix,
                    row["prompt_features"],contextual=contextual)
                loss=F.cross_entropy(
                    logits,torch.tensor([target],dtype=torch.long))
                check(bool(torch.isfinite(loss)) and loss.requires_grad,
                      "full original 49k causal CE includes EOS supervised")
                (loss/len(gold)).backward()
                total+=1
                native_targets+=1
                ce.append(float(loss.detach()))
                if token_index==len(gold)-1:
                    eos_ce.append(float(loss.detach()))
                del logits,loss
            if first_joint_norms is None and opt_updates>=2:
                grads={
                    "head_native_up":
                        adapter.language_head.up.weight.grad,
                    "head_global_native_bias":
                        adapter.language_head.output_bias.grad,
                    "core_rank4":
                        adapter.child.child.shadow.unified.core_growth.up.grad,
                    "NRA3_structured":
                        adapter.child.child.shadow.unified.structured_interface.fuse.weight.grad,
                    "NPA1_plan":
                        adapter.child.child.shadow.unified.planning_action_area.state_update.weight_ih.grad,
                }
                first_joint_norms={}
                for key,val in grads.items():
                    check(val is not None and bool(torch.isfinite(val).all())
                          and float(val.abs().sum())>1e-18,
                          "original full native teacher-forced EOS gradients reach "+key)
                    first_joint_norms[key]=float(torch.linalg.vector_norm(val))
            torch.nn.utils.clip_grad_norm_(
                language_params+child_params,1.0)
            optimizer.step()
            opt_updates+=1
            check(all(bool(torch.isfinite(p).all())
                      for p in language_params+child_params),
                  "all learned full-vocab native/head/Core/NRA3/NPA1 finite")
        epochs.append({
            "epoch":epoch+1,
            "actual_original_core12_decoder_forwards":native_targets,
            "optimizer_steps":len(train),
            "mean_TRAIN_full_native_BPE_CE_including_EOS":
                sum(ce)/len(ce),
            "mean_TRAIN_EOS_target_CE":sum(eos_ce)/len(eos_ce),
            "native_EOS_targets":len(eos_ce),
            "source_TRAIN_only":True,
        })
    check(opt_updates==len(train)*EPOCHS and
          first_joint_norms is not None and
          not torch.equal(state_before["head_up"],
                          adapter.language_head.up.weight.detach().cpu())
          and not torch.equal(state_before["head_bias"],
                              adapter.language_head.output_bias.detach().cpu()),
          "real full-vocab language head and true joint neural weights trained")
    check(all(torch.equal(old53751[k],v.detach().cpu().half())
              for k,v in
              adapter.child.child.shadow.unified.nra3_neural_weights.state_dict().items()),
          "original protected old S115 53751 FP16 NRA3 remain exact")
    return {
        "epochs":epochs,
        "total_actual_original_native_decoder_forwards":total,
        "real_ADAM_updates":opt_updates,
        "head_trainable_parameters":HEAD_PARAMETER_COUNT,
        "full_trainable_parameters":sum(p.numel() for p in adapter.trainables()),
        "first_real_joint_gradient_norms":first_joint_norms,
        "original_S115_53751_FP16_UNCHANGED":True,
        "full_target_sequence_including_real_EOS_not_first_BPE_only":True,
        "pure_original_new_handwritten_TRAIN_only":True,
    }


def own_native_dev(tok,model,sources,bridge,full,expansion,
                   example,*,variant,adapter=None):
    check(example["split"]=="DEV" and variant in
          ("frozen_original","joint_contextual","joint_tokenlocal") and
          ((adapter is None)==(variant=="frozen_original")),
          "only new untouched DEV original and two trained variants")
    tokens=[]
    prefix=example["prefix"].copy()
    complete=False
    for n in range(DEV_MAX_GEN_NATIVE):
        check(len(prefix)<160 and
              prefix==example["prefix"]+tokens,
              "no gold answer token in free-running next-token native prefix")
        if adapter is None:
            logits,original_states=first_original_pass(
                model,sources,bridge,full,expansion,prefix,
                example["prompt_features"])
            del original_states
        else:
            logits=real_joint_native(
                model,sources,bridge,full,expansion,adapter,prefix,
                example["prompt_features"],
                contextual=(variant=="joint_contextual"))
        token=int(torch.argmax(logits.detach(),dim=-1))
        tokens.append(token)
        prefix.append(token)
        del logits
        if token==tok.eos_token_id:
            complete=True
            break
    text=tok.decode(tokens,skip_special_tokens=True)
    # ONLY post-decoder compare to answer, never add gold to model input.
    from research.calibri_nra3.s119q2g_fresh_own_token_source import (
        normalized,
    )
    exact=normalized(text)==normalized(
        example["gold_answer_for_scoring_ONLY"])
    return {
        "source_id":example["id"],"kind":example["kind"],
        "variant":variant,
        "actual_original_native_decoder_forwards":len(tokens),
        "own_generated_native_token_ids":tokens,
        "own_generated_text":text,
        "true_EOS_emitted":complete,
        "truncated_at_native_cap":not complete,
        "visible_short_answer_text_exact_not_EOS_certified":exact,
        "strict_full_short_answer_exact_WITH_EOS":exact and complete,
        "source_DEV_only_never_optimized":True,
        "not_blind_or_natural_benchmark":True,
    }


def run(s93,side,s142,q2d_dir,q2f_dir,out):
    check(not out.exists(),"never overwrite original research artifacts")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source_meta=ru_provenance()
    train_rows=new_ru_rows("TRAIN")
    dev_rows=new_ru_rows("DEV")
    check(len(train_rows)==N_TRAIN and len(dev_rows)==N_DEV and
          source_meta["origin"]==
              "original_newly_authored_examples_no_third_party_corpus" and
          source_meta["TRAIN_sha256"]==ru_source_hash(train_rows) and
          source_meta["DEV_sha256"]==ru_source_hash(dev_rows),
          "original NEW train/dev source provenance before any model load")
    q2d_parent=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CONTEXT,q2d_dir/Q2D_LOCAL,
        s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    check(q2f["stage"]==Q2F_STAGE
          and q2f["original_S93_SHA256"]==sha(s93)
          and q2f["original_S142_GT_SHA256"]==sha(s142)
          and q2f["parent_q2d_training_report_SHA256"]==
              sha(q2d_dir/Q2D_REPORT)
          and q2f["candidate_accepted"] is False
          and q2f["output_gate"]==0,
          "exact genuine prior physical trained q2f parent provenance")
    parent,_=load_q2d_trained_child(
        q2d_dir/Q2D_CONTEXT,
        q2d_parent["new_research_checkpoint_contextual"])
    q2f_child=load_physical_q2f_child(
        q2f_dir/Q2F_CONTEXT,
        q2f["projector_contextual_child_checkpoint"],parent)
    torch.manual_seed(SEED+1)
    ctx=JointRuNative(q2f_child)
    local=copy.deepcopy(ctx)
    check(all(torch.equal(v,w) for v,w in zip(
        ctx.state_dict().values(),local.state_dict().values()))
        and sum(p.numel() for p in ctx.trainables())==
            sum(p.numel() for p in local.trainables()),
        "SAME actual previous q2f-contextual trained weights, new language head init and identical params")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE and tok.eos_token_id is not None,
          "original FULL 49152 native BPE + original native EOS")
    train=[verified_native_answer(tok,r) for r in train_rows]
    dev=[verified_native_answer(tok,r) for r in dev_rows]
    check({r["prompt_sha256"] for r in train}.isdisjoint(
          r["prompt_sha256"] for r in dev)
          and len(train)==N_TRAIN and len(dev)==N_DEV,
          "no DEV in original TRAIN, SEALED EVAL never opened at all")
    original,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    original.model.add_module("nra3_s119m_onpath",ctx)
    original.eval()
    frozen_digest=old_original_param_digest(original)
    check(all(not p.requires_grad for name,p in original.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "all ORIGINAL S142 43M+GreenTunnel and original output head frozen")
    before_original_params=list(original.lm_head.parameters())
    check(before_original_params and
          all(not p.requires_grad for p in before_original_params),
          "original native full 49152 LM head remains frozen")
    result_ctx=train_joint(
        original,sources,bridge,full,expansion,ctx,train,contextual=True)
    result_local=train_joint(
        original,sources,bridge,full,expansion,local,train,contextual=False)
    check(result_ctx["real_ADAM_updates"]==
          result_local["real_ADAM_updates"]==N_TRAIN*EPOCHS and
          result_ctx["full_trainable_parameters"]==
          result_local["full_trainable_parameters"] and
          result_ctx["total_actual_original_native_decoder_forwards"]==
          result_local["total_actual_original_native_decoder_forwards"],
          "identical source native full answer targets, matched params and optimizer update budget")
    scored=defaultdict(list)
    dev_decoder_calls=0
    for row in dev:
        for variant,adapter in (
            ("frozen_original",None),
            ("joint_contextual",ctx),
            ("joint_tokenlocal",local),
        ):
            report=own_native_dev(
                tok,original,sources,bridge,full,expansion,row,
                variant=variant,adapter=adapter)
            dev_decoder_calls+=report["actual_original_native_decoder_forwards"]
            scored[variant].append(report)
    for variant,trace in scored.items():
        check(len(trace)==N_DEV and
              Counter(x["kind"] for x in trace)=={k:2 for k in KINDS},
              "exact six domains, two DEV full autoregressive question each")
    check(old_original_param_digest(original)==frozen_digest
          and all(not p.requires_grad for p in original.lm_head.parameters())
          and ctx.output_gate==local.output_gate==0
          and not ctx.candidate_accepted and not local.candidate_accepted,
          "original core+GT+LM native head unchanged, research gate=0 only")
    out.mkdir(parents=True)
    checkpoints={
        "joint_contextual":persist_half(ctx,out/CONTEXT_CKPT),
        "joint_tokenlocal":persist_half(local,out/LOCAL_CKPT),
    }
    summary={
        variant:{
            "strict_correct_short_answers_WITH_NATIVE_EOS":
                sum(x["strict_full_short_answer_exact_WITH_EOS"]
                    for x in trace),
            "visible_exact_text_even_if_EOS_missing":
                sum(x["visible_short_answer_text_exact_not_EOS_certified"]
                    for x in trace),
            "true_EOS_finished_count":
                sum(x["true_EOS_emitted"] for x in trace),
            "per_domain_strict_correct_count":{
                kind:sum(x["strict_full_short_answer_exact_WITH_EOS"]
                         for x in trace if x["kind"]==kind)
                for kind in KINDS,
            },
        } for variant,trace in scored.items()
    }
    result={
        "stage":STAGE,
        "original_S93_SHA256":sha(s93),
        "original_S142_43M_GT_SHA256":sha(s142),
        "prior_physically_trained_q2f_report_sha256":sha(q2f_dir/Q2F_REPORT),
        "original_author_new_Russian_source_provenance":source_meta,
        "all_full_original_native_BPE_classes":VOCAB_SIZE,
        "genuine_training_full_target_sequence_and_real_EOS":True,
        "real_original_decoder_forward_calls_TRAIN_context":
            result_ctx["total_actual_original_native_decoder_forwards"],
        "real_original_decoder_forward_calls_TRAIN_local":
            result_local["total_actual_original_native_decoder_forwards"],
        "real_original_decoder_forward_calls_DEV_own_token":
            dev_decoder_calls,
        "no_old_S118_B_GOLD_or_EVAL128_J7":True,
        "reserved_EVAL_not_used_in_optimization_or_DEV_scoring":True,
        "source_handwritten_synthetic_DEV_not_blind":True,
        "original_Core12_GreenTunnel_and_original_native_LM_head_unchanged":True,
        "old_S115_53751_FP16_unchanged":True,
        "candidate_accepted":False,"output_gate":0,
        "training_contextual":result_ctx,
        "training_matched_tokenlocal":result_local,
        "DEV_own_token_summary":summary,
        "DEV_own_token_full_traces":dict(scored),
        "learned_checkpoint_fp16":checkpoints,
        "native_head_NEW_lowrank4_plus_output_bias_params":HEAD_PARAMETER_COUNT,
        "no_matched_KV_cache_CPU_or_original_616J7_EOS_or_allin_bpp_cert":True,
        "no_LIVE_Render_Supabase_Jessica_chat_main_APK_secrets":True,
        "decision":"RESEARCH_ONLY_NEVER_PROMOTE_WITHOUT_INDEPENDENT_VALIDATION_AND_4_GATES",
    }
    (out/REPORT).write_text(json.dumps(
        result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I1_TRUE_ORIGINAL43M_RUSSIAN_NATIVE_FULLANSWER_EOS "+
          json.dumps({
              "source_train":N_TRAIN,"source_dev":N_DEV,
              "native_targets_per_variant":result_ctx[
                  "total_actual_original_native_decoder_forwards"],
              "new_learnable_native_head_params":HEAD_PARAMETER_COUNT,
              "joint_optimizer_steps_per_variant":N_TRAIN*EPOCHS,
              "dev_own_BPE_original_full_forward_calls":dev_decoder_calls,
              "DEV_strict_full_answer_EOS":{
                  key:value["strict_correct_short_answers_WITH_NATIVE_EOS"]
                  for key,value in summary.items()},
              "DEV_visible_text_correct":{
                  key:value["visible_exact_text_even_if_EOS_missing"]
                  for key,value in summary.items()},
              "original_core_and_native_head_frozen":True,
              "candidate_accepted":False,
          },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.out)


if __name__=="__main__":
    main()
