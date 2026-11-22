"""CALIBRI S119q2d: DEEPER multi-example MULTI-POSITION native-BPE joint training.

Research-only genuine physical S142 frozen Core12 + Green Tunnel with
TRAINABLE learned rank4 Core residual, NRA3 structured mini-regions and NPA1
planning on a onepass interlayer hook. Original pretrained model and S115
53751 FP16 weights remain immutable. We continue FROM BOTH real S119q2c1
trained contextual/local children; do not restart from empty or synthesize
success. Exact same 96 TRAIN rows / 4 epochs / optimizer budget each.

72 original S119q2a TRAIN + 24 distinct S119q2c1 original counterfactual
TRAIN, six equal categories. A **single** next native answer token per
example per epoch, rotating across answer positions by fixed predeclared
schedule. Only its PRECEDING correct answer tokens are teacher forced on
the SAME native causal prefix. Answer target and future tokens NEVER enter
model input. Thus this is teacher-forced NEXT-TOKEN learning at multiple
answer positions, NOT full free-running answer generation or independent QA.

4 predeclared epochs, 384 physical ORIGINAL onepass forward calls each,
gradient accumulate 4 = 96 Adam steps per variant. Paired contrasts on
counterfactual FIRST-DISTINGUISHING native token only (coefficient 0.2).
Same optimizer settings and parameter count contextual vs matched local.
No model promotion or user-chat Jessica; no spent GOLD/EVAL/J7 answers.
The q2a 18 CALIBRATION and q2c1 12 SOURCE_PROBE prompts are explicitly
NEVER used for training; both are synthetic known-source diagnostics and
cannot be treated as new independent test or proof of intelligence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    LearnedOnPathInterlayer,old_original_param_digest,
)
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    NativeFullHeadShadow,
)
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    prompt_features,
)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    strict_chat_source_ids,
)
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    onepass_full_native_logits,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    verify_q2a_artifact,persist_half,
)
from research.calibri_nra3.s119q2c1_author_counterfactual_pairs import (
    rows as paired_rows,split as paired_split,KINDS,
    EXPECTED_SOURCE_SHA256,
)
from research.calibri_nra3.s119q2c1_real43m_onepass_counterfactual_train import (
    STAGE as PARENT_STAGE,REPORT as PARENT_REPORT,
    CONTEXT_CKPT as PARENT_CONTEXT_CKPT,
    LOCAL_CKPT as PARENT_LOCAL_CKPT,
)

STAGE="CALIBRI_S119Q2D_REAL_S142_43M_GT_96TRAIN_X4_NATIVE_MULTI_POSITION"
REPORT="S119Q2D_DEEPER_REAL43M_GT_96TRAIN4EPOCH_NATIVE_POSITION_ABLATION.json"
CONTEXT_CKPT="S119Q2D_DEEPER_CONTEXTUAL_NATIVE_96TRAIN4EPOCH_GATE0_FP16.safetensors"
LOCAL_CKPT="S119Q2D_DEEPER_LOCAL_NATIVE_96TRAIN4EPOCH_GATE0_FP16.safetensors"
SEED=119133
EPOCHS=4
EXAMPLES_PER_EPOCH=96
ACCUMULATION=4
LR=0.0005
CONTRAST_WEIGHT=0.2
VOCAB_SIZE=49152
OLD_PREFIX="model.nra3_calibri_joint.s119q2b."
SOURCE_COUNTS={kind:16 for kind in KINDS}
PROBE_COUNTS={"CALIBRATION":18,"SOURCE_PROBE":12}


def read_parent(context_file,local_file,report_file,s93,s142):
    check(context_file.name==PARENT_CONTEXT_CKPT and
          local_file.name==PARENT_LOCAL_CKPT and
          report_file.name==PARENT_REPORT,
          "exact real q2c1 physical paired-trained children only")
    parent=json.loads(report_file.read_text(encoding="utf8"))
    check(parent["stage"]==PARENT_STAGE and
          parent["original_core_S93_SHA256"]==sha(s93) and
          parent["original_S142_43M_GT_SHA256"]==sha(s142) and
          parent["q2c1_new_authored_counterfactual_source_SHA256"]==
               EXPECTED_SOURCE_SHA256 and
          parent["original_core_GT_weights_and_53751_S115_FP16_unchanged"]
          and parent["output_gate"]==0 and
          parent["candidate_accepted"] is False and
          parent["train_24_rows_disjoint_probe_12_rows"],
          "original q2c1 physical research parent failed provenance gates")
    models=[]
    for file_path,meta_name in (
        (context_file,"contextual_checkpoint_FP16_research_only"),
        (local_file,"local_checkpoint_FP16_research_only"),
    ):
        meta=parent[meta_name]
        check(meta["sha256"]==sha(file_path) and
              meta["file_bytes"]==file_path.stat().st_size and
              meta["saved_total_child_FP16_params"]>150000,
              "exact physical A/B parent SHA and tensor count required")
        raw=load_file(str(file_path),device="cpu")
        check(raw and
              all(k.startswith(OLD_PREFIX) and v.dtype==torch.float16
                  for k,v in raw.items()),
              "genuine original FP16 q2c1 all NRA3/NPA1 child tensors")
        model=LearnedOnPathInterlayer(NativeFullHeadShadow()).eval()
        state={k[len(OLD_PREFIX):]:v.float() for k,v in raw.items()}
        check(set(state)==set(model.state_dict()),
              "cannot drop original 53,751 or any trained core/planner weights")
        model.load_state_dict(state,strict=True)
        check(sum(p.numel() for p in model.shadow.unified.nra3_neural_weights.parameters())==
              53751 and model.output_gate==0 and
              not model.candidate_accepted and not model.activation_allowed,
              "still ONE unaccepted original 53,751 and grown joint child")
        models.append((model,raw))
    return parent,models[0],models[1]


def source_inventory(q2a_rows):
    old=[r for r in q2a_rows if r["split"]=="TRAIN"]
    cal=[r for r in q2a_rows if r["split"]=="CALIBRATION"]
    new_train,new_probe,meta=paired_split(paired_rows())
    check(len(old)==72 and len(cal)==18 and
          len(new_train)==24 and len(new_probe)==12 and
          meta["source_sha256"]==EXPECTED_SOURCE_SHA256,
          "six categories; only 72+24 known source TRAIN")
    train=[];probe=[]
    for r in old:
        train.append({"id":r["id"],"kind":r["kind"],
                      "prompt":r["prompt"],"answer":r["completion"],
                      "source_family":"q2a","source_split":"TRAIN",
                      "pair_index":None,"side":None})
    for r in new_train:
        train.append({"id":r["id"],"kind":r["kind"],
                      "prompt":r["prompt"],"answer":r["completion"],
                      "source_family":"q2c1","source_split":"TRAIN",
                      "pair_index":r["pair_index"],"side":r["side"]})
    for r in cal:
        probe.append({"id":r["id"],"kind":r["kind"],
                      "prompt":r["prompt"],"answer":r["completion"],
                      "source_family":"q2a","source_split":"CALIBRATION",
                      "pair_index":None,"side":None})
    for r in new_probe:
        probe.append({"id":r["id"],"kind":r["kind"],
                      "prompt":r["prompt"],"answer":r["completion"],
                      "source_family":"q2c1","source_split":"SOURCE_PROBE",
                      "pair_index":r["pair_index"],"side":r["side"]})
    check(len(train)==96 and len(probe)==30 and
          Counter(x["kind"] for x in train)==SOURCE_COUNTS and
          not {x["prompt"] for x in train}&{x["prompt"] for x in probe} and
          len({x["prompt"] for x in train+probe})==126,
          "96 TRAIN balanced; source-only probes never in gradient data")
    return train,probe


def pack_sources(tok,train,probe):
    """Native target IDs are checked against FULL original assistant prompt."""
    all_data=[]
    for entry in train+probe:
        prefix=strict_chat_source_ids(tok,entry["prompt"])
        completion=list(tok.encode(entry["answer"],add_special_tokens=False))
        check(1<=len(completion)<=64 and
              all(type(token) is int and 0<=token<VOCAB_SIZE
                  for token in completion) and
              tok.decode(completion,skip_special_tokens=True).strip()==
                entry["answer"].strip(),
              "physical full authored completion native BPE reversible")
        # A BPE token may straddle the boundary between original assistant
        # header and answer. Fail-closed unless the true combined prompt
        # tokenization is exactly question_prefix + target native tokens.
        full_prompt=tok.apply_chat_template(
            [{"role":"user","content":entry["prompt"]}],
            tokenize=False,add_generation_prompt=True)
        combined=list(tok.encode(full_prompt+entry["answer"],
                                 add_special_tokens=False))
        check(combined==prefix+completion,
              "native assistant boundary BPE must match actual concatenation")
        check(2<=len(prefix)<160 and
              len(prefix)+len(completion)<160,
              "one ORIGINAL native prefix per teacher forced token <=159")
        all_data.append({
            **entry,
            "prefix":prefix,
            "completion_ids":completion,
            "prompt_features":prompt_features(tok,entry["prompt"]),
            "prompt_sha256":hashlib.sha256(
                entry["prompt"].encode("utf8")).hexdigest(),
        })
    check(len(all_data)==126 and
          {r["prompt_sha256"] for r in all_data[:96]}.isdisjoint(
            r["prompt_sha256"] for r in all_data[96:]),
          "no future probe source prompt leakage into training")
    grouped={(r["kind"],r["pair_index"],r["side"]):r for r in all_data
             if r["source_family"]=="q2c1"}
    check(len(grouped)==36,
          "exact unique original authored q2c1 counterfactual pairs")
    for x in all_data:
        x["first_distinct_position"]=None
        x["wrong_distinct_native"]=None
        if x["source_family"]=="q2c1":
            opposite=grouped[(x["kind"],x["pair_index"],1-x["side"])]
            aa=x["completion_ids"];bb=opposite["completion_ids"]
            position=0
            while position<min(len(aa),len(bb)) and aa[position]==bb[position]:
                position+=1
            check(position<min(len(aa),len(bb))
                  and aa[position]!=bb[position],
                  "full original BPE paired answer has one causal divergence")
            x["first_distinct_position"]=position
            x["wrong_distinct_native"]=bb[position]
    return all_data[:96],all_data[96:]


def scheduled_position(example,epoch):
    """Precommitted position rotations; do not select positions by loss."""
    check(0<=epoch<EPOCHS and example["source_split"]=="TRAIN",
          "source-only gradient scheduling")
    n=len(example["completion_ids"])
    check(n>=1,"native answer token targets nonempty")
    if example["source_family"]=="q2c1":
        position=(example["first_distinct_position"]+epoch)%n
    else:
        fixed=int(example["prompt_sha256"][:8],16)%n
        position=(fixed+epoch)%n
    check(position<n and position>=0,"full native word positions, no GOLD")
    return position


def native_step(model,sources,bridge,full,expansion,adapter,example,position,
                *,contextual,gradient):
    answer=example["completion_ids"]
    check(0<=position<len(answer) and
          example["source_split"] in ("TRAIN","CALIBRATION","SOURCE_PROBE"),
          "native one step with preserved source split")
    prefix=example["prefix"]+answer[:position]
    check(prefix[-len(answer[:position]):] == answer[:position]
          if position else True,"ONLY observed correct earlier answer token(s)")
    check(answer[position] not in [] and len(prefix)<160,
          "target and future answer never in physical input")
    logits,witness=onepass_full_native_logits(
        model,sources,bridge,full,expansion,adapter,prefix,
        example["prompt_features"],contextual=contextual)
    check(logits.shape==(1,VOCAB_SIZE) and
          witness["candidate_forward_calls"]==1,
          "ONE true physical original decoder forward native vocab")
    target=int(answer[position])
    ce=F.cross_entropy(
        logits,torch.tensor([target],dtype=torch.long))
    contrast=torch.zeros_like(ce)
    contrasted=False
    if (example["source_family"]=="q2c1" and
        position==example["first_distinct_position"]):
        opposite=example["wrong_distinct_native"]
        check(opposite!=target,
              "only FIRST distinct opposing native BPE token contrasted")
        contrast=F.softplus(logits[0,opposite]-logits[0,target])
        contrasted=True
    total=ce+CONTRAST_WEIGHT*contrast
    correct_native=int(logits.detach().argmax(-1))==target
    note={"target_position":position,
          "first_discriminating_pair_position":
              example["first_distinct_position"],
          "target_native_id":target,
          "next_native_CE":float(ce.detach()),
          "correct_native_greedy_token":correct_native,
          "contrast_applied":contrasted,
          "contrast_softplus":float(contrast.detach()),
          "source_family":example["source_family"],
          "kind":example["kind"],"source_id":example["id"]}
    if not gradient:
        return None,note
    check(total.requires_grad and bool(torch.isfinite(total)),
          "true 49152 native CE must pass ORIGINAL later layers to trainable child")
    return total,note


def optimize_variant(model,sources,bridge,full,expansion,adapter,train,
                     *,contextual):
    check(len(train)==EXAMPLES_PER_EPOCH and
          all(r["source_split"]=="TRAIN" for r in train),
          "only precommitted 96 TRAIN source rows")
    params=list(adapter.trainables())
    check(len({id(p) for p in params})==len(params)
          and len(params)>2 and all(p.requires_grad for p in params),
          "all new Core-rank4, NRA3 structured and NPA1 planner trainables")
    param_count=sum(x.numel() for x in params)
    original_old={
        k:v.detach().half().cpu().clone()
        for k,v in adapter.shadow.unified.nra3_neural_weights.state_dict().items()
    }
    optimizer=torch.optim.Adam(params,lr=LR)
    before_gates=adapter.midlayer_gates.detach().cpu().tolist()
    updates=0;decoder_calls=0
    epoch_summaries=[];position_inventory=Counter()
    deep_grad=None
    for epoch in range(EPOCHS):
        epoch_ce=[];epoch_accuracy=0;epoch_contrasts=0
        for sample_idx,ex in enumerate(train):
            position=scheduled_position(ex,epoch)
            loss,note=native_step(model,sources,bridge,full,expansion,adapter,
                                  ex,position,contextual=contextual,
                                  gradient=True)
            (loss/ACCUMULATION).backward()
            decoder_calls+=1
            epoch_ce.append(note["next_native_CE"])
            epoch_accuracy+=note["correct_native_greedy_token"]
            epoch_contrasts+=note["contrast_applied"]
            position_inventory[(ex["source_family"],position)]+=1
            if deep_grad is None and (sample_idx >= ACCUMULATION):
                gate=adapter.midlayer_gates.grad
                components={
                    "core_rank4":adapter.shadow.unified.core_growth.up.grad,
                    "NRA3_structured":adapter.shadow.unified.structured_interface.fuse.weight.grad,
                    "NPA1_planning":
                        adapter.shadow.unified.planning_action_area.state_update.weight_ih.grad,
                }
                check(gate is not None and bool(torch.isfinite(gate).all())
                      and float(gate.abs().sum())>1.e-12,
                      "true frozen original native gradient into neural area gates")
                deep_grad={"three_midlayer_gates":
                           float(torch.linalg.vector_norm(gate).item())}
                for name,v in components.items():
                    check(v is not None and bool(torch.isfinite(v).all())
                          and float(v.abs().sum())>1.e-18,
                          "true ORIGINAL onpath gradient into new "+name)
                    deep_grad[name]=float(torch.linalg.vector_norm(v).item())
            if (sample_idx+1)%ACCUMULATION==0:
                torch.nn.utils.clip_grad_norm_(params,1.0)
                check(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                          for p in params if p.grad is not None),
                      "no NaN in original native joint trainable gradients")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates+=1
                check(all(bool(torch.isfinite(p).all()) for p in params),
                      "all trainable original-core integrated modules remain finite")
        epoch_summaries.append({
            "epoch":epoch+1,
            "original_native_single_pass_training_forwards":EXAMPLES_PER_EPOCH,
            "optimizer_updates":EXAMPLES_PER_EPOCH//ACCUMULATION,
            "mean_source_TRAIN_next_native_CE":sum(epoch_ce)/len(epoch_ce),
            "last_source_TRAIN_native_CE":epoch_ce[-1],
            "source_TRAIN_next_native_greedy_correct":epoch_accuracy,
            "paired_counterfactual_distinct_token_loss_cases":
                epoch_contrasts,
        })
    check(updates==EXAMPLES_PER_EPOCH//ACCUMULATION*EPOCHS and
          decoder_calls==EXAMPLES_PER_EPOCH*EPOCHS and
          deep_grad is not None,
          "physical onepass 96 training examples x4 fixed epochs with 4 accumulation")
    check(all(torch.equal(
        original_old[k],v.detach().half().cpu())
        for k,v in adapter.shadow.unified.nra3_neural_weights.state_dict().items()),
        "S115 original 53,751 FP16 is unchanged in deep training")
    return {"epochs":epoch_summaries,
            "actual_full_original_decoder_forwards":decoder_calls,
            "optimizer_updates":updates,
            "trainable_param_count":param_count,
            "first_actual_joint_gradient_norms":deep_grad,
            "initial_gates":before_gates,
            "final_gates":adapter.midlayer_gates.detach().cpu().tolist(),
            "target_positions":{f"{a}:{pos}":n for (a,pos),n
                                in sorted(position_inventory.items())},
            "all_source_examples_TRAIN_only":True}


def source_diagnostic(model,sources,bridge,full,expansion,adapter,probes,
                      *,contextual):
    check(len(probes)==30 and
          Counter(x["source_split"] for x in probes)==PROBE_COUNTS,
          "ONLY q2a calibration and q2c1 previously-used source-only probes")
    trace=[];ce_by_kind=defaultdict(list)
    n_correct=0;pair_preference={}
    for ex in probes:
        pos=(ex["first_distinct_position"] if ex["source_family"]=="q2c1"
             else 0)
        _,note=native_step(model,sources,bridge,full,expansion,adapter,ex,pos,
                           contextual=contextual,gradient=False)
        n_correct+=note["correct_native_greedy_token"]
        ce_by_kind[ex["kind"]].append(note["next_native_CE"])
        trace.append(note)
    return {
        "count_source_probes":len(trace),
        "mean_source_only_native_CE":
            sum(z["next_native_CE"] for z in trace)/len(trace),
        "source_only_greedy_correct":n_correct,
        "mean_source_only_by_domain":{
            k:sum(arr)/len(arr) for k,arr in sorted(ce_by_kind.items())},
        "native_source_details":trace,
        "not_blind_independent_or_natural_full_answer_generation":True,
    }


def run(s93,side,s142,q2a_dir,parent_ctx,parent_local,parent_report,out):
    check(not out.exists(),"never overwrite trained research children")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    q2a_rows,source_manifest=verify_q2a_artifact(q2a_dir)
    train_raw,probes_raw=source_inventory(q2a_rows)
    source_parent,(ctx,ctx_fp16),(local,local_fp16)=read_parent(
        parent_ctx,parent_local,parent_report,s93,s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE,
          "original full native BPE tokenizer, no 162-token proxy")
    train,probes=pack_sources(tok,train_raw,probes_raw)
    original,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    original.model.add_module("nra3_s119m_onpath",ctx)
    original.eval()
    orig_digest=old_original_param_digest(original)
    check(all(not p.requires_grad for n,p in original.named_parameters()
              if "nra3_s119m_onpath" not in n)
          and ctx.output_gate==local.output_gate==0
          and not ctx.candidate_accepted and not local.candidate_accepted,
          "entire original frozen Core12/GreenTunnel; gate0 research only")
    context_start={
        k:v.detach().cpu().clone() for k,v in ctx.state_dict().items()
    }
    local_start={
        k:v.detach().cpu().clone() for k,v in local.state_dict().items()
    }
    train_ctx=optimize_variant(
        original,sources,bridge,full,expansion,ctx,train,contextual=True)
    train_local=optimize_variant(
        original,sources,bridge,full,expansion,local,train,contextual=False)
    check(train_ctx["trainable_param_count"]==train_local["trainable_param_count"]
          and train_ctx["actual_full_original_decoder_forwards"]==
              train_local["actual_full_original_decoder_forwards"]==
              EXAMPLES_PER_EPOCH*EPOCHS and
          train_ctx["optimizer_updates"]==train_local["optimizer_updates"]==
              EXAMPLES_PER_EPOCH//ACCUMULATION*EPOCHS,
          "perfect source/optimizer/parameter matched A/B deep training")
    after_ctx=source_diagnostic(
        original,sources,bridge,full,expansion,ctx,probes,contextual=True)
    after_local=source_diagnostic(
        original,sources,bridge,full,expansion,local,probes,contextual=False)
    check(old_original_param_digest(original)==orig_digest,
          "immutable actual pretrained original 43M+GT EXACT after deep training")
    for adapter,raw in ((ctx,ctx_fp16),(local,local_fp16)):
        for k,v in raw.items():
            if ".nra3_neural_weights." in k:
                short=k.removeprefix(OLD_PREFIX)
                check(torch.equal(
                    adapter.state_dict()[short].detach().half().cpu(),v),
                    "original old S115 53751 FP16 unchanged in both trained children")
        check(adapter.output_gate==0 and not adapter.candidate_accepted
              and not adapter.activation_allowed,
              "NEVER accept or activate trained experimental models")
    changed_ctx=sum(not torch.equal(v,ctx.state_dict()[k].detach().cpu())
                    for k,v in context_start.items())
    changed_local=sum(not torch.equal(v,local.state_dict()[k].detach().cpu())
                      for k,v in local_start.items())
    check(changed_ctx>0 and changed_local>0,
          "actually TRAIN new neural weights for both original-native children")
    out.mkdir(parents=True)
    saved_ctx=persist_half(ctx,out/CONTEXT_CKPT)
    saved_local=persist_half(local,out/LOCAL_CKPT)
    report={
        "stage":STAGE,
        "original_S93_SHA256":sha(s93),
        "original_S142_43M_GT_SHA256":sha(s142),
        "parent_previous_q2c1_report_SHA256":sha(parent_report),
        "q2a_source_SHA256":source_manifest["synthetic_rows_SHA256"],
        "q2c1_source_SHA256":EXPECTED_SOURCE_SHA256,
        "real_original_full_native_BPE_classes":VOCAB_SIZE,
        "frozen_original_12layer_Core_GreenTunnel_unchanged":True,
        "full_old_original_53751_NRA3_FP16_unchanged":True,
        "new_joint_Corerank4_NRA3_NPA1_trained_inside_original_onpath":True,
        "native_question_and_preceding_answer_teacher_forcing_ONLY":True,
        "rotating_answer_position_not_complete_free_running_answer":True,
        "CONTRASTIVE_native_distinct_BPE_penalty":CONTRAST_WEIGHT,
        "train_96_source_rows_domains":SOURCE_COUNTS,
        "calibration_18_and_source_probe_12_NEVER_used_for_gradient":True,
        "continued_FROM_previously_trained_q2c1_contextual_and_local":True,
        "physical_original_forward_per_training_sample_step":1,
        "contextual_training":train_ctx,
        "matched_local_training":train_local,
        "updated_tensors_contextual":changed_ctx,
        "updated_tensors_local":changed_local,
        "source_diagnostic_contextual":after_ctx,
        "source_diagnostic_local":after_local,
        "source_diagnostic_contextual_minus_local_CE":
           after_ctx["mean_source_only_native_CE"]-
           after_local["mean_source_only_native_CE"],
        "source_diagnostic_contextual_wins_same_author_only":
           after_ctx["mean_source_only_native_CE"]<
           after_local["mean_source_only_native_CE"],
        "source_probes_NOT_blind_or_independent":True,
        "total_physical_original_decoder_calls":
           EXAMPLES_PER_EPOCH*EPOCHS*2+len(probes_raw)*2,
        "new_research_checkpoint_contextual":saved_ctx,
        "new_research_checkpoint_local":saved_local,
        "no_original_full616_J7_EOS_replay":True,
        "no_original_packed_allin_bpp_cert":True,
        "no_paired_CPU_KV_generation_throughput":True,
        "no_independent_quality_or_full_answer_gain_claim":True,
        "no_S118_B_GOLD_spent_EVAL128_original_J7_answers":True,
        "no_Jessica_chat_LIVE_Render_Supabase_main_APK_or_deploy":True,
        "output_gate":0,"candidate_accepted":False,
        "decision":"JOINT_96TRAIN_X4_NATIVE_ANSWER_POSITION_LEARNING_ONLY_NOT_ACCEPTED",
    }
    (out/REPORT).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2D_TRUE_ORIGINAL43M_JOINT_DEEPER_TRAIN "+json.dumps({
        "train_distinct_rows":len(train),
        "epochs":EPOCHS,
        "original_decoder_forward_total":
            report["total_physical_original_decoder_calls"],
        "optimizer_updates_each":train_ctx["optimizer_updates"],
        "trainable_param_equal":
            train_ctx["trainable_param_count"]==
            train_local["trainable_param_count"],
        "first_epoch_train_CE_ctx":
            train_ctx["epochs"][0]["mean_source_TRAIN_next_native_CE"],
        "last_epoch_train_CE_ctx":
            train_ctx["epochs"][-1]["mean_source_TRAIN_next_native_CE"],
        "last_epoch_train_CE_local":
            train_local["epochs"][-1]["mean_source_TRAIN_next_native_CE"],
        "source_probe_CE_ctx":after_ctx["mean_source_only_native_CE"],
        "source_probe_CE_local":after_local["mean_source_only_native_CE"],
        "source_probe_ctx_wins_ONLY":
            report["source_diagnostic_contextual_wins_same_author_only"],
        "old_weights_changed":False,"accepted":False,
    },sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2a_dir","parent_ctx",
                 "parent_local","parent_report","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=parser.parse_args()
    run(a.s93,a.side,a.s142,a.q2a_dir,a.parent_ctx,a.parent_local,
        a.parent_report,a.out)


if __name__=="__main__":
    main()
