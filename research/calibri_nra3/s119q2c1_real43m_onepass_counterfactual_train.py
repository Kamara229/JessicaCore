"""S119q2c1: genuine original Calibri Core12+GT paired-counterfactual TRAIN.

Research-only stage; new source labels authored and preregistered in q2c1
(24 training rows in 12 contrast pairs; 12 never-trained same-author
SOURCE_PROBE rows). Avoid reusing spent q2b six probe prompts, old S119n,
GOLD S118B, all EVAL128, Jessica, and final independent QA.

TWO equal-parameter frozen-base original S142 native LM models differ ONLY
in causal structured-interface contextual=True versus token-local False.
The original 12-layer compressed FP16 43M+GT and old S115 53,751 trained
NRA3 FP16 are frozen. Every optimizer step uses ONE original physical
decoder forward with early h0..h5 available at layer5, true original
49152-token LM head and cross-entropy + a SMALL predeclared
counterfactual log-odds softplus loss against the flipped pair's FIRST
answer BPE. Multi-token native answer generation NOT trained or scored.
This is controlled mechanism learning, NOT benchmark quality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    load_parent,prompt_features,first_original_pass,
)
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    onepass_full_native_logits,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    persist_half,verify_q2a_artifact,
)
from research.calibri_nra3.s119q2c1_author_counterfactual_pairs import (
    STAGE as SOURCE_STAGE,KINDS,rows as authored_rows,
    split as split_source,
)

STAGE = "CALIBRI_S119Q2C1_REAL43M_GT_ONEPASS_BALANCED_PAIRED_NATIVE_TRAIN"
REPORT = "S119Q2C1_REAL43M_GT_PAIRED_SOURCE_ONLY_TRAIN_ABLATION.json"
CONTEXT_CKPT = "S119Q2C1_CONTEXTUAL_24PAIRSOURCE_TRAIN_ONLY_FP16_UNACCEPTED.safetensors"
LOCAL_CKPT = "S119Q2C1_LOCAL_MATCHED_24PAIRSOURCE_TRAIN_ONLY_FP16_UNACCEPTED.safetensors"
SEED = 119132
EPOCHS = 1
LR = 0.001
CONTRAST_WEIGHT = 0.25
VOCAB_SIZE = 49152


def prepared_source(tok,records):
    check(len(records) in (24,12) and
          all(x["rule_kind"]==
              "authored_semantic_counterfactual_not_independent_gold"
              for x in records),
          "strict original authored balanced counterfactual source only")
    from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
        strict_chat_source_ids,
    )
    prepared=[]
    for row in records:
        question_prefix=strict_chat_source_ids(tok,row["prompt"])
        answer_ids=list(tok.encode(row["completion"],add_special_tokens=False))
        check(1 <= len(answer_ids) < 30
              and all(type(v) is int and 0<=v<VOCAB_SIZE for v in answer_ids)
              and tok.decode(answer_ids,skip_special_tokens=True).strip()==
                  row["completion"].strip(),
              "source answer BPE exact original tokenizer roundtrip")
        check(len(question_prefix)<160 and len(question_prefix)>=2,
              "genuine native original prompt only, no future target in input")
        prepared.append({
            "source_id":row["id"],"kind":row["kind"],"use":row["use"],
            "pair_index":row["pair_index"],"side":row["side"],
            "prompt_sha256":hashlib.sha256(
                row["prompt"].encode("utf8")).hexdigest(),
            "prefix":question_prefix,
            "original_question_prefix_length":len(question_prefix),
            "prompt_features":prompt_features(tok,row["prompt"]),
            "full_authored_answer_native_ids":answer_ids,
            "completion_native_tokens":len(answer_ids),
        })
    lookup={(r["kind"],r["pair_index"],r["side"]):r
            for r in prepared}
    check(len(lookup)==len(prepared),
          "no paired contrast aliases or repetition")
    for item in prepared:
        other=lookup[(item["kind"],item["pair_index"],1-item["side"])]
        a=item["full_authored_answer_native_ids"]
        b=other["full_authored_answer_native_ids"]
        first_difference=0
        while (first_difference<min(len(a),len(b)) and
               a[first_difference]==b[first_difference]):
            first_difference+=1
        check(item["use"]==other["use"]
              and first_difference<min(len(a),len(b)),
              "counterfactual answer must differ BEFORE one complete answer is a token prefix")
        # For Cyrillic BPE, the FIRST answer token can be identical for two
        # *different* words. Train/evaluate the FIRST DISTINGUISHING token;
        # only the COMMON preceding answer tokens (legitimate causal
        # teacher forcing) may be appended to the question prefix.
        common=a[:first_difference]
        check(common==b[:first_difference] and
              a[first_difference]!=b[first_difference],
              "only shared teacher-forced prefix before opposing native targets")
        item["prefix"]=item["prefix"]+common
        item["teacher_forced_shared_answer_native_ids"]=common
        item["teacher_forced_shared_answer_prefix_tokens"]=len(common)
        item["native_target"]=int(a[first_difference])
        item["opposing_native_distinct_token"]=int(b[first_difference])
        check(len(item["prefix"])<160 and
              item["native_target"]!=item["opposing_native_distinct_token"],
              "one original causal prefix, first discriminating next BPE only")
    return prepared

def paired_instant_metrics(logits,correct: int,opposing: int):
    check(logits.shape==(1,VOCAB_SIZE) and
          bool(torch.isfinite(logits).all())
          and 0<=correct<VOCAB_SIZE and 0<=opposing<VOCAB_SIZE
          and correct!=opposing,
          "genuine original full-native head and predeclared counterfactual")
    delta=logits[0,correct]-logits[0,opposing]
    check(bool(torch.isfinite(delta)),"native answer log odds finite")
    return {
        "native_distinguishing_BPE_CE":float(F.cross_entropy(
            logits,torch.tensor([correct],dtype=torch.long)).detach()),
        "native_distinguishing_truth_minus_counterfactual_logit":float(delta.detach()),
        "correct_ranked_over_opposing":bool(delta.detach()>0),
        "native_distinguishing_BPE_greedy_correct":
            int(logits.detach().argmax(-1))==correct,
    }


def paired_native_loss(model,sources,bridge,full,expansion,adapter,ex,
                       *,contextual):
    logits,witness=onepass_full_native_logits(
        model,sources,bridge,full,expansion,adapter,
        ex["prefix"],ex["prompt_features"],contextual=contextual)
    correct=ex["native_target"]
    wrong=ex["opposing_native_distinct_token"]
    stats=paired_instant_metrics(logits,correct,wrong)
    ce=F.cross_entropy(
        logits,torch.tensor([correct],dtype=torch.long))
    contrast=F.softplus(logits[0,wrong]-logits[0,correct])
    loss=ce+CONTRAST_WEIGHT*contrast
    check(loss.requires_grad and bool(torch.isfinite(loss)) and
          witness["candidate_forward_calls"]==1,
          "one actual original decoder forward with both gradient objectives")
    return loss,stats


def train_variant(model,sources,bridge,full,expansion,adapter,
                  train,*,contextual):
    check(len(train)==24 and EPOCHS==1 and
          all(r["use"]=="TRAIN" and r["pair_index"] in (0,1) for r in train),
          "only 24 fresh paired authored TRAIN targets")
    params=list(adapter.trainables())
    check(params and len({id(p) for p in params})==len(params)
          and all(p.requires_grad for p in params),
          "all unique new joint learned weights, old 53751 are frozen")
    ntrain=sum(x.numel() for x in params)
    old={
        k:v.detach().half().cpu().clone()
        for k,v in adapter.shadow.unified.nra3_neural_weights.state_dict().items()
    }
    optimizer=torch.optim.Adam(params,lr=LR)
    trace=[]
    source_steps=0
    deep_grad=None
    for epoch in range(EPOCHS):
        for ex in train:
            optimizer.zero_grad(set_to_none=True)
            loss,stats=paired_native_loss(
                model,sources,bridge,full,expansion,adapter,ex,
                contextual=contextual)
            loss.backward()
            gate_grad=adapter.midlayer_gates.grad
            check(gate_grad is not None and
                  bool(torch.isfinite(gate_grad).all()) and
                  float(gate_grad.abs().sum())>1e-12,
                  "physical original deep native head must reach trained gates")
            if deep_grad is None and source_steps>=1:
                observed={
                    "rank4_core":adapter.shadow.unified.core_growth.up.grad,
                    "structured_NRA3":adapter.shadow.unified.structured_interface.fuse.weight.grad,
                    "NPA1_recurrent_state":
                        adapter.shadow.unified.planning_action_area.state_update.weight_ih.grad,
                }
                deep_grad={}
                for name,gradient in observed.items():
                    check(gradient is not None and
                          bool(torch.isfinite(gradient).all())
                          and float(gradient.abs().sum())>1e-18,
                          "joint early neural native loss reaches "+name)
                    deep_grad[name]=float(
                        torch.linalg.vector_norm(gradient).item())
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optimizer.step()
            check(all(bool(torch.isfinite(p).all()) for p in params),
                  "TRAIN-only original native head Adam must remain finite")
            source_steps+=1
            trace.append({
                "epoch":epoch+1,"kind":ex["kind"],
                "source_id":ex["source_id"],
                "prompt_sha256":ex["prompt_sha256"],
                **stats,
                "combined_native_CE_plus_contrast_loss":float(loss.detach()),
            })
    check(source_steps==24 and deep_grad is not None,
          "exact paired original TRAIN-only optimizer budget")
    check(all(torch.equal(
        old[k],v.detach().half().cpu())
        for k,v in adapter.shadow.unified.nra3_neural_weights.state_dict().items()),
        "all original 53,751 FP16 neural weights unchanged post training")
    return {"trace":trace,"actual_optimizer_steps":source_steps,
            "deep_native_head_joint_gradient_norm":deep_grad,
            "trainable_parameters":ntrain}


def summarize_probe(records):
    check(len(records)==12 and
          {r["kind"] for r in records}==set(KINDS)
          and all(r["use"]=="SOURCE_PROBE" and r["pair_index"]==2
                  for r in records),
          "strict predeclared 6 pairs of never-trained source-only tests")
    scores=[r["stats"] for r in records]
    return {
        "count_source_only_probe_prompts":12,
        "mean_distinguishing_native_BPE_CE":
            sum(x["native_distinguishing_BPE_CE"] for x in scores)/len(scores),
        "mean_truth_minus_opposing_logit":
            sum(x["native_distinguishing_truth_minus_counterfactual_logit"]
                for x in scores)/len(scores),
        "correct_first_token_ranks_over_opposing":
            sum(x["correct_ranked_over_opposing"] for x in scores),
        "greedy_distinguishing_BPE_exact_matches":
            sum(x["native_distinguishing_BPE_greedy_correct"] for x in scores),
        "source_counterfactual_pairs_both_sides_correct":sum(
            all(z["stats"]["correct_ranked_over_opposing"] for z in records
                if z["kind"]==kind)
            for kind in KINDS),
        "six_pair_per_domain_details":[{
            "kind":kind,
            "both_sides_correct":all(
                z["stats"]["correct_ranked_over_opposing"]
                for z in records if z["kind"]==kind),
            "correct_sides":sum(
                z["stats"]["correct_ranked_over_opposing"]
                for z in records if z["kind"]==kind),
        } for kind in KINDS],
        "same_author_source_ONLY_not_independent_quality":True,
    }


def run(s93:Path,side:Path,s142:Path,parent:Path,
        parent_report:Path,q2a_source_dir:Path,out:Path):
    check(not out.exists(),"never overwrite old original research models")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source_rows,source_manifest=verify_q2a_artifact(q2a_source_dir)
    train_raw,probe_raw,src=split_source(authored_rows())
    check(src["TRAIN_rows_n"]==24 and src["SOURCE_PROBE_rows_n"]==12
          and src["six_domains"]==list(KINDS)
          and src["no_original_S118_B_GOLD_EVAL128_J7"] is True
          and src["no_Jessica_chats_or_external_corpus"] is True
          and not {r["prompt"] for r in train_raw+probe_raw}&
                  {r["prompt"] for r in source_rows},
          "new distinct original authored counterfactual source and old SHA")
    _,ctx,old53751=load_parent(s93,side,s142,parent,parent_report)
    local=copy.deepcopy(ctx)
    check(all(torch.equal(a,b) for a,b in zip(
        ctx.state_dict().values(),local.state_dict().values())),
        "ablation must start from exact SAME full S119m pretrained child")
    original,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE,"full native original 49152 BPE only")
    train=prepared_source(tok,train_raw)
    probe=prepared_source(tok,probe_raw)
    check(len(train)==24 and len(probe)==12 and
          {r["prompt_sha256"] for r in train}.isdisjoint(
          r["prompt_sha256"] for r in probe),
          "no source-only probe prompt observed in gradient training")
    original.model.add_module("nra3_s119m_onpath",ctx)
    original.eval()
    digest=old_original_param_digest(original)
    check(all(not p.requires_grad for n,p in original.named_parameters()
              if "nra3_s119m_onpath" not in n)
          and ctx.output_gate==local.output_gate==0
          and not ctx.candidate_accepted and not local.candidate_accepted,
          "physical original Core12+GT and old neural areas frozen")
    # Predeclare original unchanged reference to compare absolute scoring, not
    # used to tune hyperparameters; 12 additional physical CPU decoder calls.
    baseline=[]
    for ex in probe:
        orig,early_unused=first_original_pass(
            original,sources,bridge,full,expansion,
            ex["prefix"],ex["prompt_features"])
        metrics=paired_instant_metrics(
            orig,ex["native_target"],ex["opposing_native_distinct_token"])
        baseline.append({k:ex[k] for k in (
            "kind","use","pair_index","side","source_id")}|{"stats":metrics})
        del early_unused
    score_original=summarize_probe(baseline)
    ctx_result=train_variant(
        original,sources,bridge,full,expansion,ctx,train,contextual=True)
    local_result=train_variant(
        original,sources,bridge,full,expansion,local,train,contextual=False)
    check(ctx_result["actual_optimizer_steps"]==
          local_result["actual_optimizer_steps"]==24
          and ctx_result["trainable_parameters"]==
          local_result["trainable_parameters"],
          "predeclared equal trainable size optimizer steps and original source")
    contextual_probe=[]
    matched_local_probe=[]
    for ex in probe:
        for contextual,adapter,results in (
            (True,ctx,contextual_probe),
            (False,local,matched_local_probe)
        ):
            with torch.enable_grad():
                loss,stats=paired_native_loss(
                    original,sources,bridge,full,expansion,adapter,ex,
                    contextual=contextual)
            # Probe is read-only; no gradient update or auto-selection.
            results.append({k:ex[k] for k in (
                "kind","use","pair_index","side","source_id")}|{"stats":stats})
            del loss
    score_ctx=summarize_probe(contextual_probe)
    score_local=summarize_probe(matched_local_probe)
    check(old_original_param_digest(original)==digest
          and ctx.output_gate==local.output_gate==0
          and not ctx.candidate_accepted and not local.candidate_accepted,
          "no original Core/GT weights change, no model activation")
    prefix="model.nra3_calibri_joint.s119m_onpath."
    for k,old in old53751.items():
        short=k.removeprefix(prefix)
        check(torch.equal(ctx.state_dict()[short].detach().half().cpu(),old)
              and torch.equal(local.state_dict()[short].detach().half().cpu(),old),
              "original 53751 S115 neural FP16 preserved in both A/B")
    out.mkdir(parents=True)
    ckpt_ctx=persist_half(ctx,out/CONTEXT_CKPT)
    ckpt_local=persist_half(local,out/LOCAL_CKPT)
    result={
        "stage":STAGE,
        "original_core_S93_SHA256":sha(s93),
        "original_S142_43M_GT_SHA256":sha(s142),
        "S119m_original_trained_full_NRA3_NPA1_SHA256":sha(parent),
        "q2a_original_source_manifest_SHA256":
            source_manifest["synthetic_rows_SHA256"],
        "q2c1_new_authored_counterfactual_source_SHA256":src["source_sha256"],
        "q2c1_source_provenance":src,
        "original_full_native_vocabulary":VOCAB_SIZE,
        "original_12layer_onepass_early_h0_to_h5_state_only":True,
        "original_later_decoder_layers_6_11_plus_GreenTunnel_executed":True,
        "one_original_decoder_forward_per_candidate_TRAIN_or_PROBE":True,
        "one_old_original_reference_forward_per_probe":True,
        "no_virtual_later_state_claim":True,
        "first_distinguishing_answer_BPE_teacher_forced_ONLY_not_multitoken_full_answer":True,
        "predefined_contrast_weight":CONTRAST_WEIGHT,
        "predefined_one_epoch_learning_rate":LR,
        "exact_same_initial_child_weights_and_param_count":True,
        "train_24_rows_disjoint_probe_12_rows":True,
        "source_probes_not_independent_heldout_or_blind":True,
        "teacher_forced_common_answer_native_BPE_prefix_ONLY_before_target":True,
        "training_targets_first_distinct_native_BPE_not_first_answer_BPE":True,
        "source_probe_shared_answer_prefix_token_counts":[
            r["teacher_forced_shared_answer_prefix_tokens"] for r in probe],
        "source_train_shared_answer_prefix_token_counts":[
            r["teacher_forced_shared_answer_prefix_tokens"] for r in train],
        "source_probe_never_used_to_select_hyperparameters":True,
        "old_q2a_calibration_never_in_optimizer_or_scores":True,
        "q2b_old_six_sources_never_in_optimizer_or_scores":True,
        "S119n_spent_six_requests_never_in_optimizer_or_scores":True,
        "original_core_GT_weights_and_53751_S115_FP16_unchanged":True,
        "original_untrained_baseline_source_probe":score_original,
        "contextual_train":ctx_result,
        "matched_param_local_train":local_result,
        "contextual_source_probe":score_ctx,
        "matched_param_local_source_probe":score_local,
        "contextual_minus_local_source_probe_CE":
            score_ctx["mean_distinguishing_native_BPE_CE"]-
            score_local["mean_distinguishing_native_BPE_CE"],
        "contextual_better_on_source_probe_only":
            score_ctx["mean_distinguishing_native_BPE_CE"]<
            score_local["mean_distinguishing_native_BPE_CE"],
        "contextual_source_only_pair_discrimination_minus_local":
            score_ctx["source_counterfactual_pairs_both_sides_correct"]-
            score_local["source_counterfactual_pairs_both_sides_correct"],
        "contextual_checkpoint_FP16_research_only":ckpt_ctx,
        "local_checkpoint_FP16_research_only":ckpt_local,
        "total_original_CPU_decoder_forward_calls":12+24+24+12+12,
        "no_Jessica_CHAT_APK_LIVE_Render_Supabase_main_merge":True,
        "no_S118B_GOLD_old_J7_or_EVAL128_loaded":True,
        "no_original_full616_J7_EOS_replay":True,
        "no_independent_qa_or_full_answer_quality_claim":True,
        "no_physical_all_in_bpp_cert":True,
        "no_matched_KV_cache_CPU_speed_cert":True,
        "output_gate":0,"candidate_accepted":False,
        "decision":"COUNTERFACTUAL_FIRST_NATIVE_BPE_RESEARCH_TRAIN_ONLY_NOT_ACCEPTED",
    }
    (out/REPORT).write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2C1_TRUE_ORIGINAL43M_GT_PAIRED_NATIVE "+json.dumps({
        "real_CPU_forwards":result["total_original_CPU_decoder_forward_calls"],
        "teacher_forced_shared_answer_prefix_max_tokens":max(
            r["teacher_forced_shared_answer_prefix_tokens"]
            for r in train+probe),
        "train_steps_each":24,"new_source_24train_12probe":True,
        "contextual_source_probe_CE":score_ctx["mean_distinguishing_native_BPE_CE"],
        "local_source_probe_CE":score_local["mean_distinguishing_native_BPE_CE"],
        "baseline_source_probe_CE":score_original["mean_distinguishing_native_BPE_CE"],
        "contextual_complete_pairs":
            score_ctx["source_counterfactual_pairs_both_sides_correct"],
        "local_complete_pairs":
            score_local["source_counterfactual_pairs_both_sides_correct"],
        "baseline_complete_pairs":
            score_original["source_counterfactual_pairs_both_sides_correct"],
        "contextual_wins_source_probe_ONLY":
            result["contextual_better_on_source_probe_only"],
        "old_neural_and_core_unchanged":True,
        "accepted":False,
    },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for k in ("s93","side","s142","parent","parent_report",
              "q2a_source_dir","out"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.parent,a.parent_report,
        a.q2a_source_dir,a.out)


if __name__=="__main__":
    main()
