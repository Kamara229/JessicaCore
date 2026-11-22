"""S119q2g: original native own-token autoregressive new-source audit.

A genuinely NEW authored source bank is frozen in q2g source.py, disjoint
from q2a TRAIN/CAL, q2c1 TRAIN/SOURCE_PROBE and earlier S119n prompts.
It is NOT independently blind because humans wrote and can inspect it.
Every generated native BPE comes from THAT model's own prior predicted
tokens only; not one gold answer token or teacher-forced answer BPE enters
the decoder. Deterministic greedy full original 49,152-class native
head. Up to 12 generated tokens, EOS explicitly tracked; if truncated,
never call output a complete answer. Compare physically unchanged original
S142 43M+GT against genuine trained q2f contextual/local children in the
SAME original decoder and same prompt tokenization/features.

This is NOT training, not a matched KV throughput measurement, not a
packed<=0.7bpp test and not full 616 protected J7+EOS replay. All
research output gate0; no mutation of original or learned q2f weights.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import torch

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest
)
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    first_original_pass,prompt_features,
)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    strict_chat_source_ids
)
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    onepass_full_native_logits,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    verify_q2a_artifact
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    REPORT as Q2D_REPORT,CONTEXT_CKPT as Q2D_CTX,
    LOCAL_CKPT as Q2D_LOCAL,VOCAB_SIZE,source_inventory,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    REPORT as Q2F_REPORT,CONTEXT_CKPT as Q2F_CTX,
    LOCAL_CKPT as Q2F_LOCAL,STAGE as Q2F_STAGE,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child
)
from research.calibri_nra3.s119q2g_fresh_own_token_source import (
    rows,validate,normalized,STAGE as SOURCE_STAGE,
)

STAGE="CALIBRI_S119Q2G_REAL43M_GT_FRESH_SOURCE_OWN_TOKEN_GREEDY_FULL_NATIVE"
REPORT="S119Q2G_FRESH_12_PROMPTS_REAL_NATIVE_OWN_TOKEN_GREEDY.json"
SEED=119138
MAX_NEW_NATIVE=12
SAMPLE_COUNT=12
MODELS=("frozen_original","trained_contextual","trained_token_local")
MAX_POSSIBLE_REAL_ORIGINAL_FORWARDS=SAMPLE_COUNT*len(MODELS)*MAX_NEW_NATIVE


def own_token_stream(tok,question,model,sources,bridge,full,expansion,
                     *,variant,adapter=None,max_new=MAX_NEW_NATIVE):
    """FULL physical prefill repeated, never use gold or precomputed answers.

    Purposeful NO KV cache reuse: proves semantic own-token behavior but
    runtime cannot be compared to final efficient KV-cache CPU target.
    """
    check(variant in MODELS and
          1<=max_new<=MAX_NEW_NATIVE and
          (adapter is None)==(variant=="frozen_original"),
          "one and only one of three preregistered original/model children")
    prefix=strict_chat_source_ids(tok,question)
    features=prompt_features(tok,question)
    check(2<=len(prefix)<160-max_new and
          features.shape==(512,),
          "original unlabelled question only, native context fits each own token")
    own_prefix=prefix.copy()
    emitted=[];trace=[]
    for step in range(max_new):
        check(len(own_prefix)==len(prefix)+len(emitted),
              "only decoder-selected earlier BPE may feed next prefill")
        if variant=="frozen_original":
            logits,unused_original_hidden=first_original_pass(
                model,sources,bridge,full,expansion,own_prefix,features)
            del unused_original_hidden
            witness=None
        else:
            check(adapter.output_gate==0 and
                  not adapter.candidate_accepted and
                  not adapter.activation_allowed and
                  adapter.projection_enabled,
                  "only trained unaccepted physical original Core child gate0")
            raw,witness=onepass_full_native_logits(
                model,sources,bridge,full,expansion,adapter,
                own_prefix,features,
                contextual=(variant=="trained_contextual"))
            logits=raw.detach().float().cpu()
            del raw
            check(witness["candidate_forward_calls"]==1 and
                  logits.shape==(1,VOCAB_SIZE),
                  "one original full decoder native BPE per emitted token")
        check(logits.shape==(1,VOCAB_SIZE) and
              bool(torch.isfinite(logits).all()),
              "real entire native original 49152 classes not a task proxy")
        picked=int(torch.argmax(logits,dim=-1).item())
        check(0<=picked<VOCAB_SIZE,
              "greedy next original native token must be in ORIGINAL vocab")
        top=torch.topk(logits[0],k=2).values
        trace.append({
            "step":step+1,
            "true_current_native_prefix_length":len(own_prefix),
            "selected_native_token_id":picked,
            "full_native_top1_minus_top2_margin":
                float((top[0]-top[1]).item()),
            "true_original_decoder_forward_calls":1,
            "onpath_injection_real_original_layer5_calls":
                int(variant!="frozen_original"),
        })
        emitted.append(picked)
        own_prefix.append(picked)
        del logits
        if picked==tok.eos_token_id:
            break
    check(all(trace[i]["true_current_native_prefix_length"]==
              len(prefix)+i for i in range(len(trace)))
          and all(trace[i]["selected_native_token_id"]==emitted[i]
                  for i in range(len(trace))),
          "all own generated BPE tokens fed back in exact causal order")
    ended=bool(emitted[-1]==tok.eos_token_id)
    decoded=tok.decode(emitted,skip_special_tokens=True)
    return {
        "variant":variant,
        "question_native_prefix_length":len(prefix),
        "generated_native_token_ids":emitted,
        "generated_native_decoded_text":decoded,
        "native_eos_emitted":ended,
        "generation_truncated_at_token_cap":not ended,
        "native_argmax_autoregressive_steps":trace,
        "real_original_full_decoder_forward_calls":len(emitted),
        "native_full_vocab":VOCAB_SIZE,
        "no_label_or_future_answer_token_in_decoder_prefix":True,
        "recompute_full_prefix_each_step_NO_KV_speed_claim":True,
    }


def score_full_untrained_source(data):
    check(len(data)==SAMPLE_COUNT and
          all(set(item["results"])==set(MODELS) for item in data),
          "fixed 12 original user-only fresh prompts and three original streams")
    result={}
    for variant in MODELS:
        by_kind=defaultdict(list)
        all_text=[]
        for rec in data:
            g=rec["results"][variant]
            check(g["variant"]==variant and
                  g["no_label_or_future_answer_token_in_decoder_prefix"]
                  and g["real_original_full_decoder_forward_calls"]==
                      len(g["generated_native_token_ids"]),
                  "each model self-feedback native vocab; no answer conditioning")
            strict=bool(
                normalized(g["generated_native_decoded_text"])==
                normalized(rec["expected_short_answer"]) and
                g["native_eos_emitted"])
            # When cap reached, compare visible literal text separately,
            # but DO NOT count it as a complete answer (EOS not observed).
            text_only=normalized(g["generated_native_decoded_text"])==(
                normalized(rec["expected_short_answer"]))
            by_kind[rec["kind"]].append(strict)
            all_text.append((strict,text_only,g["native_eos_emitted"]))
            g["strict_finished_answer_exact_match"]=strict
            g["visible_text_exact_match_EOS_not_required"]=text_only
        check(len(by_kind)==6 and
              all(len(v)==2 for v in by_kind.values()),
              "two brand new untrained prompts across exactly six areas")
        result[variant]={
            "exact_complete_answers_WITH_native_EOS":
                sum(x[0] for x in all_text),
            "visible_exact_text_even_if_truncated":
                sum(x[1] for x in all_text),
            "native_eos_finished_count":
                sum(x[2] for x in all_text),
            "never_finished_token_cap_count":
                sum(not x[2] for x in all_text),
            "strict_complete_answers_by_kind":{
                k:sum(v) for k,v in sorted(by_kind.items())},
            "not_blind_or_independent":True,
        }
    return result


def run(s93,side,s142,q2a_dir,q2d_dir,q2f_dir,out):
    check(not out.exists(),"do not overwrite earlier own-token diagnostics")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=rows()
    source_meta=validate(source)
    check(source_meta["stage"]==SOURCE_STAGE and
          source_meta["source_example_count"]==SAMPLE_COUNT and
          source_meta["no_old_EVAL128_S118_B_GOLD_or_J7"],
          "freeze new original-authored source BEFORE loading any model")
    old_rows,old_meta=verify_q2a_artifact(q2a_dir)
    oldtrain,oldprobe=source_inventory(old_rows)
    check(not {r["question"] for r in source}&{
        x["prompt"] for x in oldtrain+oldprobe},
        "fresh original 12 prompts not exposed to gradients OR old scores")
    old_q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,
        s93,s142)
    q2f_proof=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    check(q2f_proof["stage"]==Q2F_STAGE and
          q2f_proof["original_S93_SHA256"]==sha(s93) and
          q2f_proof["original_S142_GT_SHA256"]==sha(s142) and
          q2f_proof["parent_q2d_training_report_SHA256"]==
              sha(q2d_dir/Q2D_REPORT) and
          q2f_proof["original_q2a_synthetic_source_SHA256"]==
              old_meta["synthetic_rows_SHA256"] and
          q2f_proof["original_12layer_GreenTunnel_original_core_weights_unchanged"] and
          q2f_proof["old_S115_53751_FP16_neural_weights_unchanged"] and
          q2f_proof["actual_original_decoder_forward_calls"]==504 and
          q2f_proof["candidate_accepted"] is False and
          q2f_proof["output_gate"]==0,
          "only original true physically trained q2f contextual/local FP16")
    parents=[]
    for _ in MODELS[1:]:
        parent,_=load_q2d_trained_child(
            q2d_dir/Q2D_CTX,
            old_q2d["new_research_checkpoint_contextual"])
        parents.append(parent)
    contextual=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,q2f_proof["projector_contextual_child_checkpoint"],
        parents[0])
    local=load_physical_q2f_child(
        q2f_dir/Q2F_LOCAL,q2f_proof["projector_local_matched_checkpoint"],
        parents[1])
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE and tok.eos_token_id is not None,
          "exact original native 49152 BPE tokenizer with real EOS")
    original,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    original.model.add_module("nra3_s119m_onpath",contextual)
    original.eval()
    old_digest=old_original_param_digest(original)
    states={
        "trained_contextual":{k:v.detach().cpu().clone()
                              for k,v in contextual.state_dict().items()},
        "trained_token_local":{k:v.detach().cpu().clone()
                               for k,v in local.state_dict().items()},
    }
    data=[];total_forwards=0
    # All predictions are completed before the expected text is read to
    # evaluate. Free-running user prompt is the ONLY model input.
    for rec in source:
        generated={}
        for variant in MODELS:
            child=(contextual if variant=="trained_contextual"
                   else local if variant=="trained_token_local" else None)
            generated[variant]=own_token_stream(
                tok,rec["question"],original,sources,bridge,full,expansion,
                variant=variant,adapter=child)
            total_forwards+=generated[variant][
                "real_original_full_decoder_forward_calls"]
        data.append({
            "kind":rec["source_kind"],
            "source_id":rec["source_id"],
            "question_sha256":hashlib.sha256(
                rec["question"].encode("utf8")).hexdigest(),
            "expected_short_answer":rec["canonical_short_answer"],
            "results":generated,
        })
    scored=score_full_untrained_source(data)
    check(0<total_forwards<=MAX_POSSIBLE_REAL_ORIGINAL_FORWARDS
          and old_original_param_digest(original)==old_digest,
          "actual original Native BPE own-token streams and frozen full core")
    for label,adapter in (
        ("trained_contextual",contextual),("trained_token_local",local)
    ):
        check(adapter.output_gate==0 and
              not adapter.candidate_accepted and
              not adapter.activation_allowed and
              adapter.projection_enabled and
              all(torch.equal(v,adapter.state_dict()[k].detach().cpu())
                  for k,v in states[label].items()),
              "learned and original weights read only, no activation or retraining")
    result={
        "stage":STAGE,
        "original_S93_sha256":sha(s93),
        "original_S142_43M_GT_sha256":sha(s142),
        "original_full_native_vocab":VOCAB_SIZE,
        "q2f_physical_checkpoint_report_sha256":sha(q2f_dir/Q2F_REPORT),
        "new_source_provenance":source_meta,
        "actual_original_decoder_forward_calls":total_forwards,
        "maximum_original_decoder_calls_predeclared":
            MAX_POSSIBLE_REAL_ORIGINAL_FORWARDS,
        "max_generated_native_BPE_cap":MAX_NEW_NATIVE,
        "original_decoder_full_prefix_recomputed_no_KV_cache_speed":True,
        "models_frozen_original_and_two_physically_trained_q2f":list(MODELS),
        "native_greedy_OWN_emitted_BPE_only_no_gold_teacher_forcing":True,
        "same_question_native_tokenization_and_prompt_features_per_variant":True,
        "original_core_GT_and_old_53751_FP16_unmodified":True,
        "learned_q2f_contextual_local_projector_unchanged":True,
        "full_own_token_generation_results":scored,
        "source_only_12_new_author_questions_answers":data,
        "source_not_used_for_training_but_not_independently_blind":True,
        "no_independent_generalization_or_616J7_claim":True,
        "no_native_matched_KV_CPU_throughput_or_physical_all_in_bpp_cert":True,
        "no_S118_B_GOLD_spent_EVAL128_J7_label_or_Jessica_chat":True,
        "no_LIVE_Render_Supabase_main_APK":True,
        "no_training_or_parameter_update":True,
        "candidate_accepted":False,
        "output_gate":0,
        "decision":"NEW_OWN_TOKEN_SOURCE_GENERATION_DIAGNOSTIC_ONLY_NOT_ACCEPTED",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2G_ACTUAL_FRESH_OWN_NATIVE_GREEDY_GENERATION "+
          json.dumps({
            "actual_original_decoder_calls":total_forwards,
            "authored_new_source_cases":SAMPLE_COUNT,
            "source_SHA256":source_meta["twelve_source_sha256"],
            "native_vocabulary":VOCAB_SIZE,
            "strict_full_EOS_answers":{
                k:v["exact_complete_answers_WITH_native_EOS"]
                for k,v in scored.items()},
            "visible_exact_short_text_EOS_not_required":{
                k:v["visible_exact_text_even_if_truncated"]
                for k,v in scored.items()},
            "finished_EOS":{
                k:v["native_eos_finished_count"] for k,v in scored.items()},
            "original_weights_unchanged":True,
            "accepted":False,
          },sort_keys=True,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2a_dir","q2d_dir","q2f_dir","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    args=parser.parse_args()
    run(args.s93,args.side,args.s142,args.q2a_dir,
        args.q2d_dir,args.q2f_dir,args.out)


if __name__=="__main__":
    main()
