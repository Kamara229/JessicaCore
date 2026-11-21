"""S119n: actually GENERATE with trained S119m on-path Calibri-43M+GT.

S119m proved a training gradient through ORIGINAL frozen decoder layers6..11
and 49,152 native BPE output, but only measured two teacher-forced samples.
S119n generates six newly authored unlabeled question-style prompts using
TWO genuinely independent autoregressive token-feedback paths: original
S142 CB59/GT and trained S119m neuronally-structured on-path intervention.

Each original next token comes from an actual physical CPU model.forward.
On-path trial: the SAME prefix is first processed by physical ORIGINAL
12-layer decoder, then trained NRA3-R+NPA1+context+core growth constructs
an H320 residual inserted AFTER original layer5 BEFORE layers6..11 using
another actual physical original CPU model.forward. The selected trial
native token feeds into its OWN next original prefix in the next iteration.
No target text, no restricted 162-class prototype and no fake tiny mock.
TRUE S142 49,152-class native LM head is always used. The original S93
S95 prompt route stays fixed from the user's initial question for BOTH
original and on-path streams, avoiding future leakage.

HARD LIMITS: this is costly TWO-pass original prefix recompute every
generated token, not an acceptable production CPU design or an efficient
new one-pass representation mechanism. Original 43M+GT remains frozen.
No claim of comparable trained LLM capability, language improvement or
correct answers based on non-independently-judged six prompts. Original
full S101/J7 controller not replayed; no 616/EOS cert, all-in bpp or
paired CPU throughput. The S119m old learned 53,751 FP16 tensors and
all original S142 pretrained parameters must remain bit-identical.
All candidate activation gates remain 0; this ON-PATH intervention runs
ONLY inside a temporary test hook in isolated research Python.
No GOLD from sealed S118/SynTagRus B; prior TRAIN-A SynTagRus license
CC BY-NC-SA 4.0 noncommercial. No Jessica chat tests (HTTP429), LIVE,
main, Render, Supabase, deployment or merge.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    LearnedOnPathInterlayer,research_only_midlayer_injection,
    old_original_param_digest, STAGE as PARENT_STAGE,
    CHECKPOINT as PARENT_CHECKPOINT,REPORT as PARENT_REPORT,
)
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    NativeFullHeadShadow,capture_all12,load_actual_original,sha,S142_SHA)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    strict_chat_source_ids)
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear

STAGE="S119N_REAL_S142_43M_GT_S119M_ONPATH_AUTOREGRESSIVE_SIXPROMPTS"
REPORT="S119N_REAL43M_GT_NATIVE_ONPATH_GENERATED_SIX_SOURCE_PROMPTS.json"
# These are only example inputs, not hidden GOLD, training set or grades.
PROMPTS=(
 ("arithmetic","Сколько будет 4 плюс 3? Ответь одним числом."),
 ("event_word_order","Книгу Лене передал Олег. Кто получил книгу?"),
 ("negation","Миша не давал Саше книгу. Получил ли Саша книгу от Миши?"),
 ("comparison","Петя выше Коли, а Коля выше Маши. Кто самый высокий?"),
 ("time","Сначала Анна вышла из дома, затем приехал автобус. Что произошло раньше?"),
 ("paraphrase","Как иначе сказать: «Миша купил велосипед»?"),
)
MAX_NEW=8
MAX_PREFIX=160
SEED=119127
INHERITED_PREFIX="model.nra3_calibri_joint.s119m_onpath."


def load_parent(s93,side,s142,checkpoint,parent_report):
    check(s93.name=="S93_UNACCEPTED_SELECTED_CB61.s61"
          and side.name=="S93_UNACCEPTED_SELECTED_JOINT.safetensors"
          and s142.name=="S142_ACTUAL_43M_ONE_J7_FP16.safetensors"
          and checkpoint.name==PARENT_CHECKPOINT
          and parent_report.name==PARENT_REPORT,
          "real original CB61 S142 plus physical full S119m onpath only")
    parent=json.loads(parent_report.read_text(encoding="utf8"))
    check(parent["stage"]==PARENT_STAGE
          and parent["physical_total_one_child_sha256"]==sha(checkpoint)
          and parent["original_43M_cb61_s93_sha256"]==sha(s93)
          and parent["original_s142_43m_gt_sha256"]==sha(s142)==S142_SHA
          and parent["original_pretrained_43M_and_GT_all_weights_sha_unchanged"]
          and parent["original_pretrained_S115_NRA3_53751_FP16_unchanged"]
          and parent["original_12layer_onpath_intervention_after_layer"]==5
          and parent["candidate_accepted"] is False
          and parent["output_gate"]==0,
          "S119m authentic tiny TRAIN-only onpath neural child provenance")
    fp16=load_file(str(checkpoint),device="cpu")
    check(fp16 and all(k.startswith(INHERITED_PREFIX)
                       and v.dtype==torch.float16
                       for k,v in fp16.items()),
          "actual S119m physical FP16 tensors only")
    old={k[len(INHERITED_PREFIX):]:v.float() for k,v in fp16.items()}
    adapter=LearnedOnPathInterlayer(NativeFullHeadShadow()).eval()
    check(set(old)==set(adapter.state_dict()),
          "cannot drop any old NPA1, NRA3/mini, rank4 or structured weights")
    adapter.load_state_dict(old,strict=True)
    check(sum(p.numel() for p in
        adapter.shadow.unified.nra3_neural_weights.parameters())==53751
          and adapter.output_gate==0 and not adapter.candidate_accepted
          and not adapter.activation_allowed
          and bool(torch.isfinite(adapter.midlayer_gates).all()),
          "old original learned 53,751 and unpromoted new S119m gates")
    original_53751={
        k:v.clone() for k,v in fp16.items()
        if ".nra3_neural_weights." in k
    }
    check(original_53751,"physical FP16 old neural area missing")
    return parent,adapter,original_53751


def prompt_features(tok,question):
    route=torch.as_tensor(
        s160.prompt_only_vector(tok,question),dtype=torch.float32)
    check(route.shape==(512,) and bool(torch.isfinite(route).all()),
          "same causal user prompt route vector for both streams")
    return route


def first_original_pass(model,sources,bridge,full,expansion,
                        prefix,features):
    """ONE *real* original full 12-layer native decode and hidden capture.

    The saved h12 are out-of-inference-mode clones for the second REAL
    decoder pass. No oracle, no next token source hidden, and no KV reuse.
    """
    check(2<=len(prefix)<MAX_PREFIX and features.shape==(512,),
          "strict current causal autoregressive native prefix only")
    ids=torch.tensor([prefix],dtype=torch.long)
    with torch.inference_mode(),expansion.prefill_only(model), \
         s160.prefill_only_tunnel(model,full):
        with s160.frozen_cpu_phase_scope(
            model,bridge,"S93",features,phase="prefill_only"):
            with capture_all12(model) as frame:
                clear(sources)
                out=model(input_ids=ids,past_key_values=None,
                          use_cache=True,output_hidden_states=True)
                check(out.past_key_values is not None and
                      out.hidden_states[-1].shape==(1,len(prefix),320)
                      and out.logits.shape[:2]==ids.shape
                      and out.logits.shape[-1]>1000,
                      "original physical S142 43M GT full 12layer native head")
                base=out.logits[:,-1,:].detach().float().cpu().clone()
                normalized=out.hidden_states[-1][:,-1].detach().float().clone()
                check(torch.allclose(
                    base,model.lm_head(normalized).float(),
                    atol=2e-3,rtol=1e-4),
                    "exact original native LM head and model forward match")
                clear(sources)
                del out
    original12=torch.stack([frame[i] for i in range(12)],dim=1)
    result=original12.detach().float().clone()
    check(result.shape==(1,12,len(prefix),320)
          and bool(torch.isfinite(result).all()),
          "physical original 12 layer memory snapshots aligned to prefix")
    return base,result


def actual_original_onpath_second_pass(model,sources,bridge,full,expansion,
                                        adapter,prefix,features,original12):
    """NEW trained S119m inserted after layer5, before original layers6..11.

    Midlayer hook is temporary. This is NOT original registered decoder
    forward behavior; it is isolated research ON-PATH causal evaluation.
    """
    check(2<=len(prefix)<MAX_PREFIX and
          original12.shape==(1,12,len(prefix),320)
          and adapter.output_gate==0 and not adapter.candidate_accepted,
          "real first-pass same-prefix source required before true second pass")
    ids=torch.tensor([prefix],dtype=torch.long)
    with torch.enable_grad():
        delta=adapter(original12,contextual=True)
        with expansion.prefill_only(model), \
             s160.prefill_only_tunnel(model,full), \
             bridge.prompt_scope("S93",features), \
             research_only_midlayer_injection(model,adapter,delta):
            clear(sources)
            out=model(input_ids=ids,past_key_values=None,
                      use_cache=True,output_hidden_states=False)
            check(out.past_key_values is not None
                  and out.logits.shape[:2]==ids.shape
                  and out.logits.shape[-1]>1000,
                  "ACTUAL original later layers and original native full LM head")
            logits=out.logits[:,-1].detach().float().cpu().clone()
            clear(sources)
            del out
    check(not logits.requires_grad and bool(torch.isfinite(logits).all()),
          "detach research generation native full-vocabulary logits")
    return logits


def verify_zero_gate_exact_onpath(model,sources,bridge,full,expansion,
                                   adapter,prefix,features,base,h12):
    """No fine-tuning effects allowed when new middle gates are ZERO."""
    saved=adapter.midlayer_gates.detach().clone()
    try:
        with torch.no_grad():
            adapter.midlayer_gates.zero_()
        zero=actual_original_onpath_second_pass(
            model,sources,bridge,full,expansion,adapter,
            prefix,features,h12)
        check(zero.shape==base.shape and
              torch.allclose(zero,base,rtol=1e-5,atol=2e-4)
              and torch.equal(zero.argmax(-1),base.argmax(-1)),
              "S119n strict ZERO-gate original FULL native LM parity")
    finally:
        with torch.no_grad():adapter.midlayer_gates.copy_(saved)


def stream(tok,model,sources,bridge,full,expansion,
           adapter,question,*,trained_onpath:bool):
    """Native BPE argmax FEEDBACK: every choice enters own next model.forward."""
    check(isinstance(trained_onpath,bool),"two explicit original vs trained streams")
    source_ids=strict_chat_source_ids(tok,question)
    check(2<=len(source_ids)<MAX_PREFIX,
          "native original chat tokenizer prompt only")
    route=prompt_features(tok,question)
    ids=list(source_ids)
    emitted=[];trace=[]
    start=time.perf_counter()
    zero_checked=False
    for step in range(MAX_NEW):
        base,h12=first_original_pass(
            model,sources,bridge,full,expansion,ids,route)
        if trained_onpath:
            if not zero_checked:
                verify_zero_gate_exact_onpath(
                    model,sources,bridge,full,expansion,
                    adapter,ids,route,base,h12)
                zero_checked=True
            candidate=actual_original_onpath_second_pass(
                model,sources,bridge,full,expansion,
                adapter,ids,route,h12)
        else:
            candidate=base
        source_best=int(base.argmax(-1))
        onpath_best=int(candidate.argmax(-1))
        selected=onpath_best if trained_onpath else source_best
        check(0<=selected<tok.vocab_size
              and base.shape==candidate.shape
              and base.shape[-1]==tok.vocab_size,
              "only true full original native vocabulary token output")
        top2=torch.topk(base[0],k=2).values
        margin=float((top2[0]-top2[1]).item())
        max_logit_change=float((candidate-base).abs().max().item())
        trace.append({
            "step":step+1,
            "actual_native_current_prefix_length":len(ids),
            "original_native_next_token_id":source_best,
            "actual_trained_onpath_native_next_token_id":onpath_best,
            "selected_token_id":selected,
            "original_top1_top2_logit_margin":margin,
            "actual_onpath_max_abs_logit_change":max_logit_change,
        })
        emitted.append(selected)
        ids.append(selected)
        if selected==tok.eos_token_id:break
        check(len(ids)<MAX_PREFIX,
              "no causal native prompt prefix overflow")
    check(len(trace)==len(emitted)
          and all(trace[i]["actual_native_current_prefix_length"]==
                  len(source_ids)+i for i in range(len(trace))),
          "every generated native BPE token must reenter NEXT original decoder call")
    return {
        "native_original_question_BPE_count":len(source_ids),
        "generated_native_token_ids":emitted,
        "generated_text":tok.decode(emitted,skip_special_tokens=True),
        "emitted_EOS":bool(emitted[-1]==tok.eos_token_id),
        "original_full12layer_prefill_calls":len(trace),
        "real_onpath_second_decoder_calls":len(trace) if trained_onpath else 0,
        "additional_strict_zero_gate_control_calls":int(trained_onpath),
        "actual_native_steps":trace,
        "elapsed_seconds_recompute_full_prefix_NOT_CPU_throughput":
            time.perf_counter()-start,
    }


def run(s93,side,s142,parent,parent_report,out):
    check(not out.exists(),"S119n no mutation or overwrites")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    evidence,adapter,old_fp16=load_parent(s93,side,s142,parent,parent_report)
    model,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==49152,
          "one real full original tokenizer vocabulary, no restricted proxies")
    original_digest=old_original_param_digest(model)
    model.model.add_module("nra3_s119m_onpath",adapter)
    model.eval()
    check(old_original_param_digest(model)==original_digest
          and len(model.model.layers)==12
          and adapter.output_gate==0 and not adapter.candidate_accepted,
          "actual original Calibri 43M parameters remain frozen")
    records=[]
    for group,prompt in PROMPTS:
        base=stream(tok,model,sources,bridge,full,expansion,
                    adapter,prompt,trained_onpath=False)
        onpath=stream(tok,model,sources,bridge,full,expansion,
                      adapter,prompt,trained_onpath=True)
        records.append({
            "category":group,
            "initial_question_SHA256":
                hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "source_question":prompt,
            "original_native_greedy":base,
            "actual_deep_onpath_native_greedy":onpath,
            "different_native_generated_token_sequences":
                base["generated_native_token_ids"]!=
                onpath["generated_native_token_ids"],
            "answer_grade_NOT_MEASURED":True,
        })
    check(len(records)==len(PROMPTS)
          and old_original_param_digest(model)==original_digest,
          "original 43M+GT full trained parameters cannot be changed")
    for key,v in old_fp16.items():
        name=key[len(INHERITED_PREFIX):]
        check(torch.equal(adapter.state_dict()[name].detach().half().cpu(),v),
              "all S115 53751 original trained FP16 weights byte identical")
    check(adapter.output_gate==0 and not adapter.candidate_accepted
          and not adapter.activation_allowed
          and not model.model.layers[5]._forward_hooks
          .get(-1,False), "do not leave research onpath hooks after decode")
    out.mkdir(parents=True)
    result={
        "stage":STAGE,
        "S119m_trained_complete_physical_FP16_SHA256":sha(parent),
        "original_S93_CB61_SHA256":sha(s93),
        "original_S142_43M_GT_SHA256":sha(s142),
        "physical_full_original_native_vocab_size":tok.vocab_size,
        "genuine_original_decoder_forward_called_for_each_greedy_native_token":True,
        "genuine_after_layer5_onpath_through_original_layers6_to11":True,
        "each_independent_candidate_generated_native_token_is_fed_back":True,
        "prompt_route_S95_held_constant_through_both_streams":True,
        "zero_new_interlayer_gate_reproduces_original_full_native_logits":True,
        "n_unlabeled_author_prompts":len(records),
        "prompts_where_trained_onpath_changes_generated_native_BPE":sum(
            x["different_native_generated_token_sequences"] for x in records),
        "total_actual_original43m_forward_calls":sum(
            x[z]["original_full12layer_prefill_calls"]+
            x[z]["real_onpath_second_decoder_calls"]+
            x[z]["additional_strict_zero_gate_control_calls"]
            for x in records for z in (
                "original_native_greedy","actual_deep_onpath_native_greedy")),
        "original_pretrained_43m_GT_unchanged_digest":True,
        "all_prior_S115_53751_NRA3_FP16_exact_unchanged":True,
        "source_questions_unlabeled_not_independent_quality_scoring":True,
        "not_full_S101_J7_answer_controller":True,
        "not_one_pass_model_extra_original_decoder_cost":True,
        "normal_KV_streaming_CPU_NOT_MEASURED":True,
        "all_in_bpp_NOT_measured":True,
        "original_616_J7_EOS_NOT_replayed":True,
        "not_end_to_end_updating_original_base_43m":True,
        "no_heldout_QA_gold_answers":True,
        "sealed_S118_SynTagRusB_GOLD_not_accessed":True,
        "CC_BY_NC_SA_4_prior_training_noncommercial_only":True,
        "no_live_activation":True,
        "output_gate":0,
        "candidate_accepted":False,
        "research_records":records,
        "decision":"RESEARCH_ONLY_GENUINE_DEEP_ONPATH_GREEDY_SIX_UNLABELED_NO_QUALITY_CLAIMS",
    }
    (out/REPORT).write_text(json.dumps(
        result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119N_REAL_S142_GT_S119M_ONPATH_GENERATED "+
          json.dumps({
            "n_user_prompts":len(records),
            "total_real_decoder_forwards":
                result["total_actual_original43m_forward_calls"],
            "different_native_sequences":
                result["prompts_where_trained_onpath_changes_generated_native_BPE"],
            "baseline_texts":[r["original_native_greedy"]["generated_text"]
                              for r in records],
            "onpath_texts":[r["actual_deep_onpath_native_greedy"]["generated_text"]
                            for r in records],
            "original_53751_trained_exact":True,
            "genuine_independent_quality_benchmark":False,
            "accepted":False,
          },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for k in ("s93","side","s142","parent","parent_report","out"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.parent,a.parent_report,a.out)


if __name__=="__main__":
    main()
