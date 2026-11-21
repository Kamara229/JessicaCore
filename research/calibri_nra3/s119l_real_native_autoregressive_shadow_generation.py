"""S119l: ACTUAL autoregressive token-feedback original Calibri+GT research smoke.

Transition from S119k teacher-forced native full-vocab scoring to a closed
original-decoder GENERATION LOOP. On three SHA-fixed, researcher-authored
source-only conversational prompts, rebuild REAL original trained S142 43M
compressed CB59 + original S93/12-layer Green Tunnel, call ACTUAL native
model.forward on an incrementally grown token prefix at EVERY step, and
choose tokens independently in two isolated research streams:

    baseline: actual original LM head native argmax;
    structured: full-vocab off-decoder SHADOW readout using physical S119k
        full trained old 53,751 NRA3, NPA1, core-growth and learned RS.

Every generated ID enters the next ORIGINAL Calibri full-prefix forward.
No changes to model.forward, model.lm_head, CB59, original GreenTunnel or
saved original weights; cannot activate the research child in LIVE.
The S119k shadow output_gate stays ZERO; it can only be read off-path for
this sandbox generation trace. Not a real on-path integrated trained model.
No KV cache, recompute whole full prefix each step; no production CPU claim.
No ground-truth labels used or imported for the source questions; outputs
are NOT independently judged answers and no quality gain is proven.
No 616 J7 replay, no all-in bpp, no true standalone complete evaluation.
S119k ancestor TRAIN-A SynTagRus is CC BY-NC-SA 4.0 noncommercial.
S118-B GOLD, spent QA/EVAL, Jessica ordinary chat, Render, Supabase,
LIVE, main, deploy/merge remain forbidden.
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
from research.calibri_nra3.s119g_unified_planning_internal_action_neural_area import PREFIX
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    NativeFullHeadShadow, capture_all12, load_actual_original,
    original_parameter_hash, sha,
    STAGE as PARENT_STAGE, CHECKPOINT as PARENT_CHECKPOINT,
    REPORT as PARENT_REPORT, S142_SHA,
)
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear

STAGE = "S119L_TRUE_ORIGINAL43M_GT_FULL_NATIVE_AUTOREGRESSIVE_SHADOW_TWO_STREAMS"
REPORT = "S119L_ORIGINAL43M_GT_REAL_GENERATED_NATIVE_BPE_SHADOW_GOLD_BLIND.json"
# Precommitted source-only snippets, no corresponding answers or evaluation
# labels; do NOT call these independently evaluated quality questions.
PROMPTS = (
    ("arithmetic", "Чему равно четыре плюс три? Ответь кратко."),
    ("event", "Олег передал журнал Лене. Кому достался журнал?"),
    ("negation", "Миша не дал Саше книгу. Получил ли Саша книгу от Миши?"),
)
MAX_NEW = 5
MAX_PREFIX = 160
SEED = 119125


def prompt_id_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_parent(s93, side, s142, checkpoint, report):
    check(checkpoint.name==PARENT_CHECKPOINT and
          report.name==PARENT_REPORT and
          s93.name=="S93_UNACCEPTED_SELECTED_CB61.s61" and
          side.name=="S93_UNACCEPTED_SELECTED_JOINT.safetensors" and
          s142.name=="S142_ACTUAL_43M_ONE_J7_FP16.safetensors",
          "only verified original S119k / real original S93 S142 ancestors")
    data=json.loads(report.read_text(encoding="utf8"))
    check(data["stage"]==PARENT_STAGE and
          data["physical_unified_shadow_SHA256"]==sha(checkpoint) and
          data["real_original_s93_cb61_SHA256"]==sha(s93) and
          data["real_original_S93_side_sha256"]==sha(side) and
          data["original_S142_43M_full_FP16_SHA256"]==sha(s142)==S142_SHA and
          data["real_original12_frozen_GT_and_S142_core"] and
          data["REAL_original_full_native_LM_head_used"] and
          data["all_pretrained_old_NRA3_53751_FP16_exact"] and
          data["candidate_accepted"] is False and
          data["output_gate"]==0,
          "parent physical real native-head source provenance fail closed")
    old=load_file(str(checkpoint),device="cpu")
    key=PREFIX+"s119k_shadow."
    check(all(k.startswith(key) and v.dtype==torch.float16
              for k,v in old.items()),
          "only full original previously saved FP16 shadow")
    params={k[len(key):]:v.float() for k,v in old.items()}
    shadow=NativeFullHeadShadow().eval()
    check(set(params)==set(shadow.state_dict()),
          "full S119k shadow must include original learned NRA3+NPA1+RS")
    shadow.load_state_dict(params,strict=True)
    check(sum(p.numel() for p in shadow.unified.nra3_neural_weights.parameters())
          ==53751 and shadow.output_gate==0 and
          not shadow.candidate_accepted and not shadow.activation_allowed,
          "full original 53751 weights preserved, shadow NOT live")
    return data,old,shadow


def strict_chat_source_ids(tok,question):
    """True original tokenizer + training chat template, no answer lookup."""
    ids=list(s160.prompt_tokens(tok,question))
    check(2<=len(ids)<MAX_PREFIX and all(isinstance(x,int) and
          0<=x<tok.vocab_size for x in ids),
          "native-original user chat prompt tokens only")
    return ids


def true_fullprefix_step(model,sources,bridge,full,expansion,
                         fixed_prompt_vector,prefix_ids,shadow):
    """Real CPU forward for *every* prefix, no KV or incremental shortcuts.

    S95 route is frozen from *user's initial question* across answer
    generation, no future target or generated answer text affects routing.
    Internal GT streaming equivalence is NOT assumed.
    """
    check(len(prefix_ids)>=2 and len(prefix_ids)<=MAX_PREFIX
          and shadow.output_gate==0 and not shadow.candidate_accepted
          and fixed_prompt_vector.shape==(512,),
          "original full-prefix-only research decode contract")
    ids=torch.tensor([prefix_ids],dtype=torch.long)
    with torch.inference_mode(),expansion.prefill_only(model), \
         s160.prefill_only_tunnel(model,full):
        with s160.frozen_cpu_phase_scope(
            model,bridge,"S93",fixed_prompt_vector,phase="prefill_only"):
            with capture_all12(model) as hooks:
                clear(sources)
                native=model(input_ids=ids,past_key_values=None,
                             use_cache=True,output_hidden_states=True)
                check(native.past_key_values is not None and
                      native.hidden_states[-1].shape==(1,len(prefix_ids),320),
                      "genuine original S142+GT all 12 forward required")
                original=native.logits[:,-1,:].detach().float().cpu().clone()
                last=native.hidden_states[-1][:,-1].detach().float().cpu().clone()
                manual=model.lm_head(last).detach().float().clone()
                check(torch.allclose(original,manual,rtol=1e-4,atol=2e-3),
                      "full original LM logits must equal manual true native head")
                clear(sources)
                del native
    hidden=torch.stack([hooks[k] for k in range(12)],dim=1).float().clone()
    with torch.no_grad():
        candidate=shadow(hidden,last,model.lm_head,contextual=True).float()
    check(candidate.shape==original.shape and candidate.shape[-1]>1000
          and bool(torch.isfinite(candidate).all()),
          "original full vocabulary no restricted lexicon")
    return original,candidate


def autoregressive_stream(tok,model,sources,bridge,full,expansion,
                          shadow,question,*,candidate):
    """Own feedback loop: predictions *actually determine next model input*."""
    check(isinstance(candidate,bool),"two explicit independent streams")
    prompt=strict_chat_source_ids(tok,question)
    full_ids=list(prompt)
    features=torch.as_tensor(
        s160.prompt_only_vector(tok,question),dtype=torch.float32)
    check(features.shape==(512,) and bool(torch.isfinite(features).all()),
          "frozen initial prompt route, never generated answer features")
    emitted=[]
    first_gap=None
    steps=[]
    start=time.perf_counter()
    for i in range(MAX_NEW):
        orig,side=true_fullprefix_step(
            model,sources,bridge,full,expansion,features,full_ids,shadow)
        base=int(orig.argmax(-1).item())
        trial=int(side.argmax(-1).item())
        selected=trial if candidate else base
        check(0<=selected<tok.vocab_size,
              "REAL original native BPE token selection")
        if first_gap is None and base!=trial:
            first_gap=i
        emitted.append(selected)
        steps.append({"step":i+1,
                      "native_input_prefix_len":len(full_ids),
                      "actual_original_next_id":base,
                      "research_shadow_next_id":trial,
                      "selected_native_original_vocabulary_id":selected})
        full_ids.append(selected)
        if selected==tok.eos_token_id:break
        check(len(full_ids)<MAX_PREFIX,
              "autoregressive source prefix context hard limit")
    check(steps and len(emitted)==len(steps)
          and all(steps[i]["native_input_prefix_len"]==len(prompt)+i
                  for i in range(len(steps))),
          "every selected candidate token MUST reenter next original forward")
    return {
        "new_generated_token_ids":emitted,
        "generated_text":tok.decode(emitted,skip_special_tokens=True),
        "eos_generated":bool(emitted[-1]==tok.eos_token_id),
        "n_forward_actual_original43m_calls":len(steps),
        "first_original_shadow_greedy_disagreement_step":first_gap,
        "steps":steps,
        "elapsed_CPU_seconds_fullprefix_recompute_NOT_speed_benchmark":
            time.perf_counter()-start,
    }


def run(s93,side,s142,parent,parent_report,out):
    check(not out.exists(),"must create new research-only results, no writes")
    torch.manual_seed(SEED);torch.set_num_threads(2)
    evidence,old,shadow=parse_parent(s93,side,s142,parent,parent_report)
    base,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==evidence["native_REAL_full_original_vocabulary_size"],
          "exact genuine old 49152-class native tokenizer and head")
    frozen_hash=original_parameter_hash(base)
    # Inert registration inside actual Calibri nn.Module for state
    # persistence. Its outputs NEVER alter original model.forward().
    base.model.add_module("nra3_s119k_shadow",shadow)
    base.eval().requires_grad_(False)
    check(base.model.nra3_s119k_shadow is shadow
          and original_parameter_hash(base)==frozen_hash
          and not shadow.candidate_accepted
          and shadow.output_gate==0,
          "original research model unchanged and candidate cannot be promoted")
    saved_old={
        k:v.clone() for k,v in old.items()
        if ".nra3_neural_weights." in k}
    prompt_results=[]
    for kind,question in PROMPTS:
        # Actual TWO independent original32/GT prefix streams; no hidden
        # state re-use, no teacher forcing, no answers supplied to runtime.
        original=autoregressive_stream(
            tok,base,sources,bridge,full,expansion,
            shadow,question,candidate=False)
        trial=autoregressive_stream(
            tok,base,sources,bridge,full,expansion,
            shadow,question,candidate=True)
        prompt_results.append({
            "category":kind,"initial_question_sha256":prompt_id_digest(question),
            "initial_question":question,
            "original_native_greedy":original,
            "offpath_trained_structured_native_greedy":trial,
            "different_generated_native_sequences":(
                original["new_generated_token_ids"]!=
                trial["new_generated_token_ids"]),
            "source_answers_not_used_for_inference_or_scoring":True,
        })
    check(original_parameter_hash(base)==frozen_hash
          and shadow.output_gate==0 and
          not shadow.candidate_accepted and
          all(torch.equal(
              shadow.state_dict()[k[len(PREFIX+"s119k_shadow."):]]
                  .detach().half().cpu(),v)
              for k,v in saved_old.items()),
          "old CB59/GT/S142 43M and 53751 original NRA3 weights unchanged")
    check(len(prompt_results)==3 and
          all(r["original_native_greedy"][
               "n_forward_actual_original43m_calls"]>0
              and r["offpath_trained_structured_native_greedy"][
               "n_forward_actual_original43m_calls"]>0
              for r in prompt_results),
          "must run both actual original frozen models in autoregressive loops")
    out.mkdir(parents=True)
    report={
        "stage":STAGE,
        "original_s93_cb61_sha256":sha(s93),
        "original_s93_side_sha256":sha(side),
        "original_s142_43m_sha256":sha(s142),
        "inherited_trained_neural_S119k_SHA256":sha(parent),
        "source_new_user_questions_precommitted_in_research_commit":True,
        "original_real_43M_GT_12layer_native_full_vocab":True,
        "real_tokenizer_full_native_vocab":tok.vocab_size,
        "generation_self_feeds_real_selected_BPE_tokens":True,
        "two_separate_full_prefix_original_CPU_streams_per_prompt":True,
        "generated_complete_answers_NOT_guaranteed":True,
        "max_greedy_new_native_tokens":MAX_NEW,
        "real_original_CPU_prefill_forward_calls":sum(
            r[k]["n_forward_actual_original43m_calls"]
            for r in prompt_results
            for k in ("original_native_greedy",
                      "offpath_trained_structured_native_greedy")),
        "n_prompts":len(prompt_results),
        "n_prompts_where_shadow_changes_generated_tokens":sum(
            r["different_generated_native_sequences"] for r in prompt_results),
        "actual_generated_sequences":prompt_results,
        "original_43m_and_green_tunnel_parameter_sha_prepost_equal":True,
        "all_S115_old_53751_FP16_exact":True,
        "shadow_registered_in_actual_research_Calibri_but_offpath":True,
        "original_model_logits_never_modified_by_shadow":True,
        "not_streaming_KV_token_loop_because_full_prefix_recomputed":True,
        "preexisting_GT_online_KV_causality_NOT_proven":True,
        "original_FULL_S101_controller_routing_not_replayed":True,
        "new_full_616_J7_EOS_candidate_NOT_replayed":True,
        "actual_open_unseen_generalQA_gold_NOT_used":True,
        "no_human_graded_answer_quality_no_independent_benchmark":True,
        "all_in_loader_packed_bpp_NOT_certified":True,
        "matched_native_CPU_throughput_NOT_certified":True,
        "no_real_43m_end_to_end_neural_backprop":True,
        "protected_SynTagRus_B_GOLD_NOT_OPENED":True,
        "TRAIN_A_ancestor_license_CC_BY_NC_SA_4_0_noncommercial":True,
        "output_gate":0,"candidate_accepted":False,
        "activation_allowed":False,
        "decision":"REAL_NATIVE_AUTOREGRESSIVE_RESEARCH_SHADOW_NO_QUALITY_CLAIMS",
    }
    (out/REPORT).write_text(
        json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf8")
    print("S119L_REAL_43M_GT_NATIVE_AUTOREGRESSIVE "+
          json.dumps({
            "true_original_decoder_calls":
                report["real_original_CPU_prefill_forward_calls"],
            "prompts":len(prompt_results),
            "shadow_changes_generated_tokens":report[
                "n_prompts_where_shadow_changes_generated_tokens"],
            "original_texts":[r["original_native_greedy"]["generated_text"]
                              for r in prompt_results],
            "shadow_texts":[r["offpath_trained_structured_native_greedy"][
                              "generated_text"] for r in prompt_results],
            "quality_benchmark":False,
            "candidate_accepted":False,
          },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for k in ("s93","side","s142","parent","parent_report","out"):
        p.add_argument("--"+k.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.parent,a.parent_report,a.out)


if __name__=="__main__":main()
