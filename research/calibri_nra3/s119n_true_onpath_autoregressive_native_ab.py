"""S119n: actual NATIVE 43M+GT ON-PATH autoregressive A/B, not proxy CE.

First physical free-running experiment after successful S119m interlayer
gradient (PR #913, CI 44949299494): load its physically trained FP16 full
NRA3+NPA1+structured single neural child (including original 53,751).
For *every* generated token:
  (1) forward original compressed S142 43M+12-layer GreenTunnel through all
      12 layers on current NATIVE BPE prefix, capture original H320 states,
      genuine original LM head native full-vocab logits, and verify SHA.
  (2) compute S119m learned three-component interlayer residual and inject
      it AFTER real decoder layer 5 and BEFORE original layers 6..11,
      with a second genuine original decoder forward on SAME prefix.
  (3) compare full-vocab argmax choices, append the selected next native BPE
      token to each stream independently, repeat. Stop on original native EOS.
This is real on-path intervention (unlike S119l off-path head shadow) but
still a TWO PASS expensive, CPU nonproduction, research-only module hook.
The original compressed model, GT, S142 and 53,751 S115 NRA3 FP16
pretrained weights are *never* modified. New on-path adapter has output_gate
zero, candidate_accepted false, and is only manually invoked within tightly
scoped research onpath_forward. No main/LIVE/Render/Supabase/Jessica changes.

Four commit-predeclared short Russian prompts with no answer/gold provided;
no supervised scoring, no independent quality or J7/CPU/bpp acceptance.
Raw model greedy lacks the complete original S101/J7 controller: texts are
NOT a definitive quality estimate of the full Calibri system.
SynTagRus-A parent used with CC BY-NC-SA 4.0 NONCOMMERCIAL restriction.
S118-B GOLD and any previous spent evaluation remain CLOSED.
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
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    NativeFullHeadShadow, capture_all12, load_actual_original, sha, S142_SHA,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    LearnedOnPathInterlayer, native_intervention_forward, STAGE as PARENT_STAGE,
    CHECKPOINT as PARENT_FILE, REPORT as PARENT_REPORT, HOOK_LAYER,
)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    strict_chat_source_ids, MAX_PREFIX,
)
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear

STAGE = "S119N_REAL_S142_43M_GT_ONPATH_GENERATION_NATIVE_RESEARCH_ONLY"
REPORT = "S119N_REAL_NATIVE_AUTOREGRESSIVE_ONPATH_AB_UNLABELED_4_PROMPTS.json"
PROMPTS = (
    ("arithmetic", "Чему равно четыре плюс три? Ответь кратко."),
    ("event", "Олег передал журнал Лене. Кому достался журнал?"),
    ("negation", "Миша не дал Саше книгу. Получил ли Саша книгу от Миши?"),
    ("conversation", "Привет! Как дела?"),
)
MAX_NEW = 5
SEED = 119127
PREFIX = "model.nra3_calibri_joint.s119m_onpath."


def original_digest_without_child(model):
    """Hash all physical original compressed model and GT weights."""
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if "nra3_s119n_onpath" in name: continue
        h.update(name.encode()+b"\0"+
                 p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_pinned_onpath(s93,side,s142,parent,parent_report):
    check(s93.name=="S93_UNACCEPTED_SELECTED_CB61.s61"
          and side.name=="S93_UNACCEPTED_SELECTED_JOINT.safetensors"
          and s142.name=="S142_ACTUAL_43M_ONE_J7_FP16.safetensors"
          and parent.name==PARENT_FILE
          and parent_report.name==PARENT_REPORT
          and sha(s142)==S142_SHA,
          "Only physical original S142 and S119m learned onpath parent")
    report=json.loads(parent_report.read_text(encoding="utf8"))
    check(report["stage"]==PARENT_STAGE
          and report["physical_total_one_child_sha256"]==sha(parent)
          and report["original_43M_cb61_s93_sha256"]==sha(s93)
          and report["original_s142_43m_gt_sha256"]==S142_SHA
          and report["original_pretrained_43M_and_GT_all_weights_sha_unchanged"]
          and report["original_pretrained_S115_NRA3_53751_FP16_unchanged"]
          and report["original_12layer_onpath_intervention_after_layer"]==5
          and report["output_gate"]==0
          and report["candidate_accepted"] is False,
          "Cannot load unverified S119m research candidate")
    raw=load_file(str(parent),device="cpu")
    check(bool(raw) and all(k.startswith(PREFIX)
          and v.dtype==torch.float16 for k,v in raw.items()),
          "Full FP16 S119m single in-Calibri child")
    adapter=LearnedOnPathInterlayer(NativeFullHeadShadow()).eval()
    old={k[len(PREFIX):]:v.float() for k,v in raw.items()}
    check(set(old)==set(adapter.state_dict())
          and sum(v.numel() for v in raw.values())>116_777
          and sum(p.numel() for p in
                adapter.shadow.unified.nra3_neural_weights.parameters())==53_751,
          "Only full S119m child with all old mini areas")
    adapter.load_state_dict(old,strict=True)
    frozen_old={k:v.clone() for k,v in raw.items()
                if ".nra3_neural_weights." in k}
    check(frozen_old and all(torch.equal(
              adapter.state_dict()[k[len(PREFIX):]].half(),v)
              for k,v in frozen_old.items()) and
          adapter.output_gate==0 and
          adapter.shadow.output_gate==0 and
          not adapter.candidate_accepted
          and not adapter.shadow.candidate_accepted,
          "Full 53,751 original FP16 exact and all research output gates zero")
    return adapter,raw,report,frozen_old


def authentic_two_pass_step(model,sources,bridge,full,expansion,tok,
                            adapter,prefix_ids):
    """Original 43M+GT CPU twice on SAME native self-fed prefix.

    No future labels or user-written answer. Unlike S119m teacher forcing,
    prefix may include genuine model-generated IDs. First-pass S95 features
    derived from the SAME prefix's text as second pass, never future text.
    """
    check(isinstance(prefix_ids,list) and 2<=len(prefix_ids)<MAX_PREFIX
          and all(isinstance(t,int) and 0<=t<tok.vocab_size
                  for t in prefix_ids)
          and adapter.output_gate==0 and
          not adapter.candidate_accepted,"native full-prefix research only")
    ids=torch.tensor([prefix_ids],dtype=torch.long)
    text=tok.decode(prefix_ids,skip_special_tokens=False)
    features=torch.as_tensor(s160.prompt_only_vector(tok,text),
                             dtype=torch.float32)
    check(features.shape==(512,) and bool(torch.isfinite(features).all()),
          "S95 frozen prompt route must be prefix-only")
    with torch.inference_mode(),expansion.prefill_only(model),\
         s160.prefill_only_tunnel(model,full):
        with s160.frozen_cpu_phase_scope(
            model,bridge,"S93",features,phase="prefill_only"):
            with capture_all12(model) as hooks:
                clear(sources)
                out=model(input_ids=ids,past_key_values=None,
                          use_cache=True,output_hidden_states=True)
                check(out.past_key_values is not None and
                      out.hidden_states[-1].shape==(1,len(prefix_ids),320),
                      "FIRST genuine frozen S142 native prefix all 12 layers")
                baseline=out.logits[:,-1].detach().float().cpu().clone()
                last=out.hidden_states[-1][:,-1].detach().float().cpu().clone()
                check(torch.allclose(
                    model.lm_head(last).float(),baseline,
                    atol=2e-3,rtol=1e-4),
                    "original native 49152 logits match original model LM head")
                clear(sources)
                del out
    real_hidden=torch.stack([hooks[k] for k in range(12)],1).float().clone()
    # S119m fixed physical trained model applies its learnable neural
    # delta inside SECOND original model.forward after original layer5.
    record={
        "h12":real_hidden, "prefix":len(prefix_ids),
        "native_prefix_ids":ids.clone(),
    }
    with torch.enable_grad():
        changed=native_intervention_forward(
            model,sources,bridge,full,expansion,
            adapter,record,tok,contextual=True).detach().float()
    check(changed.shape==baseline.shape and changed.shape[-1]==49_152
          and bool(torch.isfinite(changed).all())
          and bool(torch.isfinite(baseline).all()),
          "SECOND original true native full vocab onpath output")
    with torch.no_grad():
        delta=adapter(real_hidden,contextual=True)
        displacement=float((changed-baseline).abs().max())
        magnitude=float(delta.norm())
    return baseline,changed,{
        "full_vocab_top1_original":int(baseline.argmax(-1)),
        "full_vocab_top1_true_onpath":int(changed.argmax(-1)),
        "true_interlayer_residual_l2":magnitude,
        "max_abs_native_full_logits_delta":displacement,
    }


def greedy_stream(tok,model,sources,bridge,full,expansion,adapter,
                  question,*,onpath):
    """Each chosen full-native-vocab token physically re-enters original CPU."""
    check(isinstance(onpath,bool),"exactly one of two source-defined paths")
    ids=list(strict_chat_source_ids(tok,question))
    initial_len=len(ids)
    emitted=[]
    steps=[]
    start=time.perf_counter()
    for _ in range(MAX_NEW):
        original,enhanced,diagnostics=authentic_two_pass_step(
            model,sources,bridge,full,expansion,tok,adapter,ids)
        selected=diagnostics[
            "full_vocab_top1_true_onpath" if onpath
            else "full_vocab_top1_original"]
        check(0<=selected<tok.vocab_size,
              "generated next ID must be native ORIGINAL BPE vocabulary")
        steps.append({
            "step":len(emitted)+1,
            "self_fed_original_CPU_prefix_length":len(ids),
            "selected_native_token_id":selected,
            **diagnostics,
        })
        emitted.append(selected)
        ids.append(selected)
        if selected==tok.eos_token_id:
            break
        check(len(ids)<MAX_PREFIX,"research-only decoder context bound")
    check(len(steps)==len(emitted)>0 and
          all(r["self_fed_original_CPU_prefix_length"]==initial_len+i
              for i,r in enumerate(steps)),
          "each model-chosen BPE token really self-feeds into next CPU model")
    return {
        "generated_native_token_ids":emitted,
        "generated_text":tok.decode(emitted,skip_special_tokens=True),
        "eos_generated":bool(emitted[-1]==tok.eos_token_id),
        "actual_original_cpu_forward_calls":len(steps)*2,
        "two_full_original43m_prefix_passes_per_generated_token":True,
        "steps":steps,
        "CPU_elapsed_seconds_NOT_normal_throughput":
            time.perf_counter()-start,
    }


def run(s93,side,s142,parent,parent_report,out):
    check(not out.exists(),"research new outputs only, never overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    adapter,raw,prior,old53751=load_pinned_onpath(
        s93,side,s142,parent,parent_report)
    original,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==49_152,"only original complete native BPE tokenizer")
    initial_original_digest=original_digest_without_child(original)
    # The onpath child is part of only isolated research Calibri model;
    # forward hook remains scoped, never modifies normal model.forward.
    original.model.add_module("nra3_s119n_onpath",adapter)
    original.eval()
    check(original.model.nra3_s119n_onpath is adapter
          and original_digest_without_child(original)==initial_original_digest
          and all(not p.requires_grad
                  for n,p in original.named_parameters()
                  if "nra3_s119n_onpath" not in n),
          "true original 43M/GT model frozen, child lives inside nn.Module")
    runs=[]
    for category,question in PROMPTS:
        baseline=greedy_stream(
            tok,original,sources,bridge,full,expansion,adapter,
            question,onpath=False)
        onpath=greedy_stream(
            tok,original,sources,bridge,full,expansion,adapter,
            question,onpath=True)
        runs.append({
            "category":category,
            "question_SHA256":hashlib.sha256(
                question.encode("utf8")).hexdigest(),
            "question":question,
            "original_native_greedy":baseline,
            "trained_deep_onpath_native_greedy":onpath,
            "generated_token_sequences_differ":
                baseline["generated_native_token_ids"]!=
                onpath["generated_native_token_ids"],
        })
    check(original_digest_without_child(original)==initial_original_digest
          and adapter.output_gate==0
          and adapter.shadow.output_gate==0
          and not adapter.candidate_accepted
          and all(torch.equal(
                adapter.state_dict()[k[len(PREFIX):]].detach().half(),v)
                for k,v in old53751.items()),
          "old 43M+GT weights and pretrained 53,751 NRA3 exactly preserved")
    total_forwards=sum(r[key]["actual_original_cpu_forward_calls"]
        for r in runs
        for key in ("original_native_greedy",
                    "trained_deep_onpath_native_greedy"))
    n_changed=sum(r["generated_token_sequences_differ"] for r in runs)
    out.mkdir(parents=True)
    report={
        "stage":STAGE,
        "original_s93_cb61_SHA256":sha(s93),
        "original_s93_side_SHA256":sha(side),
        "original_S142_43m_GT_SHA256":sha(s142),
        "inherited_full_S119m_FP16_SHA256":sha(parent),
        "original_true12_S142_43M_forward_twice_per_generated_token":True,
        "actual_onpath_after_original_decoder_layer":HOOK_LAYER,
        "original_native_full_vocabulary_size":tok.vocab_size,
        "native_chat_template_tokenizer_and_native_self_feeding":True,
        "n_prespecified_unlabeled_Russian_prompts":len(runs),
        "max_new_tokens_per_prompt":MAX_NEW,
        "true_original43m_12layer_cpu_forward_calls":total_forwards,
        "n_questions_with_different_generated_native_sequences":n_changed,
        "actual_raw_greedy_original_vs_real_onpath_generation":runs,
        "original_model_GT_S142_weights_hash_unchanged":True,
        "pretrained_old_53751_NRA3_FP16_exact_unchanged":True,
        "S119m_parent_train_source_only_no_new_gradient_at_eval":True,
        "every_new_token_runs_2_original_43m_full_prefix_passes":True,
        "original_and_onpath_differ_only_via_research_midlayer_NRA3_NPA":
            True,
        "research_hook_scoped_and_not_live":True,
        "not_original_S101_J7_controller_or_616_replay":True,
        "short_greedy_text_is_NOT_independent_answer_quality":True,
        "no_original_full_616_J7_EOS_candidate_replay":True,
        "no_loader_complete_all_in_bpp":True,
        "no_matched_real_CPU_throughput":True,
        "no_statistical_significance_or_unseen_qa":True,
        "not_efficiency_optimal_two_full_forward_passes":True,
        "parent_SynTagRus_A_training_license_CC_BY_NC_SA_4_noncommercial":True,
        "protected_S118_B_GOLD_unopened":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
        "decision":"RESEARCH_REAL43M_GT_ONPATH_NATIVE_GENERATION_UNLABELED_4_PROMPTS_NOT_ACCEPTED",
    }
    (out/REPORT).write_text(
        json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n",
        encoding="utf8")
    print("S119N_REAL43M_GT_ONPATH_AUTOREGRESSIVE "+
          json.dumps({
            "prompts":len(runs),
            "native_full_vocab":tok.vocab_size,
            "two_pass_original43m_forwards":total_forwards,
            "changed_native_sequences":n_changed,
            "original_text":[r["original_native_greedy"]["generated_text"]
                              for r in runs],
            "trained_onpath_text":[r["trained_deep_onpath_native_greedy"][
                                  "generated_text"] for r in runs],
            "original_43m_and_53751_FP16_protected":True,
            "quality_not_evaluated":True,"candidate_accepted":False,
          },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","parent","parent_report","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    args=p.parse_args()
    run(args.s93,args.side,args.s142,args.parent,args.parent_report,args.out)


if __name__=="__main__":
    main()
