#!/usr/bin/env python3
"""Oscar v1H: fully paid TWO RECURRENT FULL-8 Green Tunnels by decode phase.

First next token from genuine pretrained original 42M FP32-compatible
Oscar v1F rank32 GT, all later tokens from v1G on-policy trained rank32 GT.
Original external pretrained 8x512 FP32 teacher used ONLY for comparison.
Both GT states are separately physically paid and original cold K32V8 core
is immutable. NO 12x320 CALIBRI model falsely claimed or touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from direct_codebook_v59 import CB59
from oscar.v1e.audit_checkpoint import source as true_pretrained_source
from oscar.v1e.safe_pretrained_llama import (
    from_checkpoint,materialize_decoded_cb59,
)
from oscar.v1f.green_tunnel import FullLayerGreenTunnel,DEPTH,WIDTH,RANK
from oscar.v1f.train_real_all8_green_tunnel import (
    SOURCE_SHA,SOURCE_REV,TOKENIZER,TOKENIZER_REV,
    EXPECTED_ORIGINAL_UNIQUE,MAX_NEW,prompt_ids,EVAL_OLD,EVAL_OTHER,
    true_greedy,
)
from oscar.v1g.real_onpolicy_gt8 import (
    V1F_CB59_SHA,V1F_CB61_SHA,V1F_GT_SHA,DEBUG_NEW,
)

V1G_SHA="506ea6793069e2b913d1322d506f10743133278a46bdf3bc916b4c08ed1a1d1d"
MAX_PHYSICAL_BYTES=int(.7*EXPECTED_ORIGINAL_UNIQUE/8)
RUNTIME_MANIFEST_BYTES=3072
NEW_DEBUG=(
  "The teacher wrote a kind message on the classroom board.",
  "A brown puppy waited by the door for his family.",
  "The children saw a bright rainbow after the storm.",
  "A fisherman found a tiny silver fish in the pond.",
  "Nora carried a basket of apples to her grandmother.",
  "The little turtle slowly crossed the garden path.",
  "Ben lost his yellow hat while running in the park.",
  "The stars glowed over the quiet mountain town.",
)
PHASES=("V1F_FIRST_ALL_STEPS","V1G_ONPOLICY_ALL_STEPS",
        "V1H_FIRST_V1F_LATER_V1G")


def sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def dump(path,data):
    Path(path).write_text(json.dumps(data,sort_keys=True,indent=2,
                      ensure_ascii=False)+"\n",encoding="utf8")


def log(stage,**details):
    print(json.dumps({"stage":stage,**details},
                     ensure_ascii=False),flush=True)


def scopes(mode,step):
    if mode not in PHASES:
        raise AssertionError("Oscar v1H invalid first/later route mode")
    on_old=mode==PHASES[0] or (mode==PHASES[2] and step==0)
    on_new=mode==PHASES[1] or (mode==PHASES[2] and step>0)
    if on_old==on_new:
        raise AssertionError("Oscar phase routing must activate EXACTLY ONE original 8layer GT")
    return on_old,on_new


def repetition(seq:list[int]):
    if not seq:return {"unigram_repeat":0., "bigram_repeat":0.}
    n=len(seq)
    bigrams=[(seq[i],seq[i+1]) for i in range(n-1)]
    return {
      "unigram_repeat":(n-len(set(seq)))/n,
      "bigram_repeat":(len(bigrams)-len(set(bigrams)))/
                      max(1,len(bigrams)),
    }


@torch.inference_mode()
def dual_phase_greedy(student,teacher,old_gt,new_gt,tok,input_ids,mode):
    """Truly causal recursive full8 FP32-decoded K32/V8 inference.

    Both full8 hooks are installed but EXACTLY one is active each
    actual forward call. Underlying original K32/V8 weights and output
    head NEVER switch, only original paid GT state switches at token1.
    """
    ctx=input_ids.clone()
    output=[]
    teacher_next=[]
    final_cos=[]
    eos=False
    for step in range(MAX_NEW):
        old_enabled,new_enabled=scopes(mode,step)
        with old_gt.scope(old_enabled),new_gt.scope(new_enabled):
            student_logit,_=student(ctx)
        if (old_gt.last_coverage!=tuple(range(DEPTH)) or
            new_gt.last_coverage!=tuple(range(DEPTH)) or
            len(new_gt.latest_layers)!=DEPTH):
            raise AssertionError("Oscar v1H original recurrent 8layer hooks missing")
        tl,ths=teacher(ctx,hidden=True)
        teacher_choice=int(tl[0,-1].argmax())
        score=student_logit[0,-1]
        chosen=int(score.argmax())
        teacher_next.append({
          "student_generated_prefix_len":ctx.shape[1]-input_ids.shape[1],
          "teacher_FP32_next_token_on_student_prefix":teacher_choice,
          "Oscar_student_emitted_token":chosen,
          "teacher_next_rank_under_student":int(
             (score>score[teacher_choice]).sum())+1,
          "original_FP32_first_match_on_same_student_prefix":
             chosen==teacher_choice,
          "selected_original_gt":"v1F_first" if old_enabled else "v1G_later",
        })
        cosine=float(torch.nn.functional.cosine_similarity(
            new_gt.latest_layers[-1][0,-1].float()[None],
            ths[DEPTH][0,-1].float()[None]))
        final_cos.append(cosine)
        output.append(chosen)
        if chosen==tok.eos_token_id:
            eos=True
            break
        ctx=torch.cat((ctx,ctx.new_tensor([[chosen]])),dim=-1)
    return {
      "generated_original_Oscar_token_ids":output,
      "decoded_text":tok.decode(
          output,skip_special_tokens=True,clean_up_tokenization_spaces=False),
      "real_EOS_observed":eos,
      "actual_original_FP32_targets_on_OWN_student_prefix":teacher_next,
      "real_last_hidden_cosine_on_student_generated_prefix":final_cos,
      "mean_last_hidden_cosine_on_OWN_student_prefix":sum(final_cos)/len(final_cos),
      "actual_repetition":repetition(output),
      "full8_decoder_GT_visit_verified_on_every_generated_token":True,
    }


def summarize(rows,route):
    data=[r[route] for r in rows]
    return {
       "real_public_DEBUG_n":len(data),
       "first_next_token_matches_independently_pretrained_FP32":sum(
          int(x["generated_original_Oscar_token_ids"][0]==
              r["original_pretrained_original_FP32"]["ids"][0])
          for x,r in zip(data,rows)),
       "full_actual_free_autoregressive_8tokens_with_EOS_matches_FP32":sum(
          int(x["generated_original_Oscar_token_ids"]==
              r["original_pretrained_original_FP32"]["ids"])
          for x,r in zip(data,rows)),
       "mean_actual_original_FP32_student_prefix_last_hidden_cosine":sum(
          x["mean_last_hidden_cosine_on_OWN_student_prefix"] for x in data)/len(data),
       "mean_unigram_repeat_fraction":sum(
          x["actual_repetition"]["unigram_repeat"] for x in data)/len(data),
       "mean_bigram_repeat_fraction":sum(
          x["actual_repetition"]["bigram_repeat"] for x in data)/len(data),
       "mean_emitted_token_count":sum(
          len(x["generated_original_Oscar_token_ids"]) for x in data)/len(data),
       "actually_emitted_EOS_count":sum(int(x["real_EOS_observed"]) for x in data),
    }


def run(a):
    torch.manual_seed(421988)
    torch.set_num_threads(2)
    output=Path(a.out)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Oscar v1H report directory must start truly empty")
    refs={
      "cb59":(Path(a.original_CB59),V1F_CB59_SHA),
      "cb61":(Path(a.original_CB61),V1F_CB61_SHA),
      "gt_first":(Path(a.v1f_first_GT_FP16),V1F_GT_SHA),
      "gt_later":(Path(a.v1g_later_GT_FP16),V1G_SHA),
    }
    for name,(path,expected) in refs.items():
        if not path.is_file() or sha(path.read_bytes())!=expected:
            raise AssertionError("Original independent pretrained Oscar v1H source mismatch: "+name)
    prompts=EVAL_OLD+EVAL_OTHER+DEBUG_NEW+NEW_DEBUG
    if (len(prompts)!=20 or len(set(prompts))!=20 or
        len(EVAL_OLD)!=4 or len(EVAL_OTHER)!=4 or
        len(DEBUG_NEW)!=4 or len(NEW_DEBUG)!=8):
        raise AssertionError("Oscar v1H original prior+new debug cohort invalid")
    prereg={
      "stage":"OSCAR_v1H_FIRST_ORIGINAL_v1F_GreenTunnel_LATER_ONPOLICY_v1G_PRECOMMIT",
      "independently_pretrained_FP32_source":"karpathy/tinyllamas/stories42M.pt",
      "source_revision_pinned":SOURCE_REV,
      "original_F32_checkpoint_SHA256":SOURCE_SHA,
      "original_trainable_unique_FP32_parameters":EXPECTED_ORIGINAL_UNIQUE,
      "original_teacher_layers":DEPTH,
      "original_teacher_hidden":WIDTH,
      "original_core_CB59_sha":V1F_CB59_SHA,
      "original_core_CB61_sha":V1F_CB61_SHA,
      "physically_paid_OLD_FIRST_GT_v1F_sha":V1F_GT_SHA,
      "physically_paid_NEW_LATER_GT_v1G_sha":V1G_SHA,
      "specific_position_phase_only_switch_no_prompt_text_routing":True,
      "original_S142_12layer_43M_CALIBRI_NOT_MODIFIED":True,
      "no_extra_32k_head":True,
      "no_new_model_training_or_public_debug_labels_used_as_optimizer":True,
      "original_public_previously_used_debug12":list(prompts[:12]),
      "new_precommitted_not_previously_scored_debug8":list(prompts[12:]),
      "original_teacher_proxy_LLaMA_tokenizer_training_equivalence_UNVERIFIED":True,
      "original_full_CB61_tied_embedding_and_original_head_immutable":True,
      "three_modes":list(PHASES),
      "model_quality_QA_not_established_by_any_public_DEBUG":True,
      "original_CALIBRI_616_J7_not_retested":True,
      "native_CPU_direct5bit_kernel_90pct_not_evaluated":True,
      "no_LIVE_main_Render_Supabase_NRA_APK_ordinaryJessicaChat":True,
    }
    dump(output/"OSCAR_V1H_PRECOMMIT_BEFORE_LOADING_MODELS.json",prereg)
    checkpoint,weights,metadata=true_pretrained_source(output)
    if (metadata["archive_SHA256"]!=SOURCE_SHA or
        metadata["pinned_exact_HF_revision_sha40"]!=SOURCE_REV or
        metadata["model_parameters_real_unique"]!=EXPECTED_ORIGINAL_UNIQUE):
        raise AssertionError("Original independently pretrained 42M FP32 teacher provenance mismatch")
    teacher=from_checkpoint(weights).eval().requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(TOKENIZER,revision=TOKENIZER_REV)
    if tok.vocab_size!=32000 or tok.eos_token_id is None:
        raise AssertionError("Original proxy tokenizer changed from pinned Oscar v1F tests")
    compressed=CB59(refs["cb59"][0].read_bytes())
    if compressed.meta["original_unique_parameter_count"]!=EXPECTED_ORIGINAL_UNIQUE:
        raise AssertionError("Oscar original independent pretrained 41.69M K32/V8 source missing")
    original=materialize_decoded_cb59(teacher,refs["cb59"][0].read_bytes())
    original.eval().requires_grad_(False)
    old_gt=FullLayerGreenTunnel().install(original)
    new_gt=FullLayerGreenTunnel().install(original)
    old_payload=load_file(str(refs["gt_first"][0]))
    new_payload=load_file(str(refs["gt_later"][0]))
    old_gt.restore_paid(old_payload)
    new_gt.restore_paid(new_payload)
    old_gt.requires_grad_(False)
    new_gt.requires_grad_(False)
    rows=[]
    with torch.inference_mode():
        for ix,story in enumerate(prompts):
            inp=prompt_ids(tok,story)
            teacher_answer=true_greedy(teacher,tok,inp)
            by_mode={}
            for mode in PHASES:
                by_mode[mode]=dual_phase_greedy(
                    original,teacher,old_gt,new_gt,tok,inp,mode)
            # Exactly first original v1F forward, independent of disabled v1G.
            if by_mode[PHASES[0]]["generated_original_Oscar_token_ids"][0]!=\
               by_mode[PHASES[2]]["generated_original_Oscar_token_ids"][0]:
                raise AssertionError("Phase switching altered original v1F FIRST token")
            row={
               "debug_index":ix,
               "part_of_previous_PUBLIC_debug":ix<12,
               "new_public_debug_not_used_in_any_gradient":ix>=12,
               "prompt_SHA256":sha(story.encode()),
               "actual_prompt":story,
               "original_pretrained_original_FP32":teacher_answer,
               **by_mode,
            }
            rows.append(row)
            log("OSCAR_V1H_ACTUAL_DUAL_PHASE_ORIGINAL_PRETRAINED42M",
                idx=ix+1,orig=teacher_answer["text"][:65],
                old=by_mode[PHASES[0]]["decoded_text"][:65],
                onpolicy=by_mode[PHASES[1]]["decoded_text"][:65],
                hybrid=by_mode[PHASES[2]]["decoded_text"][:65],
                matched_teacher_first=by_mode[PHASES[2]][
                    "generated_original_Oscar_token_ids"][0]==teacher_answer["ids"][0],
                matched_teacher_full=by_mode[PHASES[2]][
                    "generated_original_Oscar_token_ids"]==teacher_answer["ids"])
    old_results=[summarize(rows[:8],r) for r in PHASES]
    if old_results[0]["first_next_token_matches_independently_pretrained_FP32"]!=5:
        raise AssertionError("Oscar v1H must regenerate exact previously measured V1F original first5/8")
    if old_results[1]["first_next_token_matches_independently_pretrained_FP32"]!=0:
        raise AssertionError("Oscar v1H must regenerate exact previously measured V1G regressed first0/8")
    if old_results[2]["first_next_token_matches_independently_pretrained_FP32"]!=5:
        raise AssertionError("Oscar phase-specific first original GT not preserving exactly5/8")
    if (sha(refs["cb59"][0].read_bytes())!=V1F_CB59_SHA or
        sha(refs["cb61"][0].read_bytes())!=V1F_CB61_SHA or
        sha(refs["gt_first"][0].read_bytes())!=V1F_GT_SHA or
        sha(refs["gt_later"][0].read_bytes())!=V1G_SHA):
        raise AssertionError("Original pretrained42M/source/GT model physically altered during routing")
    # Physically paid artifact uses TWO trained full8 rank32 GT states
    # under positional choice. Every byte counts; no third head sidecar.
    physical=output/"OSCAR_V1H_REAL_DUAL_ROUTE_PAID_RUNTIME"
    physical.mkdir(parents=True,exist_ok=True)
    files={
       "OSCAR_SOURCE_42M_ONE_AUTHENTIC_COLD.cb61":refs["cb61"][0],
       "OSCAR_REAL_FIRST_STEP_v1F_GT8_FP16.safetensors":refs["gt_first"][0],
       "OSCAR_REAL_LATER_STEP_v1G_GT8_FP16.safetensors":refs["gt_later"][0],
    }
    copied={}
    for name,path in files.items():
        dest=physical/name
        shutil.copyfile(path,dest)
        copied[name]={"bytes":dest.stat().st_size,
                      "SHA256":sha(dest.read_bytes())}
    # Critical physical budget includes BOTH actual trained GTs.
    realbytes=sum(x["bytes"] for x in copied.values())+RUNTIME_MANIFEST_BYTES
    if realbytes>MAX_PHYSICAL_BYTES:
        raise AssertionError("Oscar positional specialization exceeds the genuine original42M 0.7bpp rate")
    metrics={
       "original_eight_PUBLIC_debug":{
           m:summarize(rows[:8],m) for m in PHASES},
       "all_prior_twelve_PUBLIC_debug":{
           m:summarize(rows[:12],m) for m in PHASES},
       "new_eight_precommitted_PUBLIC_debug":{
           m:summarize(rows[12:],m) for m in PHASES},
       "all_twenty_PUBLIC_debug":{
           m:summarize(rows,m) for m in PHASES},
    }
    final_manifest={
      "model_type":"OSCAR v1H authentic externally pretrained original42M first/later GT phase conditioned",
      "original_42M_true_unique_params":EXPECTED_ORIGINAL_UNIQUE,
      "input_original_pretrained_LLaMA_layers":8,
      "original_hidden_width":512,
      "NOT_CALIBRI_original_12layer":True,
      "position_only_switch":"v1F full8 GT for next token 0; v1G full8 GT for tokens >=1",
      "one_shared_original_CB61_no_duplicate_weights":True,
      "two_ACTUAL_pretrained_FP32_calibrated_GT8_FP16_checkpoints_PAID":True,
      "no_32k_head_correction":True,
      "original_input_CB61_and_FP16_files":copied,
      "literal_manifest_bytes":RUNTIME_MANIFEST_BYTES,
      "complete_real_model_plus_GT_manifest_bytes":realbytes,
      "original_41689600_unique_full_PHYSICAL_bpp":8*realbytes/EXPECTED_ORIGINAL_UNIQUE,
      "all_physical_files_under_0p7":8*realbytes/EXPECTED_ORIGINAL_UNIQUE<=.7,
      "pretrained_FP32_teacher_source_not_shipped":True,
      "original_tokenizer_and_native_CPU_runtime_binary_NOT_in_source_budget":True,
      "actual_causal_8token_greedy_through_original_K32V8_model":True,
      "real_original_FP32_full_teacher_public_debug_match":
          metrics["all_twenty_PUBLIC_debug"][PHASES[2]][
              "full_actual_free_autoregressive_8tokens_with_EOS_matches_FP32"],
      "model_NOT_accepted_or_deployed":True,
      "independent_sealed_QA_and_original616J7_NOT_run":True,
    }
    blob=(json.dumps(final_manifest,sort_keys=True,indent=2,
                     ensure_ascii=False)+"\n").encode()
    if len(blob)>RUNTIME_MANIFEST_BYTES:
        raise AssertionError("Oscar real physical first/later 8layer inference manifest oversize")
    manifest=physical/"OSCAR_V1H_POSITIONAL_DUAL_GT_REAL_RUNTIME_CONFIG.json"
    manifest.write_bytes(blob+b" "*(RUNTIME_MANIFEST_BYTES-len(blob)))
    if realbytes!=sum(p.stat().st_size for p in physical.iterdir()):
        raise AssertionError("Oscar v1H physical model GT2 rate not literal")
    report={
      **prereg,
      "all_modes_actually_evaluated_against_independent_original_FP32_teacher":True,
      "actual_20_PUBLIC_DEBUG_real_original_FP32_answers_and_all_GT_modes":rows,
      "actual_original_FP32_match_and_repetition_metrics":metrics,
      "physical_original_42M_rate":{
         "one_compressed_COLD_CB61_original_42M":copied[
             "OSCAR_SOURCE_42M_ONE_AUTHENTIC_COLD.cb61"]["bytes"],
         "first_original_v1F_rank32_full8_GT_FP16":copied[
             "OSCAR_REAL_FIRST_STEP_v1F_GT8_FP16.safetensors"]["bytes"],
         "later_original_v1G_rank32_full8_GT_FP16":copied[
             "OSCAR_REAL_LATER_STEP_v1G_GT8_FP16.safetensors"]["bytes"],
         "charged_full_runtime_manifest":RUNTIME_MANIFEST_BYTES,
         "complete_original_unique_trained_42M_source_and_GT2_bytes":realbytes,
         "real_noninflated_original_pretrained_42M_bpp":8*realbytes/EXPECTED_ORIGINAL_UNIQUE,
         "actual_0p7_full_weights_and_GT2_and_config_pass":realbytes<=MAX_PHYSICAL_BYTES,
         "tokenizer_executable_C_runtime_NOT_included":True
      },
      "no_full_original_FP32_pretrained_language_quality_accepted":True,
      "original_CALIBRI_S142_LIVE_NRA2_616J7_UNTOUCHED":True,
      "no_clean_unseen_QA":True,
      "real_original_pretrained_teacher_NOT_reconstructed_for_deployment":True,
    }
    dump(output/"OSCAR_V1H_FINAL_TRUE_EXTERNAL42M_FP32_TWO_GT_PHASE_REAL_GREEDY.json",report)
    log("OSCAR_V1H_FINAL_REAL_42M_FIRST_LATER_GREEN_TUNNEL",
        original_old8_FP32_first=metrics["original_eight_PUBLIC_debug"][PHASES[0]][
            "first_next_token_matches_independently_pretrained_FP32"],
        original_v1g_old8_FP32_first=metrics["original_eight_PUBLIC_debug"][PHASES[1]][
            "first_next_token_matches_independently_pretrained_FP32"],
        original_phase_switched_old8_FP32_first=metrics["original_eight_PUBLIC_debug"][PHASES[2]][
            "first_next_token_matches_independently_pretrained_FP32"],
        first_later_original_FP32_full_8token_old8=metrics[
            "original_eight_PUBLIC_debug"][PHASES[2]][
                "full_actual_free_autoregressive_8tokens_with_EOS_matches_FP32"],
        first_later_original_FP32_full_8token_new8=metrics[
            "new_eight_precommitted_PUBLIC_debug"][PHASES[2]][
                "full_actual_free_autoregressive_8tokens_with_EOS_matches_FP32"],
        original_FP32_full_match_all20=metrics[
            "all_twenty_PUBLIC_debug"][PHASES[2]][
                "full_actual_free_autoregressive_8tokens_with_EOS_matches_FP32"],
        phase_switched_all20_unigram_repeat=metrics[
            "all_twenty_PUBLIC_debug"][PHASES[2]][
                "mean_unigram_repeat_fraction"],
        source_plus_2_real_FP16_GT_paid_bytes=realbytes,
        all_original_42M_full_source_with_GT_bpp=8*realbytes/EXPECTED_ORIGINAL_UNIQUE,
        no_production_or_Calibri12layer_merged=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("original_CB59","original_CB61",
                "v1f_first_GT_FP16","v1g_later_GT_FP16","out"):
        p.add_argument(key,type=Path)
    run(p.parse_args())
