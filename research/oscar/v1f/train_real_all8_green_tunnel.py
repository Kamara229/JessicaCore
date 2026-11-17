#!/usr/bin/env python3
"""Oscar v1F: actual trained layerwise Green Tunnel against original FP32.

Research-only. External actual PRETRAINED original 41.6896M FP32 LLaMA 8x512;
fresh Oscar COLD CB59 K32V8/CB61; frozen source matrices and tied embedding.
Train only exactly paid 8layer recurrent rank32 GT+rank8 output correction on
12 TRAIN-only story contexts. Evaluate eight separate fixed story contexts
with original pretrained free greedy decoding vs newly packaged and reloaded
GT+head / GT-only / head-only / original cold source. No clean outside QA.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer
from huggingface_hub import HfApi

from direct_codebook_v59 import CB59
from oscar.encoder import encode_fp32,to_cb61,MODE_COLD
from oscar.v1e.audit_checkpoint import source as true_fp32_source
from oscar.v1e.safe_pretrained_llama import (
    from_checkpoint,materialize_decoded_cb59,
)
from oscar.v1f.green_tunnel import (
    DEPTH,WIDTH,RANK,HEAD_RANK,VOCAB,
    FullLayerGreenTunnel,PaidHeadCorrection,
    merge_paid,restore_paid,actual_full_inference,
)

SOURCE_SHA="fbe124df8794c2e033432c22b59cff64485455b4644046c52a456d0bbd8859fe"
SOURCE_REV="0bd21da7698eaf29a0d7de3992de8a46ef624add"
EXPECTED_ORIGINAL_UNIQUE=41689600
TOKENIZER="hf-internal-testing/llama-tokenizer"
TOKENIZER_REV="d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"
STEPS=192
SEED=421919
TEMP=2.0
MAX_NEW=8
MAX_BPP=.7
MAX_REAL_BYTES=int(MAX_BPP*EXPECTED_ORIGINAL_UNIQUE/8)

TRAIN_STORIES=(
  "The little dog was happy because he had a new toy.",
  "The girl walked into the kitchen to get a glass of water.",
  "One sunny morning, a boy found a blue ball.",
  "Tom and his friends were playing together in the park.",
  "The bird sat in the tree and sang a pretty song.",
  "Once there was a big bear who wanted to eat honey.",
  "Lily saw a small cat next to the old house.",
  "After school, Sam liked to read a funny book.",
  "The princess opened the door and saw a beautiful garden.",
  "Mary helped her little brother put on his shoes.",
  "Jack and Ben went to the pond to see the ducks.",
  "On a cold day, the kids made a snowman outside.",
)
EVAL_OLD=(
    "Once upon a time, there was a little girl named Lily.",
    "In a quiet forest, the small rabbit was looking for food.",
    "The dog ran across the green field and found a red ball.",
    "A boy named Max wanted to help his friend learn to swim.",
)
EVAL_OTHER=(
    "The shy mouse found a piece of cheese in the barn.",
    "At night, the stars shone brightly above the village.",
    "Anna smiled when her father brought home a puppy.",
    "The red kite flew high above the hills in the wind.",
)


def sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()


def dump(path,record):
    Path(path).write_text(json.dumps(record,ensure_ascii=False,indent=2,
                sort_keys=True)+"\n",encoding="utf8")


def log(stage,**kw):
    print(json.dumps({"stage":stage,**kw},ensure_ascii=False),flush=True)


def prompt_ids(tok,text):
    ids=tok(text,return_tensors="pt",add_special_tokens=True).input_ids
    if (ids.ndim!=2 or ids.shape[0]!=1 or
        ids.shape[1]>80 or ids.shape[1]<=1 or
        int(ids.max())>=VOCAB):
        raise AssertionError("Oscar v1F original external 8layer training prompt invalid")
    return ids


@torch.inference_mode()
def true_greedy(model,tok,context,green=None,head=None):
    current=context.clone()
    result=[]
    eos=False
    for _ in range(MAX_NEW):
        if green is None:
            logits,_=model(current)
        else:
            logits=actual_full_inference(model,green,head,current)
        token=int(logits[0,-1].argmax())
        result.append(token)
        if token==tok.eos_token_id:
            eos=True
            break
        current=torch.cat((current,current.new_tensor([[token]])),dim=1)
    return {"ids":result,"actual_EOS":eos,
            "text":tok.decode(result,skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)}


@torch.inference_mode()
def paired_debug(teacher,compressed,green,head,tok,stories,tag):
    """8 or 4 NEVER TRAIN labels; actual old source vs trained true FP32."""
    result=[]
    for ix,text in enumerate(stories):
        ids=prompt_ids(tok,text)
        tlog,states=teacher(ids,hidden=True)
        slog=actual_full_inference(compressed,green,head,ids)
        if len(green.latest_layers)!=DEPTH or len(states)!=DEPTH+2:
            raise AssertionError("Oscar v1F all real pretrained 8 hidden states missing")
        t_last=tlog[0,-1].float()
        s_last=slog[0,-1].float()
        target_top=int(t_last.argmax())
        observed_top=int(s_last.argmax())
        per_layer=[]
        for i in range(DEPTH):
            a=states[i+1][0,-1].float()
            b=green.latest_layers[i][0,-1].float()
            per_layer.append({
               "original_pretrained_FP32_layer_zero_based":i,
               "original_original42M_FP32_vs_student_cosine":
                   float(F.cosine_similarity(a[None],b[None])),
               "relative_L2":float(torch.linalg.vector_norm(a-b)/
                                   torch.linalg.vector_norm(a).clamp_min(1e-12))
            })
        teacher_ans=true_greedy(teacher,tok,ids)
        student_ans=true_greedy(compressed,tok,ids,green,head)
        row={
           "stage":tag,
           "prompt_number":ix+1,
           "independent_original_prompt_SHA256":sha(text.encode()),
           "prompt_text":text,
           "original_FP32_pretrained_answer":teacher_ans,
           "real_OSCAR_new_compressed_answer":student_ans,
           "same_teacher_original_first_token":target_top==observed_top,
           "original_teacher_top1_in_new_Oscar_rank":
               int((s_last>s_last[target_top]).sum())+1,
           "original_pretrained_42M_same_full_8token_greedy":(
                teacher_ans["ids"]==student_ans["ids"]),
           "original_pretrained_teacher_FP32_vs_student_full_vocab_KL_T2":
                float(F.kl_div(
                    F.log_softmax(s_last[None]/TEMP,dim=-1),
                    F.softmax(t_last[None]/TEMP,dim=-1),
                    reduction="batchmean")*TEMP**2),
           "original_FP32_vs_Oscar_all_eight_layer_cosines":per_layer,
           "after_eight_layers_mean_cosine":
               sum(x["original_original42M_FP32_vs_student_cosine"]
                   for x in per_layer)/DEPTH,
           "after_last_layer_cosine":
               per_layer[-1]["original_original42M_FP32_vs_student_cosine"],
           "original_original_FP32_logits_entropy":
               float(torch.distributions.Categorical(logits=t_last).entropy()),
           "student_original_model_logits_entropy":
               float(torch.distributions.Categorical(logits=s_last).entropy()),
        }
        result.append(row)
        log("OSCAR_V1F_ACTUAL_ORIGINAL_FP32_VS_RELOADED_GT",
            stage=tag,debug_ix=ix+1,
            same_first=row["same_teacher_original_first_token"],
            same_full=row["original_pretrained_42M_same_full_8token_greedy"],
            final_hidden_cos=round(row["after_last_layer_cosine"],4),
            teacher=teacher_ans["text"][:70],
            student=student_ans["text"][:70])
    return result


def totals(rows):
    return {
      "n":len(rows),
      "real_fp32_teacher_first_match":sum(
          int(x["same_teacher_original_first_token"]) for x in rows),
      "real_fp32_teacher_FULL_8token_match":sum(
          int(x["original_pretrained_42M_same_full_8token_greedy"]) for x in rows),
      "mean_all_decoder_states_cos":sum(
          x["after_eight_layers_mean_cosine"] for x in rows)/len(rows),
      "mean_final_decoder_state_cos":sum(
          x["after_last_layer_cosine"] for x in rows)/len(rows),
      "mean_original_FP32_vs_calibrated_32k_logits_KL_T2":sum(
          x["original_pretrained_teacher_FP32_vs_student_full_vocab_KL_T2"]
          for x in rows)/len(rows),
    }


def build_train_targets(teacher,tok):
    result=[]
    with torch.inference_mode():
        for text in TRAIN_STORIES:
            ids=prompt_ids(tok,text)
            logits,h=teacher(ids,hidden=True)
            if (len(h)!=DEPTH+2 or
                logits.shape[2]!=VOCAB or
                not torch.isfinite(logits).all()):
                raise AssertionError("Oscar original pretrained independent 42M all layer targets incomplete")
            result.append({
              "ids":ids,
              "original_prompt_SHA256":sha(text.encode()),
              "teacher_whole_vocab_logits":logits.detach().clone().float(),
              "teacher_all8_intermediate":[v.detach().clone().float()
                                            for v in h[1:DEPTH+1]],
            })
    return result


def real_train_step(student,green,head,data):
    student_logits=actual_full_inference(
        student,green,head,data["ids"])
    expected=data["teacher_whole_vocab_logits"]
    if (expected.shape!=student_logits.shape or
        len(green.latest_layers)!=DEPTH):
        raise AssertionError("Oscar v1F genuine all layer original FP32 matched geometry missing")
    # Original FP32 teacher is *independently pretrained*, not derived from
    # compressed source. Full sequence next-token KL and teacher top1 CE.
    s=student_logits.float()
    t=expected.float()
    kl=F.kl_div(F.log_softmax(s/TEMP,dim=-1),
                F.softmax(t/TEMP,dim=-1),
                reduction="batchmean")*TEMP**2
    ce=F.cross_entropy(s.reshape(-1,VOCAB),
                       t.detach().argmax(-1).reshape(-1))
    layer_costs=[]
    layer_cos=[]
    for i in range(DEPTH):
        x=green.latest_layers[i].float()
        y=data["teacher_all8_intermediate"][i]
        if x.shape!=y.shape:
            raise AssertionError("Oscar v1F reconstructed student layer target shape mismatch")
        cosine=F.cosine_similarity(
            F.layer_norm(x,(WIDTH,)),F.layer_norm(y,(WIDTH,)),dim=-1)
        layer_costs.append((1.-cosine).mean())
        layer_cos.append(float(cosine.detach().mean()))
    hidden=torch.stack(layer_costs).mean()
    # Minimize BOTH real original transformer trajectory and original model's
    # actual next-token outputs. No custom expected answer lookup table.
    objective=3.*hidden+0.6*kl+0.12*ce
    if not torch.isfinite(objective):
        raise AssertionError("Oscar real original-pretrained FP32 full Green Tunnel loss nonfinite")
    return objective,{
       "original_pretrained_FP32_full_vocab_KL_T2":float(kl.detach()),
       "original_FP32_top1_cross_entropy":float(ce.detach()),
       "original_8decoder_layer_mean_cosine_loss":float(hidden.detach()),
       "actual_repaired_vs_FP32_cosine_by_decoder_layer":layer_cos,
    }


def export_runtime(out,cb61:bytes,trained_fp16:Path,receipt):
    target=out/"OSCAR_V1F_REAL_PAID_RUNTIME_SOURCE"
    target.mkdir(parents=True,exist_ok=True)
    core=target/"OSCAR_V1F_REAL_INDEPENDENT_42M_FP32_ENCODED_CORE.cb61"
    route=target/"OSCAR_V1F_FULL8_RECURRENT_GT_AND_32K_HEAD_FP16.safetensors"
    manifest=target/"OSCAR_V1F_RUNTIME_CONFIG.json"
    core.write_bytes(cb61)
    shutil.copyfile(trained_fp16,route)
    data={
      "Oscar_v1F":"real original independently pretrained TinyStories LLaMA 41.6896M",
      "original_HF_teacher_checkpoint_SHA256":SOURCE_SHA,
      "original_8layer_512hidden":True,
      "not_Calibri_12layer":True,
      "original_42M_true_unique_parameters":EXPECTED_ORIGINAL_UNIQUE,
      "original_student_core_format":"CB61 physically compressed K32V8 matrix source",
      "original_CB61_sha256":sha(cb61),
      "original_complete_green_tunnel_fp16_SHA256":sha(route.read_bytes()),
      "green_tunnel_registered_interlayer_decoder_hooks":DEPTH,
      "green_tunnel_hidden_rank":RANK,
      "extra_physically_paid_head_logit_correction_rank":HEAD_RANK,
      "input_original_trained_fp32_not_at_inference":True,
      "source_original_fp32_orig_weight_matrices_frozen":True,
      "original_tokenizer_PROXY_NOT_VERIFIED":True,
      "torch_decoder_FP32_materialization_Diagnostic_ONLY":True,
      "real_native_C_CPU_inference_NOT_tested":True,
      "clean_unseen_QA_not_tested":True,
      "no_12layer_CALIBRI_GreenTunnel_or_original_J7_claim":True,
      "full_experiment_trace_stored_OFFLINE_not_in_runtime":True,
      "same_geometry_8layer_original_pretrained_lossless_SKILL_NOT_PROVEN":True,
      "original_heldout_debug_first_match":receipt["eight_debug_after"]["real_fp32_teacher_first_match"],
      "original_heldout_debug_complete8_match":receipt["eight_debug_after"]["real_fp32_teacher_FULL_8token_match"],
      "model_not_deployed":True,
    }
    manifest.write_text(json.dumps(data,sort_keys=True,indent=2,
                                   ensure_ascii=False)+"\n",encoding="utf8")
    total=sum(p.stat().st_size for p in (core,route,manifest))
    physical={
       "original_cb61_full_42m_core_bytes":core.stat().st_size,
       "original_all8_recurrent_GT_and_head_FP16_bytes":route.stat().st_size,
       "actual_runtime_config_json_bytes":manifest.stat().st_size,
       "actual_complete_physical_source_plus_GT_head_manifest_bytes":total,
       "original_fp32_teacher_at_inference":False,
       "full_original_unique_pretrained_count":EXPECTED_ORIGINAL_UNIQUE,
       "full_correct_noninflated_bpp":8*total/EXPECTED_ORIGINAL_UNIQUE,
       "actual_0p7_full_source_plus_GT_head_physical_gate":
           total<=MAX_REAL_BYTES,
       "original_fp32_teacher_NOT_embedded":True,
       "inference_time_FP32_dense_diagnostic_decode_NOT_charged_or_CPU_proven":True,
       "original_training_source_tokenizer_not_in_package":True,
    }
    if total>MAX_REAL_BYTES:
        raise AssertionError("Oscar v1F REAL 0.7bpp full original pretrained source + full8 GT/head rate violated")
    return physical


def run(out:Path):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar v1F research result directory must be initially empty")
    if set(TRAIN_STORIES)&set(EVAL_OLD+EVAL_OTHER):
        raise AssertionError("Oscar v1F TRAIN accidentally duplicates old/new debugging prompts")
    eval_prompts=EVAL_OLD+EVAL_OTHER
    if len(TRAIN_STORIES)!=12 or len(eval_prompts)!=8 or len(set(eval_prompts))!=8:
        raise AssertionError("Oscar v1F precommitted TRAIN and public-debug counts changed")
    old_original_pt,source_state,audit=true_fp32_source(out)
    if (audit["archive_SHA256"]!=SOURCE_SHA or
        audit["pinned_exact_HF_revision_sha40"]!=SOURCE_REV or
        audit["model_parameters_real_unique"]!=EXPECTED_ORIGINAL_UNIQUE):
        raise AssertionError("Oscar v1F true externally pretrained original FP32 checkpoint provenance changed")
    revision=HfApi().model_info(TOKENIZER).sha
    if revision!=TOKENIZER_REV:
        raise AssertionError("Oscar v1F original external proxy tokenizer provenance changed")
    tok=AutoTokenizer.from_pretrained(TOKENIZER,revision=revision)
    if tok.vocab_size!=VOCAB or tok.eos_token_id is None:
        raise AssertionError("Oscar v1F genuine original pretrained 32k tokenizer violated")
    protocol={
      "stage":"OSCAR_V1F_REAL_ORIGINAL_PRETRAINED_FP32_ALL8_GREEN_TUNNEL_PRECOMMIT",
      "original_pretrained_FP32_repo":"karpathy/tinyllamas/stories42M.pt",
      "original_pretrained_FP32_exact_HF_revision":SOURCE_REV,
      "original_pretrained_FP32_checkpoint_SHA256":SOURCE_SHA,
      "real_original_pretrained_true_unique_parameters":EXPECTED_ORIGINAL_UNIQUE,
      "same_original_architecture_before_and_after":"8x512",
      "NOT_calibri_S142_12layer_320_hidden":True,
      "calibration_TRAIN_ONLY_contexts":list(TRAIN_STORIES),
      "prior_published_four_TEST_debug":list(EVAL_OLD),
      "additional_four_TEST_debug_NEVER_USED_TRAIN":list(EVAL_OTHER),
      "TRAIN_prompt_sha256":[sha(x.encode()) for x in TRAIN_STORIES],
      "EVAL_prompt_sha256":[sha(x.encode()) for x in eval_prompts],
      "steps":STEPS,
      "new_full8_recurrent_interlayer_GT_rank":RANK,
      "additional_tied_original_head_logit_correction_rank":HEAD_RANK,
      "all_original_fp32_and_new_CB59_original_source_weights_FROZEN":True,
      "full_8layer_hidden_state_F32_teacher_calibration":True,
      "teacher_full_next_vocab_KL_and_top1_CE_CALIBRATION":True,
      "no_target_leak_from_EVAL_to_training_or_checkpoint_selection":True,
      "original_tokenizer_proxy_equivalence_to_training_NOT_verified":True,
      "original_S142_trained_GT3_9_NOT_reused_for_wrong_geometry":True,
      "full_616_J7_QA_not_run":True,
      "no_LIVE_main_Render_Supabase_APK_NRA_ordinaryJessicaChat":True,
      "real_native_C_direct_lookup_not_tested":True,
      "full_original_fp32_teacher_not_in_model":True,
    }
    dump(out/"OSCAR_V1F_PRECOMMIT_BEFORE_MODEL_TRAINING.json",protocol)
    teacher=from_checkpoint(source_state)
    orig_count=sum(p.numel() for p in teacher.parameters())
    if (orig_count!=EXPECTED_ORIGINAL_UNIQUE or
        teacher.p.layers!=DEPTH or teacher.p.dim!=WIDTH or
        teacher.p.vocab!=VOCAB):
        raise AssertionError("Oscar original pretrained 42M source 8layer geometry changed")
    # Oscar genuine source-only COLD K32/V8 from independently trained FP32,
    # not any prior CALIBRI learned dictionary and not the original packed indices.
    original_core=encode_fp32(teacher.state_dict(),mode=MODE_COLD)
    authenticated=original_core.authenticated_decoder()
    if authenticated.meta["original_unique_parameter_count"]!=EXPECTED_ORIGINAL_UNIQUE:
        raise AssertionError("Oscar v1F missing independently pretrained original 42M FP32 source")
    raw_cb59=out/"OSCAR_V1F_ACTUAL_PRETRAINED_42M_SOURCE_ONLY_COLD.cb59"
    raw_cb59.write_bytes(original_core.blob)
    packed_cb61=to_cb61(original_core.blob)
    decoded=materialize_decoded_cb59(teacher,raw_cb59.read_bytes())
    decoded.eval().requires_grad_(False)
    fullgt=FullLayerGreenTunnel().install(decoded)
    head=PaidHeadCorrection()
    expected_zero=all(int(torch.count_nonzero(x))==0 for x in (
        fullgt.up,head.up))
    if not expected_zero:
        raise AssertionError("Oscar real interlayer correction must begin EXACT IDENTITY")
    # Measure eight public DEBUG prompts BEFORE any gradient updates, but
    # none is used for backprop or hyperparameter/model selection afterwards.
    before=paired_debug(teacher,decoded,fullgt,head,tok,eval_prompts,
                        "BEFORE_REAL_INDEPENDENT_FP32_RECURRENT_GT_ACTIVATION_TRAIN")
    train_targets=build_train_targets(teacher,tok)
    log("OSCAR_V1F_SOURCE_ALIGNED",
        external_unique=orig_count,
        original_pretrained_8x512=True,
        independently_cold_source_bytes=len(original_core.blob),
        actual_CB61_source_bytes=len(packed_cb61),
        no_older_calibri_prior=True,
        pretrain_probe_first=totals(before)["real_fp32_teacher_first_match"],
        pretrain_probe_full=totals(before)["real_fp32_teacher_FULL_8token_match"])
    optimizer=torch.optim.AdamW([
      {"params":[fullgt.up],"lr":0.0025},
      {"params":[fullgt.down],"lr":0.001},
      {"params":[fullgt.gain,fullgt.mix_logits],"lr":0.0007},
      {"params":[head.up],"lr":0.003},
      {"params":[head.down],"lr":0.001},
    ],weight_decay=0.)
    original_state_sha=sha(raw_cb59.read_bytes())
    saved_trace=[]
    previous_head_up=head.up.detach().clone()
    previous_gt_up=fullgt.up.detach().clone()
    start=time.monotonic()
    for step in range(STEPS):
        item=train_targets[step%len(train_targets)]
        optimizer.zero_grad(set_to_none=True)
        loss,details=real_train_step(decoded,fullgt,head,item)
        loss.backward()
        gradients={
          "all8_GT_UP_grad_norm":float(fullgt.up.grad.norm()),
          "paid_32k_head_UP_grad_norm":float(head.up.grad.norm()),
          "all8_GT_DOWN_grad_norm":float(fullgt.down.grad.norm()),
          "paid_32k_head_DOWN_grad_norm":float(head.down.grad.norm()),
        }
        if (step>=2 and
            (not all(v>0 for v in gradients.values())
             or not all(torch.isfinite(torch.tensor(v)) for v in gradients.values()))):
            raise AssertionError("Oscar v1F teacher original per-layer gradient failed to reach GT and head")
        torch.nn.utils.clip_grad_norm_(
            [fullgt.up,fullgt.down,fullgt.gain,fullgt.mix_logits,
             head.up,head.down],1.0,error_if_nonfinite=True)
        optimizer.step()
        trace={
          "actual_original_fp32_teacher_TRAIN_step":step+1,
          "original_train_prompt_sha256":item["original_prompt_SHA256"],
          "REAL_genuine_original_pretrained_FP32_step_loss":float(loss.detach()),
          **details,**gradients,
        }
        saved_trace.append(trace)
        if (step+1)%32==0:
            log("OSCAR_V1F_TRUE_ORIGINAL_42M_FP32_ALL8_STATE_CORRECTION_TRAIN",
                step=step+1,
                real_loss=round(float(loss.detach()),5),
                hidden_loss=round(details[
                    "original_8decoder_layer_mean_cosine_loss"],4),
                logit_KL_T2=round(details[
                    "original_pretrained_FP32_full_vocab_KL_T2"],4),
                GT_UP_grad=round(gradients["all8_GT_UP_grad_norm"],4),
                head_UP_grad=round(gradients["paid_32k_head_UP_grad_norm"],4),
                all_original_pretrained_source_frozen=True)
    if not torch.any(previous_gt_up!=fullgt.up) or not torch.any(
            previous_head_up!=head.up):
        raise AssertionError("Oscar v1F original pretrained independent teacher did not update both real mechanisms")
    paid=merge_paid(fullgt,head)
    if not all(torch.isfinite(t).all() for t in paid.values()):
        raise AssertionError("Oscar full8 GT physical FP16 tensor nonfinite")
    path=out/"OSCAR_V1F_TRAINED_REAL_42M_INDEPENDENT_FP32_FULL8_GT_R32_HEAD_R8_FP16.safetensors"
    save_file(paid,str(path))
    from_binary=load_file(str(path))
    if (set(from_binary)!=set(paid) or
        any(not torch.equal(from_binary[k],paid[k]) for k in paid)):
        raise AssertionError("Oscar v1F newly trained physical FP16 roundtrip failure")
    dump(out/"OSCAR_V1F_AUTHENTIC_GOLD_FP32_ALL8_REAL_TRAIN_STEP_TRACE.json",
         saved_trace)
    fullgt.uninstall()
    del decoded,fullgt,head
    gc.collect()
    if sha(raw_cb59.read_bytes())!=original_state_sha:
        raise AssertionError("Oscar v1F independently pretrained original cold CB59 was rewritten by train")
    new_reader=CB59(raw_cb59.read_bytes())
    if new_reader.meta["original_unique_parameter_count"]!=EXPECTED_ORIGINAL_UNIQUE:
        raise AssertionError("Physical original FP32 K32V8 CB59 reloaded incomplete")
    restored=materialize_decoded_cb59(teacher,raw_cb59.read_bytes())
    restored.eval().requires_grad_(False)
    gt=FullLayerGreenTunnel().install(restored)
    hd=PaidHeadCorrection()
    restore_paid(gt,hd,from_binary)
    try:
        after=paired_debug(teacher,restored,gt,hd,tok,eval_prompts,
                           "AFTER_ACTUAL_FP16_8LAYER_GT_HEAD_PHYSICAL_RELOAD")
        with hd.scope(False):
            gt_only=paired_debug(teacher,restored,gt,hd,tok,eval_prompts,
                                 "ABLATION_REAL_GT_ONLY_NO_EXTRA_HEAD")
        with gt.scope(False):
            head_only=paired_debug(teacher,restored,gt,hd,tok,eval_prompts,
                                   "ABLATION_REAL_PAID_HEAD_ONLY_NO_GT")
    finally:
        gt.uninstall()
    if (sha(path.read_bytes())!=sha(
            (out/"OSCAR_V1F_TRAINED_REAL_42M_INDEPENDENT_FP32_FULL8_GT_R32_HEAD_R8_FP16.safetensors").read_bytes())
        or sha(raw_cb59.read_bytes())!=original_state_sha):
        raise AssertionError("Original independent pretrained 42M source or physical route modified after evaluation")
    table={
      "eight_debug_original_cold_before":totals(before),
      "eight_debug_after":totals(after),
      "eight_debug_ablated_trained_GT_only":totals(gt_only),
      "eight_debug_ablated_trained_head_only":totals(head_only),
    }
    physical=export_runtime(out,packed_cb61,path,table)
    report={
      **protocol,
      "external_FP32_trained_full_unique_41p6896M":True,
      "real_original_teacher_FP32_not_Calibri_compressed_derived":True,
      "new_genuine_8layer_recurrent_state_carry_active_in_all8":True,
      "original_model_architecture_unchanged_8x512":True,
      "original_compressed_FP32_source_5bit_pretrained_SHA256":original_state_sha,
      "original_new_binary_CB61_SHA256":sha(packed_cb61),
      "new_full8_layer_GT_HEAD_FP16_SHA256":sha(path.read_bytes()),
      "original_teacher_final_FP32_memory_only":True,
      "real_original_FP32_teacher_gradient_optimizer_steps":STEPS,
      "actual_research_training_seconds":round(time.monotonic()-start,3),
      "old_cold_compressed_baseline_eight_debug":before,
      "new_reloaded_actual_full8_recurrent_GT_and_paid_head_eight_debug":after,
      "trained_GT_without_head_ablated_eight_debug":gt_only,
      "trained_head_without_GT_ablated_eight_debug":head_only,
      "metric_summaries":table,
      "real_physically_saved_all_sidecars_original_runtime":physical,
      "original_8layers_all_hidden_metrics_tested":True,
      "no_unseen_independent_QA_or_original_616_J7":True,
      "source_vocab_proxy_not_proven_training_equivalent":True,
      "CPU_direct_CB59_C_inference_90percent_NOT_measured":True,
      "12x320_Calibri_skill_conversion_not_claimed":True,
      "production_model_accepted":False,
    }
    dump(out/"OSCAR_V1F_FINAL_TRUE_EXTERNAL_42M_FP32_ALL8_GT_REPAIR_REPORT.json",
         report)
    log("OSCAR_V1F_FINAL_RECURRENT_ALL8_FP32_GREEN_TUNNEL",
        real_teacher_model_original_unique=EXPECTED_ORIGINAL_UNIQUE,
        REAL_cpu_steps=STEPS,
        actual_eight_debug_cold_first=table[
           "eight_debug_original_cold_before"]["real_fp32_teacher_first_match"],
        actual_eight_debug_new_first=table[
           "eight_debug_after"]["real_fp32_teacher_first_match"],
        original_debug_full8_before=table[
           "eight_debug_original_cold_before"]["real_fp32_teacher_FULL_8token_match"],
        original_debug_full8_after=table[
           "eight_debug_after"]["real_fp32_teacher_FULL_8token_match"],
        last_layer_cos_before=table[
           "eight_debug_original_cold_before"]["mean_final_decoder_state_cos"],
        last_layer_cos_after=table[
           "eight_debug_after"]["mean_final_decoder_state_cos"],
        original_trained_full8_interlayer_Calibri12_not_claimed=True,
        complete_inference_bytes=physical[
           "actual_complete_physical_source_plus_GT_head_manifest_bytes"],
        full_rate=physical["full_correct_noninflated_bpp"],
        original_independent_quality_PROVEN=False,
        no_S142_LIVE_NRA_or_616_J7=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    run(p.parse_args().out)
