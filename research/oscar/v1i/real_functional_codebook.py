#!/usr/bin/env python3
"""Oscar v1I: actually re-encode original pretrained 42M FP32 K32/V8 SOURCE.

Functional FP32 teacher supervision tunes EXISTING physically serialized
shared tok_embeddings=output 32k row scales and K32×8 centroids, holding
ALL original K32/V8 packed indices/other tensors and both ORIGINAL
trained 8-layer GT phase checkpoints fixed. Evaluate original independently
pretrained 42M FP32 genuine full free-run before vs physically NEW CB61 after.
NOT CALIBRI 12x320 and never LIVE/main or original 616 J7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import shutil
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file

from direct_codebook_v59 import CB59
from oscar.encoder import to_cb61
from oscar.v1e.audit_checkpoint import source as pretrained_source
from oscar.v1e.safe_pretrained_llama import from_checkpoint,materialize_decoded_cb59
from oscar.v1f.green_tunnel import FullLayerGreenTunnel
from oscar.v1f.train_real_all8_green_tunnel import (
    SOURCE_SHA,SOURCE_REV,TOKENIZER,TOKENIZER_REV,
    TRAIN_STORIES,EVAL_OLD,EVAL_OTHER,MAX_NEW,prompt_ids,true_greedy,
    EXPECTED_ORIGINAL_UNIQUE,
)
from oscar.v1g.real_onpolicy_gt8 import (
    V1F_CB59_SHA,V1F_CB61_SHA,V1F_GT_SHA,DEBUG_NEW,
)
from oscar.v1h.real_dual_phase_inference import (
    V1G_SHA,NEW_DEBUG,PHASES,dual_phase_greedy,
)
from oscar.v1i.tied_codebook import LearnableOriginalTiedEmbedding

SEED=421911
STEPS=48
TEMP=2.0
MAX_BPP=.7
ACTUAL_PHYSICAL_LIMIT=int(MAX_BPP*EXPECTED_ORIGINAL_UNIQUE/8)
MANIFEST_BYTES=3072
# Additional never previously evaluated public debug prompts. NOT sealed QA.
NEW_TEST_DEBUG=(
    "A friendly dolphin swam beside the little boat.",
    "The mother duck led her ducklings through the grass.",
    "When the old clock rang, Peter hurried downstairs.",
    "A purple flower grew beside the garden fence.",
    "The brave girl climbed the hill to find her brother.",
    "The puppy barked at a butterfly near the stream.",
    "The baker made warm bread for the hungry villagers.",
    "At sunset, the rabbit went back to its cozy home.",
)


def sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()


def dump(p,x):
    Path(p).write_text(json.dumps(x,sort_keys=True,indent=2,
                 ensure_ascii=False)+"\n",encoding="utf8")


def log(stage,**kwargs):
    print(json.dumps({"stage":stage,**kwargs},
                     ensure_ascii=False),flush=True)


def install_two_gt(model,path1,path2):
    old=FullLayerGreenTunnel().install(model)
    recent=FullLayerGreenTunnel().install(model)
    old.restore_paid(load_file(str(path1)))
    recent.restore_paid(load_file(str(path2)))
    old.requires_grad_(False)
    recent.requires_grad_(False)
    if torch.count_nonzero(old.up)==0 or torch.count_nonzero(recent.up)==0:
        raise AssertionError("Oscar v1I actual v1F/v1G two recurrent GT states not trained")
    return old,recent


@torch.inference_mode()
def evaluate_debug(teacher,student,old,new,tok,texts,cohort):
    rows=[]
    for i,text in enumerate(texts):
        ctx=prompt_ids(tok,text)
        fp32=true_greedy(teacher,tok,ctx)
        paired=dual_phase_greedy(
            student,teacher,old,new,tok,ctx,PHASES[2])
        ids=paired["generated_original_Oscar_token_ids"]
        if not ids:
            raise AssertionError("Oscar actual original 42M phase-routed decoder emitted no token")
        rows.append({
           "precommitted_debug":cohort,
           "debug_index":i+1,
           "prompt_SHA256":sha(text.encode()),
           "prompt":text,
           "real_original_independent_FP32_42M":fp32,
           "actual_compressed_phase_GT8":paired,
           "real_original_FP32_FIRST_match":ids[0]==fp32["ids"][0],
           "real_original_FP32_full8_plus_EOS_match":ids==fp32["ids"],
        })
        if (i+1)%8==0:
            log("OSCAR_V1I_ORIGINAL_42M_FP32_VS_TRUE_NEW_CB59_GREEDY_DEBUG",
                cohort=cohort,seen=i+1,
                first_matches=sum(int(r["real_original_FP32_FIRST_match"]) for r in rows),
                full_matches=sum(int(r["real_original_FP32_full8_plus_EOS_match"]) for r in rows))
    return rows


def scores(rows):
    return {
      "n":len(rows),
      "first_token_same_as_true_independent_FP32":sum(
           int(r["real_original_FP32_FIRST_match"]) for r in rows),
      "complete_real_greedy_EOS_same_as_FP32":sum(
           int(r["real_original_FP32_full8_plus_EOS_match"]) for r in rows),
      "mean_last_hidden_cosine_on_student_OWN_prefix":sum(
           r["actual_compressed_phase_GT8"][
             "mean_last_hidden_cosine_on_OWN_student_prefix"] for r in rows)/len(rows),
      "mean_token_repetition":sum(
           r["actual_compressed_phase_GT8"]["actual_repetition"]["unigram_repeat"]
           for r in rows)/len(rows),
      "mean_bigram_repetition":sum(
           r["actual_compressed_phase_GT8"]["actual_repetition"]["bigram_repeat"]
           for r in rows)/len(rows),
    }


def real_teacher_loss(student_logits,teacher_logits,student_gt_states,
                      reference_original_states):
    if (student_logits.shape!=teacher_logits.shape or
        student_logits.shape[-1]!=32000 or
        len(student_gt_states)!=8 or
        len(reference_original_states)!=8):
        raise AssertionError("Oscar v1I same-source functional teacher calibration invalid")
    a=student_logits[0,-1].float()
    b=teacher_logits[0,-1].float().detach()
    target=int(b.argmax())
    log_s=F.log_softmax(a/TEMP,dim=-1)
    log_t=F.log_softmax(b/TEMP,dim=-1)
    kl=(log_t.exp()*(log_t-log_s)).sum()*TEMP*TEMP
    ce=F.cross_entropy(a[None],torch.tensor([target],dtype=torch.long))
    wrong_margin=F.softplus(1.+a.max()-a[target])
    hcos=[]
    for pred,ref in zip(student_gt_states,reference_original_states):
        x=pred[0,-1].float()
        y=ref[0,-1].float().detach()
        if x.shape!=y.shape or x.numel()!=512:
            raise AssertionError("Oscar REAL original FP32 42M layerwise activation geometry drift")
        hcos.append(F.cosine_similarity(
            F.layer_norm(x[None],(512,)),
            F.layer_norm(y[None],(512,))).mean())
    hidden=(1.-torch.stack(hcos)).mean()
    loss=1.0*kl+0.16*ce+0.2*wrong_margin+2.5*hidden
    if not torch.isfinite(loss):
        raise AssertionError("Oscar v1I real independent FP32 teacher functional objective nonfinite")
    return loss,{
      "full32k_original_pretrained_FP32_next_KL_T2":float(kl.detach()),
      "original_pretrained_teacher_top1_CE":float(ce.detach()),
      "original_FP32_correct_margin_loss":float(wrong_margin.detach()),
      "independent_original_FP32_mean_8layer_hidden_cosine":
          float(torch.stack(hcos).detach().mean()),
      "teacher_original_FP32_top1":target,
      "student_original_5bit_source_first":int(a.argmax()),
    }


def run(a):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar v1I outputs must start in empty research directory")
    paths={
      "original_CB59":(Path(a.original_CB59),V1F_CB59_SHA),
      "original_CB61":(Path(a.original_CB61),V1F_CB61_SHA),
      "first_GT_v1F":(Path(a.v1f_first_GT_FP16),V1F_GT_SHA),
      "later_GT_v1G":(Path(a.v1g_later_GT_FP16),V1G_SHA)
    }
    for name,(p,h) in paths.items():
        if not p.is_file() or sha(p.read_bytes())!=h:
            raise AssertionError("Oscar v1I genuine original pre-trained source SHA/GT source mismatch "+name)
    original_prompts=EVAL_OLD+EVAL_OTHER+DEBUG_NEW+NEW_DEBUG
    debug=original_prompts+NEW_TEST_DEBUG
    if (len(TRAIN_STORIES)!=12 or len(debug)!=28 or
        len(set(debug))!=28 or
        set(TRAIN_STORIES)&set(debug)):
        raise AssertionError("Oscar v1I TRUE TRAIN and prior/new public debug cohorts not separated")
    precommit={
       "algorithm":"Oscar v1I physically re-encode ORIGINAL pretrained source K32/V8 tied embedding",
       "original_pretrained_FP32":"karpathy/tinyllamas/stories42M.pt",
       "original_pretrained_source_SHA256":SOURCE_SHA,
       "original_original_FP32_unique_trained_params":EXPECTED_ORIGINAL_UNIQUE,
       "original_original_FP32_8layers_512hidden_NOT_12layer_Calibri":True,
       "original_preexisting_CB59_SHA":V1F_CB59_SHA,
       "original_preexisting_CB61_SHA":V1F_CB61_SHA,
       "unchanged_trained_FIRST_full8_GT_FP16_SHA":V1F_GT_SHA,
       "unchanged_trained_LATER_full8_GT_FP16_SHA":V1G_SHA,
       "only_embedded_tied_original_32k_rowFP16_scales_and_K32V8_FP16_centroids_mutable":True,
       "original_5bit_indices_immutable":True,
       "all_other_original_PRETRAINED_42M_FP32_compressed_tensors_immutable":True,
       "training_precommitted_twelve_original_stories":list(TRAIN_STORIES),
       "old_twenty_public_debug_never_training":list(original_prompts),
       "NEW_eight_public_debug_never_training":list(NEW_TEST_DEBUG),
       "never_open_original_616_J7_or_clean_QA":True,
       "train_optimizer_steps":STEPS,
       "train_phases":"original v1F first next token; v1G second next token on current compressed model's first generated token",
       "original_full32k_teacher_and_original_eight_hidden_real_FP32":True,
       "original_proxy_tokenizer_training_equivalence_UNVERIFIED":True,
       "no_trainable_new_neural_sidecar":True,
       "both_original_full8_GT_FP16_frozen":True,
       "source_original_FP32_teacher_not_shipped":True,
       "same_K32V8_binary_length_and_alias_guaranteed":True,
       "full_source_plus_2GT_physically <=0p7_REQUIRED":True,
       "native_C_decoder_and_tokenizer_NOT_INCLUDED":True,
       "no_main_LIVE_Render_Supabase_NRA_APK_ordinary_Jessica_chat":True,
    }
    dump(out/"OSCAR_V1I_PRECOMMIT_BEFORE_TRUE_FP32_TEACHER.json",precommit)
    _,original_fp32,audit=pretrained_source(out)
    if (audit["archive_SHA256"]!=SOURCE_SHA or
        audit["pinned_exact_HF_revision_sha40"]!=SOURCE_REV or
        audit["model_parameters_real_unique"]!=EXPECTED_ORIGINAL_UNIQUE):
        raise AssertionError("Original independently pretrained original 42M checkpoint mismatch")
    teacher=from_checkpoint(original_fp32).eval().requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(TOKENIZER,revision=TOKENIZER_REV)
    if tok.vocab_size!=32000 or tok.eos_token_id is None:
        raise AssertionError("Original external prepaid model/32k tokenizer incompatible")
    raw=paths["original_CB59"][0].read_bytes()
    cold=CB59(raw)
    student=materialize_decoded_cb59(teacher,raw)
    student.eval().requires_grad_(False)
    first,other=install_two_gt(
        student,paths["first_GT_v1F"][0],paths["later_GT_v1G"][0])
    log("OSCAR_V1I_REAL_ORIGINAL_42M_FP32_COLD_SOURCE_START",
        true_unique=EXPECTED_ORIGINAL_UNIQUE,
        pretrained_original_CB59_physical=len(raw),
        true_pretrained_external_teacher_SHA256=SOURCE_SHA)
    baseline=evaluate_debug(teacher,student,first,other,tok,debug,
                             "BEFORE_original_v1H_tied32k_source_uncalibrated")
    if scores(baseline[:20])["first_token_same_as_true_independent_FP32"]!=11:
        raise AssertionError("Oscar v1I original v1H 11/20 authentic first-token pretest changed")
    correct_early=scores(baseline[:20])
    quant=LearnableOriginalTiedEmbedding(raw)
    real_old_weight=student.tok_embeddings.weight.detach().clone()
    if not torch.equal(quant.weight().detach(),real_old_weight):
        raise AssertionError("Oscar v1I rewritten original codebook does not bitwise match existing original model before training")
    quant.clear_cache()
    quant.install(student)
    for p in first.parameters():p.requires_grad_(False)
    for p in other.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW([
      {"params":[quant.center],"lr":.00055},
      {"params":[quant.log_gain],"lr":.0012},
    ],weight_decay=0.)
    train_records=[]
    start=time.monotonic()
    try:
        for step in range(STEPS):
            row=TRAIN_STORIES[step%len(TRAIN_STORIES)]
            prompt=prompt_ids(tok,row)
            with torch.no_grad():
                with first.scope(True),other.scope(False):
                    current_logits,_=student(prompt)
                student_first=int(current_logits[0,-1].argmax())
                if student_first==tok.eos_token_id:
                    second_ctx=None
                else:
                    second_ctx=torch.cat((
                        prompt,prompt.new_tensor([[student_first]])),dim=1)
                t0,h0=teacher(prompt,hidden=True)
                if second_ctx is not None:
                    t1,h1=teacher(second_ctx,hidden=True)
            optimizer.zero_grad(set_to_none=True)
            with first.scope(True),other.scope(False):
                slog0,_=student(prompt)
                a0,d0=real_teacher_loss(slog0,t0,other.latest_layers,
                                        h0[1:9])
            if second_ctx is not None:
                with first.scope(False),other.scope(True):
                    slog1,_=student(second_ctx)
                    a1,d1=real_teacher_loss(slog1,t1,other.latest_layers,
                                            h1[1:9])
            else:
                a1=quant.center.sum()*0.
                d1={"second_later_position_omitted_due_to_REAL_student_EOS":True}
            regularization=quant.original_anchor_loss()
            loss=1.7*a0+0.8*a1+regularization
            if not torch.isfinite(loss):
                raise AssertionError("Oscar v1I independently pretrained full source functional loss nonfinite")
            loss.backward()
            grads={
              "learned_existing_32x8_codebook_grad_norm":float(
                  quant.center.grad.norm()) if quant.center.grad is not None else 0.,
              "learned_existing_32k_FP16_row_log_gain_grad_norm":float(
                  quant.log_gain.grad.norm()) if quant.log_gain.grad is not None else 0.,
            }
            if not all(torch.isfinite(torch.tensor(v)) and v>0
                       for v in grads.values()):
                raise AssertionError("Oscar v1I original-source 5bit codebook/scale real teacher gradient zero/nonfinite")
            torch.nn.utils.clip_grad_norm_(
                [quant.center,quant.log_gain],1.0,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                quant.log_gain.clamp_(-.24,.24)
            train_records.append({
               "actual_training_step":step+1,
               "TRAIN_original_precommitted_story_SHA256":sha(row.encode()),
               "actual_current_compressed_original_student_first_token":student_first,
               "original_FP32_on_same_first_prompt":d0,
               "original_FP32_on_student_emitted_first_token_context":d1,
               "full_independent_original_FP32_both_phases_objective":float(loss.detach()),
               "original_K32V8_centroids_and_existing_FP16_scales_only":True,
               **grads,
            })
            if (step+1)%8==0:
                log("OSCAR_V1I_REAL_FUNCTIONAL_ORIGINAL_FP32_K32V8_TIED_CODEBOOK_TRAIN",
                    step=step+1,
                    original_pretrained_student_FIRST_KL=round(
                       d0["full32k_original_pretrained_FP32_next_KL_T2"],4),
                    original_pretrained_student_later_KL=round(
                       d1.get("full32k_original_pretrained_FP32_next_KL_T2",-1),4),
                    original_FP32_first_student_cos=round(
                       d0["independent_original_FP32_mean_8layer_hidden_cosine"],4),
                    gradients=grads)
        changed=quant.physically_changed_FP16_count()
        if (changed["original_embedded_fp16_row_scales_changed"]==0 or
            changed["original_embedded_fp16_codebook_centroids_changed"]==0):
            raise AssertionError("Oscar v1I functional training did not physically change original FP16 source")
    finally:
        quant.uninstall()
        first.uninstall()
        other.uninstall()
    # CRITICAL: put new source quantizer inside original physical CB59,
    # NOT as an unpaid separate torch/FP32 runtime postprocessing step.
    refined=quant.rewrite_actual_CB59_source()
    packed=to_cb61(refined)
    encoded_original=out/"OSCAR_V1I_TRULY_REFINED_ORIGINAL_PRETRAINED42M_K32V8.cb59"
    encoded_cb61=out/"OSCAR_V1I_NEW_TIED_EMBEDDING_CODEBOOK_ORIGINAL42M.cb61"
    encoded_original.write_bytes(refined)
    encoded_cb61.write_bytes(packed)
    if sha(refined)==V1F_CB59_SHA:
        raise AssertionError("Oscar v1I codebook FP16 changed yet encoded CB59 still source-identical")
    write_json= lambda name,value: dump(out/name,value)
    write_json("OSCAR_V1I_REAL_FP32_TEACHER_48_OPTIMIZER_STEPS.json",train_records)
    student_refined=materialize_decoded_cb59(
        teacher,encoded_original.read_bytes())
    student_refined.eval().requires_grad_(False)
    ref_gt0,ref_gt1=install_two_gt(
        student_refined,paths["first_GT_v1F"][0],paths["later_GT_v1G"][0])
    try:
        after=evaluate_debug(teacher,student_refined,ref_gt0,ref_gt1,tok,debug,
                        "AFTER_physical_new_K32V8_source_embedded_original_TIED_codebook")
    finally:
        ref_gt0.uninstall()
        ref_gt1.uninstall()
    for path,h in paths.values():
        if sha(path.read_bytes())!=h:
            raise AssertionError("Oscar v1I mutated ORIGINAL pinned 42M source or two GT sidecars")
    # physically paid inferable ONE new CB61 + exactly the same two old GTs
    # plus literal config manifest: no old CB59/FP32 teacher shipped.
    package=out/"OSCAR_V1I_PHYSICALLY_PAID_REAL42M_2GT_NEW_SOURCE"
    package.mkdir(parents=True,exist_ok=True)
    dest={
      "OSCAR_42M_ACTUAL_FUNCTIONALLY_REFINED_K32V8.cb61":encoded_cb61,
      "OSCAR_42M_FIRST_v1F_GreenTunnel_FP16.safetensors":
          paths["first_GT_v1F"][0],
      "OSCAR_42M_LATER_v1G_GreenTunnel_FP16.safetensors":
          paths["later_GT_v1G"][0],
    }
    for k,v in dest.items():shutil.copyfile(v,package/k)
    charged={
      k:{"physical_bytes":(package/k).stat().st_size,
         "sha256":sha((package/k).read_bytes())}
      for k in dest
    }
    total=sum(v["physical_bytes"] for v in charged.values())+MANIFEST_BYTES
    runtime={
       "algorithm":"Oscar v1I functional calibration of physical original tied K32V8 FP16 codebook",
       "original_independently_pretrained_TinyStories_42M":"karpathy/tinyllamas/stories42M.pt",
       "actual_unique_original_FP32_41689600":True,
       "original_model_8layer_width512_NOT_CALIBRI_12layer":True,
       "single_NEW_authenticated_5bit_CB61_source":True,
       "source_fp16_tied_embedding_32k_row_scales_adapted":True,
       "source_fp16_K32x8_centroids_adapted":True,
       "original_input_CB59_SHA256":V1F_CB59_SHA,
       "new_refined_CB59_SHA256":sha(refined),
       "new_refined_CB61_SHA256":sha(packed),
       "same_original_all_5bit_indices_and_other_matrix_weights":True,
       "original_source_tied_embedding_head_alias_preserved":True,
       "two_existing_pretrained_FP32_GreenTunnels_frozen":True,
       "original_FP32_teacher_not_in_student_runtime":True,
       "physical_files":charged,
       "manifest_charged_fixed_bytes":MANIFEST_BYTES,
       "actual_complete_new_source_plus_2GT_plus_manifest_bytes":total,
       "actual_original_41689600_noninflated_bpp":8*total/EXPECTED_ORIGINAL_UNIQUE,
       "real_0p7_bpp_requirement_pass":total<=ACTUAL_PHYSICAL_LIMIT,
       "first_token_PUBLIC_debug20_matches":scores(after[:20])[
           "first_token_same_as_true_independent_FP32"],
       "full_token_PUBLIC_debug20_matches":scores(after[:20])[
           "complete_real_greedy_EOS_same_as_FP32"],
       "model_not_accepted_or_deployed":True,
       "external_tokenizer_and_binary_not_in_physical_rate":True,
       "full_616_J7_native_CPU_and_clean_independent_QA_NOT_tested":True
    }
    blob=(json.dumps(runtime,sort_keys=True,indent=2,
                     ensure_ascii=False)+"\n").encode()
    if len(blob)>MANIFEST_BYTES:
        raise AssertionError("Oscar v1I physical K32/V8 runtime manifest reservation too small")
    manifest=package/"OSCAR_V1I_ACTUAL_SOURCE_CODEBOOK_BINARY_MANIFEST.json"
    manifest.write_bytes(blob+b" "*(MANIFEST_BYTES-len(blob)))
    if total!=sum(x.stat().st_size for x in package.iterdir()):
        raise AssertionError("Oscar original source+GT2 full physical 0.7 measure invalid")
    before_stats={
       "old_twenty_public_DEBUG":scores(baseline[:20]),
       "new_eight_PUBLIC_DEBUG_not_used_for_training":scores(baseline[20:]),
       "all_twenty_eight_PUBLIC_DEBUG":scores(baseline),
    }
    after_stats={
       "old_twenty_public_DEBUG":scores(after[:20]),
       "new_eight_PUBLIC_DEBUG_not_used_for_training":scores(after[20:]),
       "all_twenty_eight_PUBLIC_DEBUG":scores(after),
    }
    report={
      **precommit,
      "real_independent_FP32_original_checkpoint_retrieved_and_validated":True,
      "48_actual_FP32_teacher_supervised_optimizer_steps":len(train_records),
      "actually_existing_FP16_original_source_parameters_changed":changed,
      "old_original_input_CB59_compressed_unique_SHA256":V1F_CB59_SHA,
      "new_original_real_trained_CB59_SHA256":sha(refined),
      "new_physical_CB61_SHA256":sha(packed),
      "original_5bit_core_physical_CB59_bytes":len(raw),
      "new_original_5bit_source_physical_CB59_bytes":len(refined),
      "original_pretrained_fp32_teacher_not_in_new_runtime":True,
      "original_42M_SOURCE_reencoded_NOT_new_adapter_training":True,
      "physically_reconstructed_independent_model_after_training":True,
      "real_original_FP32_vs_twoGreenTunnels_BEFORE":baseline,
      "real_original_FP32_vs_twoGreenTunnels_AFTER":after,
      "predeclared_DEBUG_original_FP32_scores_BEFORE":before_stats,
      "predeclared_DEBUG_original_FP32_scores_AFTER":after_stats,
      "new_complete_source_with_2GT_actual_physical":runtime,
      "original_42M_genuine_pretrained_FP32_full_quality_not_guaranteed":True,
      "original_CALIBRI_12x320_S142_616J7_NOT_touched_or_tested":True,
      "model_accepted":False,
    }
    dump(out/"OSCAR_V1I_FINAL_REAL_FP32_FUNCTIONAL_TIED_CODEBOOK_REFINEMENT.json",
         report)
    log("OSCAR_V1I_FINAL_INDEPENDENT_ORIGINAL42M_FP32_FUNCTIONAL_CODEBOOK",
        real_FP32_teacher_training_steps=STEPS,
        original_tied_32k_row_scale_cells_changed=changed[
           "original_embedded_fp16_row_scales_changed"],
        original_codebook_fp16_cells_changed=changed[
           "original_embedded_fp16_codebook_centroids_changed"],
        old20_actual_FP32_FIRST_before=before_stats[
           "old_twenty_public_DEBUG"]["first_token_same_as_true_independent_FP32"],
        old20_actual_FP32_FIRST_after=after_stats[
           "old_twenty_public_DEBUG"]["first_token_same_as_true_independent_FP32"],
        old20_FP32_full_before=before_stats[
           "old_twenty_public_DEBUG"]["complete_real_greedy_EOS_same_as_FP32"],
        old20_FP32_full_after=after_stats[
           "old_twenty_public_DEBUG"]["complete_real_greedy_EOS_same_as_FP32"],
        NEW_debug8_FP32_first=after_stats[
           "new_eight_PUBLIC_DEBUG_not_used_for_training"]["first_token_same_as_true_independent_FP32"],
        NEW_debug8_FP32_full=after_stats[
           "new_eight_PUBLIC_DEBUG_not_used_for_training"]["complete_real_greedy_EOS_same_as_FP32"],
        physical_new_CB61_bytes=len(packed),
        new_original_pretrained_FP32_paired_rate=8*total/EXPECTED_ORIGINAL_UNIQUE,
        physically_pass_under0p7=total<=ACTUAL_PHYSICAL_LIMIT,
        independent_original_42M_recovered_language_NOT_proven=True,
        original_CALIBRI_S142_untouched=True)
    if total>ACTUAL_PHYSICAL_LIMIT:
        raise AssertionError("Oscar v1I new authenticated physically refined source + two GT sidecars exceeds strict 0.7 bpp; candidate rejected")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("original_CB59","original_CB61",
                 "v1f_first_GT_FP16","v1g_later_GT_FP16","out"):
        p.add_argument(name,type=Path)
    run(p.parse_args())
