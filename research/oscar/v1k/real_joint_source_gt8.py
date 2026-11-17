#!/usr/bin/env python3
"""OSCAR v1K: jointly train original PRETRAINED 42M 5bit source AND GT.

Physical original independently pretrained \`karpathy/tinyllamas/stories42M.pt\`
FP32 41,689,600 unique, 8 decoder layers × 512, is a frozen real teacher.
The model being compressed remains the SAME original geometry, NOT CALIBRI
S142 12×320. Train exactly four EXISTING K32V8 pretrained matrix centroid/
row scale tensors together with both existing v1F FIRST + v1G LATER rank32
recurrent Green Tunnel sets on 12 original TRAIN contexts; physically repack
the original authenticated CB59 and CB61, save and reload BOTH new FP16 GTs.
All old 5-bit packed indices, other source tensors/tied embedding immutable.
NO source FP32 / unpublished TEST labels at inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer

from direct_codebook_v59 import CB59
from oscar.encoder import to_cb61
from oscar.v1e.audit_checkpoint import source as real_fp32_source
from oscar.v1e.safe_pretrained_llama import (
    from_checkpoint, materialize_decoded_cb59,
)
from oscar.v1f.green_tunnel import FullLayerGreenTunnel
from oscar.v1f.train_real_all8_green_tunnel import (
    SOURCE_SHA,SOURCE_REV,TOKENIZER,TOKENIZER_REV,
    EXPECTED_ORIGINAL_UNIQUE,TRAIN_STORIES,
    EVAL_OLD,EVAL_OTHER,prompt_ids,true_greedy,
)
from oscar.v1g.real_onpolicy_gt8 import (
    V1F_CB59_SHA,V1F_CB61_SHA,V1F_GT_SHA,DEBUG_NEW,
)
from oscar.v1h.real_dual_phase_inference import (
    V1G_SHA,NEW_DEBUG,PHASES,dual_phase_greedy,
)
from oscar.v1i.real_functional_codebook import NEW_TEST_DEBUG
from oscar.v1k.joint_original_codebooks import JointOriginalSource,TARGETS

SEED=4219117
STEPS=72
TEMP=2.0
PHYSICAL_LIMIT=int(0.7*EXPECTED_ORIGINAL_UNIQUE/8)
PAID_MANIFEST=3072
# These never appear in TRAIN. They are PUBLIC debug, NOT clean QA.
EXTRA_DEBUG=(
    "One morning, the little deer walked through the meadow.",
    "Kate found a lost mitten beneath the wooden bench.",
    "The farmer planted tiny seeds before the warm rain.",
    "The baby bird waited quietly for its mother.",
    "A young girl drew a yellow sun on her paper.",
    "The little robot rolled across the school yard.",
    "An old man told stories beside the warm fireplace.",
    "The boy carried his blue backpack to the bus.",
)


def digest(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def write(path,x):
    Path(path).write_text(
        json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
        encoding="utf8")


def log(stage,**data):
    print(json.dumps({"stage":stage,**data},ensure_ascii=False),flush=True)


def init_gt(model,first_path,later_path):
    first=FullLayerGreenTunnel().install(model)
    later=FullLayerGreenTunnel().install(model)
    first.restore_paid(load_file(str(first_path)))
    later.restore_paid(load_file(str(later_path)))
    if (not torch.any(first.up) or not torch.any(later.up)):
        raise AssertionError("Oscar v1K original FULL8 trained FP16 GTs were never loaded")
    return first,later


@torch.inference_mode()
def actual_greedy_debug(teacher,student,gt0,gt1,tok,question,cohort):
    prompt=prompt_ids(tok,question)
    ref=true_greedy(teacher,tok,prompt)
    real=dual_phase_greedy(student,teacher,gt0,gt1,tok,prompt,PHASES[2])
    generated=real["generated_original_Oscar_token_ids"]
    return {
      "cohort":cohort,
      "original_PROMPT_sha256":digest(question.encode()),
      "debug_question":question,
      "actual_independent_pretrained_FP32_teacher_response":ref,
      "real_5bit_student_dual8_GT_response":real,
      "original_FP32_FIRST_greedy_token_exact":generated[0]==ref["ids"][0],
      "original_FP32_FULL_greedy_8token_and_EOS_exact":generated==ref["ids"],
    }


def grade(rows):
    return {
      "public_debug_n":len(rows),
      "real_FP32_first_token_exact":sum(int(
          r["original_FP32_FIRST_greedy_token_exact"]) for r in rows),
      "real_FP32_full_greedy8_EOS_exact":sum(int(
          r["original_FP32_FULL_greedy_8token_and_EOS_exact"]) for r in rows),
      "mean_last_hidden_cosine_on_same_student_generated_contexts":sum(
          r["real_5bit_student_dual8_GT_response"][
            "mean_last_hidden_cosine_on_OWN_student_prefix"]
          for r in rows)/len(rows),
      "mean_unigram_repetition":sum(
          r["real_5bit_student_dual8_GT_response"][
            "actual_repetition"]["unigram_repeat"]
          for r in rows)/len(rows),
      "mean_bigram_repetition":sum(
          r["real_5bit_student_dual8_GT_response"][
            "actual_repetition"]["bigram_repeat"]
          for r in rows)/len(rows),
    }


@torch.inference_mode()
def evaluate_public(teacher,student,gt0,gt1,tok,cohorts,name):
    rows=[]
    for group,items in cohorts.items():
        for phrase in items:
            rows.append(actual_greedy_debug(
                teacher,student,gt0,gt1,tok,phrase,group))
        log("OSCAR_V1K_ORIGINAL_FP32_VS_COLD_PACKED_AND_DUAL_GT_PUBLIC_DEBUG",
            phase=name,cohort=group,
            first=sum(int(r["original_FP32_FIRST_greedy_token_exact"])
                      for r in rows),
            full=sum(int(r["original_FP32_FULL_greedy_8token_and_EOS_exact"])
                     for r in rows),evaluated=len(rows))
    return rows


def real_fp32_functional_loss(logits,reference,student_hidden,teacher_hidden):
    s=logits[0,-1].float()
    t=reference[0,-1].float().detach()
    if s.shape!=(32000,) or t.shape!=s.shape or len(student_hidden)!=8 or len(teacher_hidden)!=8:
        raise AssertionError("Oscar v1K true FP32 teacher/student mismatch, not 8×512 original model")
    log_s=F.log_softmax(s/TEMP,dim=-1)
    log_t=F.log_softmax(t/TEMP,dim=-1)
    kl=(log_t.exp()*(log_t-log_s)).sum()*TEMP**2
    best=int(t.argmax())
    ce=F.cross_entropy(s[None],s.new_tensor([best],dtype=torch.long))
    competing=torch.cat((s[:best],s[best+1:])).max()
    margin=F.softplus(competing-s[best]+.6)
    hidden_cos=[]
    for h,r in zip(student_hidden,teacher_hidden):
        a=h[0,-1].float()
        b=r[0,-1].float().detach()
        if a.shape!=(512,) or b.shape!=a.shape:
            raise AssertionError("Original independent pretrained FP32 inner layer tensor wrong")
        hidden_cos.append(
            F.cosine_similarity(F.layer_norm(a[None],(512,)),
                                F.layer_norm(b[None],(512,))).mean())
    hidden=(1-torch.stack(hidden_cos)).mean()
    if not all(torch.isfinite(v) for v in (kl,ce,margin,hidden)):
        raise AssertionError("True pretrained 42M original FP32 teacher joint loss nonfinite")
    return 0.85*kl+0.18*ce+0.18*margin+2.4*hidden,{
       "original_full_vocab_32k_soft_KL_T2":float(kl.detach()),
       "original_teacher_top1_CE":float(ce.detach()),
       "original_teacher_top1_MARGIN_loss":float(margin.detach()),
       "original_eight_layer_hidden_alignment_loss":float(hidden.detach()),
       "independent_original_teacher_next_token_id":best,
       "current_student_next_token":int(s.detach().argmax()),
    }


def source_gt_anchor(curr,previous):
    # Anchor the live differentiable full-precision pretrained GT values,
    # never the intentionally detached source_paid_tensors() export.
    trainable={
        "oscar_v1f.down_fp16":curr.down,
        "oscar_v1f.up_fp16":curr.up,
        "oscar_v1f.gains_fp16":curr.gain,
        "oscar_v1f.mix_fp16":curr.mix_logits,
    }
    if set(trainable)!=set(previous):
        raise AssertionError("Missing pretrained original full8 GT FP16 physical anchors")
    return sum((val-previous[name].float()).square().mean()
               for name,val in trainable.items())/len(trainable)


def full_trained_GT_params(m):
    return [m.up,m.down,m.gain,m.mix_logits]


def run(a):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar v1K research outputs must begin empty")
    files={
       "old_source_CB59":(Path(a.original_CB59),V1F_CB59_SHA),
       "old_source_CB61":(Path(a.original_CB61),V1F_CB61_SHA),
       "old_original_trained_v1F_first_full8_GT":(Path(a.v1f_first_FP16),V1F_GT_SHA),
       "old_original_trained_v1G_later_full8_GT":(Path(a.v1g_later_FP16),V1G_SHA)
    }
    for label,(p,h) in files.items():
        if not p.is_file() or digest(p.read_bytes())!=h:
            raise AssertionError("Oscar v1K refuses invalid pinned ORIGINAL pretrained 42M source: "+label)
    cohorts={
        "old_v1H_public20":EVAL_OLD+EVAL_OTHER+DEBUG_NEW+NEW_DEBUG,
        "old_v1I_public8":NEW_TEST_DEBUG,
        "new_v1K_public8":EXTRA_DEBUG,
    }
    debug=tuple(x for values in cohorts.values() for x in values)
    if (len(TRAIN_STORIES)!=12 or len(debug)!=36 or len(set(debug))!=36 or
        set(TRAIN_STORIES)&set(debug) or len(TARGETS)!=4):
        raise AssertionError("Oscar precommitted TRAIN and public debug overlap")
    prior={
       "stage":"Oscar v1K original PRETRAINED 42M true JOINT 5bit source+GT8 calibration",
       "original_external_pretrained_TinyStories_checkpoint_sha256":SOURCE_SHA,
       "original_pretrained_unique_weights":EXPECTED_ORIGINAL_UNIQUE,
       "original_pretrained_network_8layer512_not_CALIBRI12x320":True,
       "original_CB59_5bit_SHA":V1F_CB59_SHA,
       "original_CB61_sha":V1F_CB61_SHA,
       "original_trained_v1F_first_GT8_SHA":V1F_GT_SHA,
       "original_trained_v1G_later_GT8_SHA":V1G_SHA,
       "exactly_selected_4_original_original_5bit_matrices_mutable":
          list(TARGETS),
       "all_existing_5bit_original_code_indices_AND_other_matrices_frozen":True,
       "actually_JOINT_8layer_GT_and_physical_source_codebook_optimization":True,
       "actual_original_pretrained_teacher_FP32_stop_gradient":True,
       "first_and_actual_student_generated_later_contexts":True,
       "TRAIN_12_known_story_prompts_only":list(TRAIN_STORIES),
       "all36_public_debug_NEVER_USED_FOR_TRAINING":cohorts,
       "optimizer_steps":STEPS,
       "target_max_fully_paid_rate_bpp":.7,
       "code_and_proxy_tokenizer_not_counted_in_weights_only_rate":True,
       "original_CALIBRI_S142_J7_NRA_and_LIVE_UNTOUCHED":True,
       "old_source_pretrained_FP32_teacher_NOT_shipped":True,
       "not_clean_heldout_QA":True,
    }
    write(out/"OSCAR_V1K_PRECOMMIT_BEFORE_TRUE_FP32_CHECKPOINT.json",prior)
    _,origin,audit=real_fp32_source(out)
    if (audit["archive_SHA256"]!=SOURCE_SHA or
        audit["pinned_exact_HF_revision_sha40"]!=SOURCE_REV or
        audit["model_parameters_real_unique"]!=EXPECTED_ORIGINAL_UNIQUE):
        raise AssertionError("Oscar true independent original 41.69M pretrained FP32 not authenticated")
    teacher=from_checkpoint(origin).eval().requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(TOKENIZER,revision=TOKENIZER_REV)
    if tok.vocab_size!=32000 or tok.eos_token_id is None:
        raise AssertionError("Oscar independently pretrained original 8layer 32k tokenizer changed")
    packed_original=files["old_source_CB59"][0].read_bytes()
    original_source=CB59(packed_original)
    if original_source.meta["original_unique_parameter_count"]!=EXPECTED_ORIGINAL_UNIQUE:
        raise AssertionError("Oscar v1K real true unique pretrained original 42M checkpoint count wrong")
    student=materialize_decoded_cb59(teacher,packed_original)
    student.eval().requires_grad_(False)
    first,later=init_gt(
        student,files["old_original_trained_v1F_first_full8_GT"][0],
        files["old_original_trained_v1G_later_full8_GT"][0])
    anchored1={k:v.detach().clone().float()
               for k,v in first.source_paid_tensors().items()}
    anchored2={k:v.detach().clone().float()
               for k,v in later.source_paid_tensors().items()}
    start=time.monotonic()
    # NO debug is used during train / optimizer / checkpoint decisions.
    baseline=evaluate_public(teacher,student,first,later,tok,cohorts,
                "BASELINE original physically packed 5bit+two pretrained GT8")
    base20=grade(baseline[:20])
    if base20["real_FP32_first_token_exact"]!=11 or base20["real_FP32_full_greedy8_EOS_exact"]!=0:
        raise AssertionError("Original v1H control no longer authentic 11/20 first,0/20 full")
    source=JointOriginalSource(packed_original)
    for name,mod in zip(source.names,source.layers):
        priorw=student.get_submodule(name.removesuffix(".weight")).weight.detach()
        if not torch.equal(mod.weight().detach(),priorw):
            raise AssertionError("Oscar v1K physical source layer BEFORE training is not exact original decoded 5bit FP32 geometry "+name)
    source.install(student)
    for param in full_trained_GT_params(first)+full_trained_GT_params(later):
        param.requires_grad_(True)
    optimizer=torch.optim.Adam([
       {"params":[x.center for x in source.layers],"lr":.00032},
       {"params":[x.log_scale for x in source.layers],"lr":.00055},
       {"params":[first.up],"lr":.00018},
       {"params":[first.down],"lr":.000055},
       {"params":[first.gain,first.mix_logits],"lr":.00004},
       {"params":[later.up],"lr":.00022},
       {"params":[later.down],"lr":.000075},
       {"params":[later.gain,later.mix_logits],"lr":.00005},
    ],weight_decay=0.)
    trace=[]
    try:
        for n in range(STEPS):
            text=TRAIN_STORIES[n%len(TRAIN_STORIES)]
            prompt=prompt_ids(tok,text)
            with torch.no_grad():
                with first.scope(True),later.scope(False):
                    student0,_=student(prompt)
                actual_first=int(student0[0,-1].argmax())
                if actual_first==tok.eos_token_id:
                    second=None
                    orig1=None
                    h1=None
                else:
                    second=torch.cat((prompt,prompt.new_tensor([[actual_first]])),dim=1)
                    orig1,h1=teacher(second,hidden=True)
                orig0,h0=teacher(prompt,hidden=True)
            optimizer.zero_grad(set_to_none=True)
            with first.scope(True),later.scope(False):
                logits0,_=student(prompt)
            loss0,metrics0=real_fp32_functional_loss(
                logits0,orig0,later.latest_layers,h0[1:9])
            if second is None:
                loss1=loss0.new_zeros(())
                metrics1={"actual_student_EOS_ends_later_context":True}
            else:
                with first.scope(False),later.scope(True):
                    logits1,_=student(second)
                loss1,metrics1=real_fp32_functional_loss(
                    logits1,orig1,later.latest_layers,h1[1:9])
            reg=(.05*source.original_anchoring_loss()+
                 .03*source_gt_anchor(first,anchored1)+
                 .03*source_gt_anchor(later,anchored2))
            loss=2.0*loss0+.85*loss1+reg
            if not torch.isfinite(loss):
                raise AssertionError("Oscar v1K real original 42M pretrained FP32 joint source+GT loss nonfinite")
            loss.backward()
            grads={
               "existing_original_5bit_centroids":sum(float(
                    x.center.grad.norm()) for x in source.layers),
               "existing_original_5bit_row_scales":sum(float(
                    x.log_scale.grad.norm()) for x in source.layers),
               "REAL_first_all8_GT_up":float(first.up.grad.norm()),
               "REAL_later_all8_GT_up":float(later.up.grad.norm()),
            }
            if not all(torch.isfinite(torch.tensor(v)) and v>0
                       for v in grads.values()):
                raise AssertionError("Oscar 42M independently pretrained FP32 teacher actual joint 5bit and both GT gradients zero")
            torch.nn.utils.clip_grad_norm_(
                list(source.parameters())+
                full_trained_GT_params(first)+full_trained_GT_params(later),
                1.0,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for q in source.layers:
                    q.log_scale.clamp_(-.15,.15)
            trace.append({
              "actual_optimizer_step":n+1,
              "TRAIN_original_12_prompts_only_SHA256":digest(text.encode()),
              "actual_student_generated_first_token":actual_first,
              "actual_true_original_pretrained_FP32_first":metrics0,
              "actual_true_original_pretrained_FP32_after_student_first":metrics1,
              "total_joint_source_and_both_GT_loss":float(loss.detach()),
              "trained_only_original_FP16_physically_charged_source_fields":True,
              **grads,
            })
            if (n+1)%12==0:
                log("OSCAR_V1K_TRUE_FP32_JOINT_K32V8_ORIGINAL_8LAYER_GT_TRAIN",
                    step=n+1,
                    real_TRAIN_prompt_id=n%len(TRAIN_STORIES),
                    original_FP32_FIRST_KL=round(metrics0[
                        "original_full_vocab_32k_soft_KL_T2"],4),
                    original_FP32_LATER_KL=round(metrics1.get(
                        "original_full_vocab_32k_soft_KL_T2",-1),4),
                    all_eight_TRUE_FP32_hidden_loss_first=round(metrics0[
                        "original_eight_layer_hidden_alignment_loss"],4),
                    gradients=grads)
    finally:
        source.uninstall()
        first.uninstall()
        later.uninstall()
    updated_source=source.rewrite_physical_CB59()
    changed=source.measure_changed_fp16()
    if not all(v["centroid_fp16_changed"]>0 and v["row_scale_fp16_changed"]>0
               for v in changed.values()):
        raise AssertionError("Oscar original 42M 5bit source four K32/V8 matrix elements not physically changed")
    new_5bit=out/"OSCAR_V1K_PHYSICALLY_REPACKED_ORIGINAL42M_4SOURCE.cb59"
    new_core=out/"OSCAR_V1K_PHYSICALLY_REPACKED_ORIGINAL42M_4SOURCE.cb61"
    new_5bit.write_bytes(updated_source)
    new_core.write_bytes(to_cb61(updated_source))
    first_file=out/"OSCAR_V1K_JOINT_TRAINED_FULL8_FIRST_GT_FP16.safetensors"
    later_file=out/"OSCAR_V1K_JOINT_TRAINED_FULL8_LATER_GT_FP16.safetensors"
    first_weights=first.source_paid_tensors()
    later_weights=later.source_paid_tensors()
    save_file(first_weights,str(first_file))
    save_file(later_weights,str(later_file))
    new_first=load_file(str(first_file))
    new_later=load_file(str(later_file))
    if (any(not torch.equal(new_first[k],first_weights[k])
            for k in first_weights) or
        any(not torch.equal(new_later[k],later_weights[k])
            for k in later_weights)):
        raise AssertionError("Oscar v1K full8 dual GT FP16 physical roundtrip failed")
    for a1,b1 in ((new_first,anchored1),(new_later,anchored2)):
        if not all(int(torch.count_nonzero(a1[k]!=b1[k].half()))>0 for k in b1):
            raise AssertionError("Oscar v1K joint optimization did NOT physically change every existing rank32 GT sidecar")
    write(out/"OSCAR_V1K_REAL_ORIGINAL_FP32_JOINT_TRAINING_72_STEPS.json",trace)
    # TRUE standalone physical reload: train adapter and model both released;
    # old MODEL FP32 diagnostic is NOT a source for the new final model.
    new_student=materialize_decoded_cb59(teacher,new_5bit.read_bytes())
    new_student.eval().requires_grad_(False)
    gt0,gt1=init_gt(new_student,first_file,later_file)
    try:
        after=evaluate_public(teacher,new_student,gt0,gt1,tok,cohorts,
            "AFTER_NEW_PAID_CB61_SOURCE_AND_BOTH_RELOADED_TRAINED_GT8")
    finally:
        gt0.uninstall()
        gt1.uninstall()
    for _,(p,hash_original) in files.items():
        if digest(p.read_bytes())!=hash_original:
            raise AssertionError("Oscar v1K modified original pretrained 5bit source or original trained GT instead of new copies")
    package=out/"OSCAR_V1K_REAL42M_NEW_5BIT_SOURCE_TWO_JOINT_GT_PAID_INFERENCE"
    package.mkdir(parents=True,exist_ok=True)
    names={
      "ORIGINAL_PRETRAINED42M_JOINT_REPACKED_K32V8.cb61":new_core,
      "ORIGINAL42M_FIRST_GT8_JOINT_FP16.safetensors":first_file,
      "ORIGINAL42M_LATER_GT8_JOINT_FP16.safetensors":later_file
    }
    for n,p in names.items():shutil.copyfile(p,package/n)
    paid={
      n:{"bytes":(package/n).stat().st_size,
         "sha256":digest((package/n).read_bytes())}
      for n in names
    }
    total=sum(x["bytes"] for x in paid.values())+PAID_MANIFEST
    if total>PHYSICAL_LIMIT:
        raise AssertionError("Oscar v1K FULL ACTUAL trained new 5bit original42M source + two GT sidecars exceeds 0.7 bpp")
    config={
       "stage":"Oscar v1K actual original42M independent FP32 functional joint CB61+two GT",
       "original_real_unique_pretrained_params":EXPECTED_ORIGINAL_UNIQUE,
       "8_original_decoder_layers_x_512_hidden_not_CALIBRI12":True,
       "existing_packed_5bit_codes_and_other_matrices_unchanged":True,
       "joint_fp32_training_only_4_existing_centroid_scale_matrices":list(TARGETS),
       "two_physically_stored_updated_real_interlayer_rank32_GT8_FP16":True,
       "original_FP32_teacher_not_at_inference":True,
       "original_source_tied_input_output_not_updated":True,
       "full_real_inference_files":paid,
       "physically_paid_manifest_bytes":PAID_MANIFEST,
       "real_total_source_and_GT_binaries_bytes":total,
       "real_source_plus_GT_bpp":8*total/EXPECTED_ORIGINAL_UNIQUE,
       "all_0p7_source_model_rate_pass":total<=PHYSICAL_LIMIT,
       "original_pretrained_FP32_full_greedy_PUBLIC_debug36_matches":grade(after)[
           "real_FP32_full_greedy8_EOS_exact"],
       "unseen_clean_QA_and_original_616J7_not_replayed":True,
       "original_tokenizer_and_C_native_runtime_not_in_source_byte_budget":True,
       "original_CALIBRI_S142_LIVE_NRA2_untouched":True,
       "model_not_accepted_or_deployed":True,
    }
    encoded=(json.dumps(config,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    if len(encoded)>PAID_MANIFEST:
        raise AssertionError("Oscar literal physically paid runtime config exceeded reservation")
    (package/"OSCAR_V1K_PAID_FUNCTIONAL_MODEL_MANIFEST.json").write_bytes(
        encoded+b" "*(PAID_MANIFEST-len(encoded)))
    if total!=sum(f.stat().st_size for f in package.iterdir()):
        raise AssertionError("Oscar v1K real model byte accounting dishonest")
    grades={
      "old20_before":grade(baseline[:20]),
      "old20_after":grade(after[:20]),
      "old28_before":grade(baseline[:28]),
      "old28_after":grade(after[:28]),
      "NEW_8_public_DEBUG_before":grade(baseline[28:]),
      "NEW_8_public_DEBUG_after":grade(after[28:]),
      "all36_before":grade(baseline),
      "all36_after":grade(after)
    }
    report={
      **prior,
      "real_external_original_pretrained_42M_FP32_teacher_validated":True,
      "original_SOURCE_K32V8_4_matrices_physical_FP16_changed":changed,
      "new_physically_authenticated_CB59_SHA256":digest(updated_source),
      "new_physically_authenticated_CB61_SHA256":digest(new_core.read_bytes()),
      "physically_training_ALL_original_rank32_GT8":True,
      "new_FIRST_full8_GT_FP16_SHA256":digest(first_file.read_bytes()),
      "new_LATER_full8_GT_FP16_SHA256":digest(later_file.read_bytes()),
      "original_all_5bit_indices_untouched":True,
      "source_new_only_original4_centroids_and_row_scales":True,
      "all_model_and_GT_FP16_physically_reloaded_INDEPENDENTLY":True,
      "real_PRETRAINED_FP32_vs_ORIGINAL_5bit_source_PLUS_GT_DEBUG_BEFORE":baseline,
      "real_PRETRAINED_FP32_vs_NEW_5bit_source_PLUS_GT_DEBUG_AFTER":after,
      "original_FP32_full_responses_and_hidden_metrics":grades,
      "actual_physically_complete_original42M_source_plus_bothGT":config,
      "entire_research_optimizer_trace_offline_not_in_inference":True,
      "original_S142_CALIBRI_12layer_transfer_NOT_COMPLETED":True,
      "independent_clean_QA_and_full_616_J7_NOT_TESTED":True,
      "model_accepted":False,
    }
    write(out/"OSCAR_V1K_FINAL_TRUE_PRETRAINED42M_FP32_JOINT_INTERLAYER_SOURCE_GT_REPORT.json",report)
    log("OSCAR_V1K_FINAL_REAL_ORIGINAL_FP32_SOURCE_AND_DUAL_GT_JOINT_TRAIN",
        true_original_pretrained_unique=EXPECTED_ORIGINAL_UNIQUE,
        real_steps=STEPS,
        real_5bit_original_matrices_jointly_calibrated=4,
        original_first_GT_and_later_GT_physically_retrained=True,
        old20_first_before=grades["old20_before"]["real_FP32_first_token_exact"],
        old20_first_after=grades["old20_after"]["real_FP32_first_token_exact"],
        old20_full_before=grades["old20_before"]["real_FP32_full_greedy8_EOS_exact"],
        old20_full_after=grades["old20_after"]["real_FP32_full_greedy8_EOS_exact"],
        new8_first_before=grades["NEW_8_public_DEBUG_before"]["real_FP32_first_token_exact"],
        new8_first_after=grades["NEW_8_public_DEBUG_after"]["real_FP32_first_token_exact"],
        all36_first_before=grades["all36_before"]["real_FP32_first_token_exact"],
        all36_first_after=grades["all36_after"]["real_FP32_first_token_exact"],
        all36_FULL_before=grades["all36_before"]["real_FP32_full_greedy8_EOS_exact"],
        all36_FULL_after=grades["all36_after"]["real_FP32_full_greedy8_EOS_exact"],
        old20_hidden_after=round(grades["old20_after"][
            "mean_last_hidden_cosine_on_same_student_generated_contexts"],5),
        original_model_source_plus2GT_real_paid_bytes=total,
        strict_original41689600_bpp=8*total/EXPECTED_ORIGINAL_UNIQUE,
        no_Calibri_S142_LIVE_or_original_616J7_changes=True,
        accepted_for_deployment=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("original_CB59","original_CB61",
                 "v1f_first_FP16","v1g_later_FP16","out"):
        parser.add_argument(name,type=Path)
    run(parser.parse_args())
