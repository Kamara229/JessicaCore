#!/usr/bin/env python3
"""Oscar v1G: ACTUAL ON-POLICY original FP32 hidden-state Green Tunnel.

Original pretrained 41.6896M FP32: Karpathy TinyStories LLaMA 8x512.
Frozen K32V8 compressed 8x512 student, inherited trained FULL8 recurrent
rank32 GT-only checkpoint from real Oscar v1F. NO 32k learned head or hidden
expansion. Student's own greedy continuation supplies exact NEXT-TOKEN
conditioning context to independently pretrained teacher and differentiable
8-layer student. No S142 12x320 neural components mutated, no LIVE/J7/QA.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
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
from oscar.v1e.audit_checkpoint import source as pretrained_source
from oscar.v1e.safe_pretrained_llama import (
    from_checkpoint,materialize_decoded_cb59,
)
from oscar.v1f.green_tunnel import (
    DEPTH,WIDTH,RANK,FullLayerGreenTunnel,
)
from oscar.v1f.train_real_all8_green_tunnel import (
    SOURCE_SHA,SOURCE_REV,EXPECTED_ORIGINAL_UNIQUE,
    TOKENIZER,TOKENIZER_REV,TRAIN_STORIES,
    EVAL_OLD,EVAL_OTHER,prompt_ids,paired_debug,totals,
)

V1F_CB59_SHA="f7b5fe62edba2edfe9ef98b79536685f7d8182e9c5cd19e0b092ab15fc68f5f1"
V1F_CB61_SHA="724ac4086910859487772a78d7a264047ca1543ffa97af867477aec7857c372a"
V1F_GT_SHA="2fc4b5c989257b70a1d76a20c68eacbcf9be079d0add2912e684969926933543"
STEPS=160
MAX_NEW=6
DEBUG_NEW=(
    "The old farmer opened the gate to let the horses out.",
    "A small fox discovered a shiny key beside the river.",
    "When the rain stopped, the children ran to the playground.",
    "Mia put the cookies on the table for her family.",
)
TEMP=2.0
SEED=421977
LR_UP=.00045
LR_DOWN=.00015
LR_GAIN=.00008
LR_MIX=.00008
MAX_PAID=.7*EXPECTED_ORIGINAL_UNIQUE/8
ACTUAL_SHA_BYTES=(V1F_CB59_SHA,V1F_CB61_SHA,V1F_GT_SHA)


def sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(
        data,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")


def log(stage,**kwargs):
    print(json.dumps({"stage":stage,**kwargs},
                     ensure_ascii=False),flush=True)


def full8_student_logits(student,gt,ids):
    scores,_=student(ids)
    if (gt.last_coverage!=tuple(range(DEPTH)) or
        len(gt.latest_layers)!=DEPTH or
        gt.latest_norm is None or
        scores.shape!=(1,ids.shape[1],32000)):
        raise AssertionError("Oscar v1G full genuine 8layer GT causal path not executed")
    if not bool(torch.isfinite(scores).all()):
        raise AssertionError("Oscar v1G actual original pretrained compressed source logits nonfinite")
    return scores


@torch.inference_mode()
def greedy_student_rollout(student,gt,tok,prompt):
    """Collect real recursively sampled MODEL tokens with real recurrence.

    Tokens are generated from the live student; never replace them with
    original teacher targets, and stop when the student emits actual EOS.
    """
    current=prompt.clone()
    generated=[]
    for _ in range(MAX_NEW):
        scores=full8_student_logits(student,gt,current)
        choice=int(scores[0,-1].argmax())
        generated.append(choice)
        if choice==tok.eos_token_id:
            break
        current=torch.cat((current,current.new_tensor([[choice]])),dim=1)
    if not generated or not 1<=len(generated)<=MAX_NEW:
        raise AssertionError("Oscar genuine student policy didn't generate valid token context")
    return generated


def build_same_real_student_prefix(prompt,generated,tok):
    """All target prediction positions have ACTUAL student-generated history.

    E.g. prefix prompt + [A,B,C] scores A,B,C at positions
         [P-1,P,P+1] respectively; do not append the last predicted token.
    """
    if (not isinstance(generated,list) or
        not 1<=len(generated)<=MAX_NEW or
        any(type(z)!=int or not 0<=z<32000 for z in generated)):
        raise AssertionError("Oscar v1G real student policy tokens missing")
    if tok.eos_token_id in generated[:-1]:
        raise AssertionError("Oscar original student history continued AFTER actual EOS")
    if not generated[:-1]:
        return prompt.clone()
    ext=prompt.new_tensor([generated[:-1]])
    return torch.cat((prompt,ext),dim=1)


def posterior_objective(slogits,tlogits,student_hidden,teacher_hidden,
                        prompt_length,actual_student_tokens):
    """True original-FP32 teacher next-token and eight-layer state matching.

    Teacher and student see EXACTLY same student-produced prefix; no
    original teacher-greedy histories or reference answers inserted.
    Every precommitted token position contributes genuine full-32k
    original-FP32 KL, CE/margin, and actual all-eight hidden cosines.
    """
    n=len(actual_student_tokens)
    if (n<1 or n>MAX_NEW or
        slogits.shape!=tlogits.shape or
        slogits.shape[1]!=prompt_length+n-1 or
        slogits.shape[-1]!=32000 or
        len(student_hidden)!=DEPTH or len(teacher_hidden)!=DEPTH):
        raise AssertionError("Oscar original pretrained FP32 teacher alignment invalid")
    positions=torch.arange(prompt_length-1,prompt_length+n-1)
    s=slogits[0,positions,:].float()
    t=tlogits[0,positions,:].float().detach()
    actual=s.new_tensor(actual_student_tokens,dtype=torch.long)
    teacher_target=t.argmax(-1).long()
    weights=[]
    occurrences=Counter()
    for idx,target in enumerate(teacher_target.tolist()):
        occurrences[target]+=1
        weights.append((5.0 if idx==0 else 1.0)/
                       (occurrences[target]**.5))
    w=torch.tensor(weights,device=s.device,dtype=s.dtype)
    weights_norm=w/w.sum()
    logits_student=F.log_softmax(s/TEMP,dim=-1)
    log_teacher=F.log_softmax(t/TEMP,dim=-1)
    teacher_probs=log_teacher.exp()
    soft_KL=(teacher_probs*(log_teacher-logits_student)).sum(-1)*TEMP**2
    top1_ce=F.cross_entropy(s,teacher_target,reduction="none")
    # This directly pushes down a WRONG token that the student's *actual*
    # policy chose, without introducing a generic ban on repetition.
    wrong=actual!=teacher_target
    bad_vs_teacher=F.softplus(1.0+s.gather(-1,actual[:,None]).squeeze(-1)-
                             s.gather(-1,teacher_target[:,None]).squeeze(-1))
    wrong_margin=torch.where(wrong,bad_vs_teacher,torch.zeros_like(bad_vs_teacher))
    # Causal token-specific repetitions are penalized only when teacher
    # itself gives LESS probability than student at the same contexts.
    repeated=torch.zeros(n,dtype=torch.bool,device=s.device)
    for j in range(1,n):
        if actual_student_tokens[j]==actual_student_tokens[j-1]:
            repeated[j]=True
    p_s=F.softmax(s,dim=-1).gather(-1,actual[:,None]).squeeze(-1)
    p_t=F.softmax(t,dim=-1).gather(-1,actual[:,None]).squeeze(-1)
    repetition_penalty=torch.where(repeated,
                                    F.relu(p_s-p_t),torch.zeros_like(p_s))
    # Match ORIGINAL FP32 at each of its actual corresponding network
    # layers, including information that student passes to next layer.
    hidden_losses=[]
    hidden_cos=[]
    for i,(pred,ref) in enumerate(zip(student_hidden,teacher_hidden)):
        if pred.shape!=ref.shape or pred.shape[-1]!=WIDTH:
            raise AssertionError("True original pretrained FP32 8layer hidden geometry mismatch")
        x=pred[0,positions].float()
        y=ref[0,positions].float().detach()
        cos=F.cosine_similarity(
            F.layer_norm(x,(WIDTH,)),F.layer_norm(y,(WIDTH,)),dim=-1)
        hidden_losses.append(((1.-cos)*weights_norm).sum())
        hidden_cos.append(float(cos.mean().detach()))
    token_objective=((.70*soft_KL+.18*top1_ce+
                      .16*wrong_margin+.15*repetition_penalty)*
                     weights_norm).sum()
    hidden_objective=torch.stack(hidden_losses).mean()
    loss=token_objective+3.0*hidden_objective
    if not torch.isfinite(loss):
        raise AssertionError("Oscar real student-context per-layer teacher loss is nonfinite")
    info={
        "real_generated_student_ids":actual_student_tokens,
        "true_original_teacher_target_ids_on_student_own_prefix":teacher_target.tolist(),
        "current_student_teacher_target_rank":[
            int((row>row[k]).sum())+1 for row,k in zip(
                s.detach(),teacher_target)],
        "teacher_FP32_original_causal_next_logits_SHA256":[
            sha(row.detach().cpu().numpy().tobytes()) for row in t],
        "full_32k_source_teacher_mean_KL_T2":float(
            (soft_KL*weights_norm).sum().detach()),
        "teacher_target_CE":float((top1_ce*weights_norm).sum().detach()),
        "actual_student_wrong_token_penalty":float(
            (wrong_margin*weights_norm).sum().detach()),
        "genuine_teacher_excess_repetition_penalty":float(
            (repetition_penalty*weights_norm).sum().detach()),
        "original_FP32_eight_decoder_state_alignment_loss":float(
            hidden_objective.detach()),
        "original_FP32_eight_decoder_state_cosines":hidden_cos,
        "full_true_FP32_onpolicy_total_loss":float(loss.detach()),
        "original_pretrained_FP32_teacher_answers_NOT_used_as_history":True,
    }
    return loss,info


def repeated_fractions(rows):
    sample=[]
    for row in rows:
        ids=row["real_OSCAR_new_compressed_answer"]["ids"]
        if not ids:
            raise AssertionError("No actual greedy generated token IDs in Oscar diagnostic")
        sample.append({
            "real_token_repeat_fraction":float(
                (len(ids)-len(set(ids)))/len(ids)),
            "real_bigram_repeat_fraction":float(
                (max(0,len(ids)-1)-
                 len(set(tuple(ids[i:i+2]) for i in range(len(ids)-1))))/
                max(1,len(ids)-1)),
        })
    return {k:sum(item[k] for item in sample)/len(sample)
            for k in sample[0]}


def run(a):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar v1G output must be empty independent research directory")
    source_cb59=Path(a.existing_CB59)
    source_cb61=Path(a.existing_CB61)
    trained_gt=Path(a.existing_v1f_GT_FP16)
    for expected,path in [
            (V1F_CB59_SHA,source_cb59),
            (V1F_CB61_SHA,source_cb61),
            (V1F_GT_SHA,trained_gt)]:
        if not path.is_file() or sha(path.read_bytes())!=expected:
            raise AssertionError("Oscar v1G source/v1F trained GT incorrect physical SHA: "+str(path))
    if (len(TRAIN_STORIES)!=12 or len(EVAL_OLD)!=4 or
        len(EVAL_OTHER)!=4 or len(DEBUG_NEW)!=4 or
        set(TRAIN_STORIES)&set(EVAL_OLD+EVAL_OTHER+DEBUG_NEW) or
        len(set(EVAL_OLD+EVAL_OTHER+DEBUG_NEW))!=12):
        raise AssertionError("Oscar v1G independently preregistered TRAIN-vs-DEBUG set invalid")
    debug=EVAL_OLD+EVAL_OTHER+DEBUG_NEW
    precommit={
      "stage":"OSCAR_v1G_PRECOMMIT_REAL_INDEPENDENT_42M_PRETRAINED_FP32_ON_POLICY_GT8",
      "externally_pretrained_FP32_original":"karpathy/tinyllamas/stories42M.pt",
      "original_external_full_checkpoint_SHA256":SOURCE_SHA,
      "original_external_pretrained_HF_revision":SOURCE_REV,
      "all_original_unique_parameters":EXPECTED_ORIGINAL_UNIQUE,
      "source_architecture_8layers_width512_NOT_CALIBRI_12layer":True,
      "original_complete_CB59_SHA256":V1F_CB59_SHA,
      "original_complete_CB61_SHA256":V1F_CB61_SHA,
      "start_original_pretrained_v1F_GT_only_FP16_SHA256":V1F_GT_SHA,
      "uses_TRUE_original_42M_FP32_teacher_for_own_student_prefixes":True,
      "original_core_all_5bit_weights_frozen":True,
      "original_teacher_also_frozen":True,
      "train_only_12_contexts":list(TRAIN_STORIES),
      "DEBUG_original_8_not_trained":list(EVAL_OLD+EVAL_OTHER),
      "DEBUG_new_4_not_trained":list(DEBUG_NEW),
      "total_optimizer_steps":STEPS,
      "max_student_generated_prefix_predictions_per_step":MAX_NEW,
      "teacher_never_supplies_gold_token_history_for_current_student":True,
      "train_only_4_existing_physically_paid_GT_FP16_tensors":True,
      "extra_32k_head_added":False,
      "no_source_FP32_at_inference":True,
      "original_proxy_tokenizer_verification_UNPROVEN":True,
      "full_C_direct_decoder_CPU_speed_unmeasured":True,
      "original_616_CALIBRI_J7_unmodified_and_not_retested":True,
      "no_main_LIVE_NRA_Render_Supabase_APK_ordinaryJessicaChat":True,
      "not_independent_sealed_QA":True
    }
    write_json(out/"OSCAR_V1G_PRECOMMIT_BEFORE_REAL_MODEL_LOAD.json",precommit)
    fp32_path,source,audit=pretrained_source(out)
    if (audit["archive_SHA256"]!=SOURCE_SHA or
        audit["pinned_exact_HF_revision_sha40"]!=SOURCE_REV or
        audit["model_parameters_real_unique"]!=EXPECTED_ORIGINAL_UNIQUE):
        raise AssertionError("Oscar v1G independently pretrained FP32 original source untrusted")
    teacher=from_checkpoint(source).eval().requires_grad_(False)
    if sum(p.numel() for p in teacher.parameters())!=EXPECTED_ORIGINAL_UNIQUE:
        raise AssertionError("Original independent pretrained FP32 source tensor aliases not counted")
    tok=AutoTokenizer.from_pretrained(TOKENIZER,revision=TOKENIZER_REV)
    if tok.vocab_size!=32000 or tok.eos_token_id is None:
        raise AssertionError("Original pretrained TinyStories external proxy tokenization shape changed")
    compressed=CB59(source_cb59.read_bytes())
    if compressed.meta["original_unique_parameter_count"]!=EXPECTED_ORIGINAL_UNIQUE:
        raise AssertionError("Original independent 42M K32V8 cold source absent")
    student=materialize_decoded_cb59(teacher,source_cb59.read_bytes())
    student.eval().requires_grad_(False)
    green=FullLayerGreenTunnel().install(student)
    original_fp16=load_file(str(trained_gt))
    green.restore_paid(original_fp16)
    if int(torch.count_nonzero(green.up))==0:
        raise AssertionError("Oscar v1G must START from real trained original v1F FP16 GT")
    # No output-head corrector and no source-architecture substitution.
    from oscar.v1f.green_tunnel import PaidHeadCorrection
    zero=PaidHeadCorrection()
    zero.enabled=False
    baseline=paired_debug(teacher,student,green,zero,tok,debug,
          "V1G_BEFORE_actual_v1F_trained_GT_ONLY_no_32k_head")
    before_old=totals(baseline[:8])
    if before_old["real_fp32_teacher_first_match"]!=5:
        raise AssertionError("Oscar original v1F trained GT-only first 5/8 must exactly reproduce BEFORE additional optimization")
    # New FP32 teacher queries from current student's actual generated
    # continuations. Original 42M teacher context is NEVER its gold history.
    opt=torch.optim.Adam([
      {"params":[green.up],"lr":LR_UP},
      {"params":[green.down],"lr":LR_DOWN},
      {"params":[green.gain],"lr":LR_GAIN},
      {"params":[green.mix_logits],"lr":LR_MIX},
    ],weight_decay=0.)
    originals={k:v.clone() for k,v in green.source_paid_tensors().items()}
    steps=[]
    count_actual_student_prefixes=0
    different_from_teacher_prefix_contexts=0
    started=time.monotonic()
    try:
        for n in range(STEPS):
            case=n%len(TRAIN_STORIES)
            question=TRAIN_STORIES[case]
            prompt=prompt_ids(tok,question)
            actual=greedy_student_rollout(student,green,tok,prompt)
            history=build_same_real_student_prefix(prompt,actual,tok)
            with torch.inference_mode():
                tlog,th=teacher(history,hidden=True)
            if len(th)!=DEPTH+2:
                raise AssertionError("Oscar original full 8 FP32 teacher activations not captured")
            opt.zero_grad(set_to_none=True)
            slog=full8_student_logits(student,green,history)
            loss,parts=posterior_objective(
                slog,tlog,green.latest_layers,th[1:DEPTH+1],
                prompt.shape[1],actual)
            loss.backward()
            grads={
               "real_recurrent8_UP_grad":float(green.up.grad.norm()),
               "real_recurrent8_DOWN_grad":float(green.down.grad.norm()),
               "real_recurrent8_gain_grad":float(green.gain.grad.norm()),
               "real_recurrent8_mix_grad":float(green.mix_logits.grad.norm()),
            }
            if (not all(torch.isfinite(torch.tensor(v)) for v in grads.values())
                or grads["real_recurrent8_UP_grad"]<=0 or
                grads["real_recurrent8_DOWN_grad"]<=0):
                raise AssertionError("Oscar original FP32 teacher has no genuine trainable rank32 GT gradients")
            torch.nn.utils.clip_grad_norm_(
                [green.up,green.down,green.gain,green.mix_logits],
                0.8,error_if_nonfinite=True)
            opt.step()
            count_actual_student_prefixes+=len(actual)
            with torch.inference_mode():
                gold_first=int(tlog[0,prompt.shape[1]-1].argmax())
                mismatch=gold_first!=actual[0]
                different_from_teacher_prefix_contexts+=int(mismatch)
            steps.append({
              "real_FP32_teacher_on_live_student_policy_optimizer_step":n+1,
              "train_only_original_42M_prompt_sha256":sha(question.encode()),
              "genuine_student_original_5bit_GT8_generated_ids":actual,
              "same_original_teacher_and_student_context_sha256":sha(
                  history.detach().cpu().numpy().tobytes()),
              "true_teacher_first_vocabulary_target":gold_first,
              "current_student_FIRST_different_from_FP32_teacher":bool(mismatch),
              **parts,**grads,
            })
            if (n+1)%20==0:
                log("OSCAR_V1G_REAL_ORIGINAL_42M_FP32_ONPOLICY_INTERLAYER_TRAIN",
                    step=n+1,
                    real_FP32_teacher_student_prefix_count=count_actual_student_prefixes,
                    true_student_first_token_mismatches=different_from_teacher_prefix_contexts,
                    full_original_teacher_KL_T2=round(
                        parts["full_32k_source_teacher_mean_KL_T2"],4),
                    all8_original_hidden_loss=round(
                        parts["original_FP32_eight_decoder_state_alignment_loss"],4),
                    real_grad=round(grads["real_recurrent8_UP_grad"],4))
        updated=green.source_paid_tensors()
        physically_changed={
            k:int(torch.count_nonzero(v!=originals[k]))
            for k,v in updated.items()}
        if not all(n>0 for n in physically_changed.values()):
            raise AssertionError("Oscar v1G original FP32 trained FP16 four recurrent tensors did not actually change")
        paid=out/"OSCAR_V1G_REAL_42M_F32_ON_POLICY_GT_ONLY_R32_ALL8_FP16.safetensors"
        save_file(updated,str(paid))
        reloaded=load_file(str(paid))
        if (set(reloaded)!=set(updated) or
            any(not torch.equal(updated[k],reloaded[k]) for k in updated)):
            raise AssertionError("Oscar new true FP32 teacher trained full8 GT physical sidecar reload failed")
        write_json(out/"OSCAR_V1G_REAL_ORIGINAL42M_ONPOLICY_160_STEP_GRAD_TRACE.json",steps)
    finally:
        green.uninstall()
        del student,green,opt
        gc.collect()
    # All inference and first/complete answers must be regenerated from a
    # FRESH reconstructed CB59 student and physically reloaded FP16 weights.
    fresh=materialize_decoded_cb59(teacher,source_cb59.read_bytes())
    fresh.eval().requires_grad_(False)
    gt=FullLayerGreenTunnel().install(fresh)
    gt.restore_paid(reloaded)
    zero_head=PaidHeadCorrection()
    zero_head.enabled=False
    try:
        after=paired_debug(teacher,fresh,gt,zero_head,tok,debug,
            "V1G_AFTER_ACTUALLY_PAID_FP16_GT_ONLY_TRUE_ORIGINAL_FP32_STUDENT_PREFIX")
    finally:
        gt.uninstall()
    if (sha(source_cb59.read_bytes())!=V1F_CB59_SHA or
        sha(source_cb61.read_bytes())!=V1F_CB61_SHA or
        sha(trained_gt.read_bytes())!=V1F_GT_SHA):
        raise AssertionError("Oscar v1G mutated unmodifiable original 42M source or v1F trained GT")
    measures={
      "previous_v1F_original_GT_ONLY_old_eight":before_old,
      "previous_v1F_original_GT_ONLY_all_twelve":totals(baseline),
      "after_v1G_original_FP32_all_eight_old":totals(after[:8]),
      "after_v1G_original_FP32_all_twelve":totals(after),
      "before_old8_actual_repetitions":repeated_fractions(baseline[:8]),
      "after_old8_actual_repetitions":repeated_fractions(after[:8]),
      "before_all12_actual_repetitions":repeated_fractions(baseline),
      "after_all12_actual_repetitions":repeated_fractions(after),
    }
    # One real original CB61 source + ONE GT-only FP16 route + SMALL runtime
    # ABI manifest. No original FP32 teacher, no debug/training trace.
    target=out/"OSCAR_V1G_ORIGINAL_PRETRAINED_42M_REAL_PAID_RUNTIME_GT_ONLY"
    target.mkdir(parents=True,exist_ok=True)
    source_out=target/"ORIGINAL_42M_FP32_COLD_K32V8_OSCAR.cb61"
    gt_out=target/"OSCAR_V1G_REAL_FULL8_GT_R32_FP16.safetensors"
    manifest=target/"OSCAR_V1G_RUNTIME_MANIFEST.json"
    shutil.copyfile(source_cb61,source_out)
    shutil.copyfile(paid,gt_out)
    recipe={
       "stage":"Oscar v1G real full8 GT-trained original independently pretrained FP32 42M",
       "original_pretrained_weight_source_SHA256":SOURCE_SHA,
       "original_model_8layer_512_width_not_Calibri_12layer":True,
       "original_pretrained_unique_params":EXPECTED_ORIGINAL_UNIQUE,
       "original_42M_full_CB61_sha256":sha(source_out.read_bytes()),
       "true_student_onpolicy_8layer_GT_FP16_sha256":sha(gt_out.read_bytes()),
       "GT_original_full_decoder_coverage":DEPTH,
       "GT_original_hidden_rank":RANK,
       "extra_output_head_adapter":False,
       "real_FP32_teacher_is_NOT_needed_at_inference":True,
       "full_original_independent_teacher_fp32_TRAIN_labels_NOT_in_inference_package":True,
       "original_CB61_source_bytes":source_out.stat().st_size,
       "actual_all8_GT_only_new_FP16_bytes":gt_out.stat().st_size,
       "original_tokenizer_and_executable_decoder_NOT_included":True,
       "original_native_C_CPU_direct_lookup_NOT_speed_tested":True,
       "true_original_42M_full_sequence_quality_NOT_PROVEN":True,
       "no_new_independent_sealed_QA_evaluated":True,
       "original_616_J7_and_CALIBRI_S142_untouched":True,
       "old_eight_public_debug_first_token_matches":measures[
          "after_v1G_original_FP32_all_eight_old"]["real_fp32_teacher_first_match"],
       "old_eight_public_debug_full8_matches":measures[
          "after_v1G_original_FP32_all_eight_old"]["real_fp32_teacher_FULL_8token_match"],
       "new_debug12_original_FP32_full8_matches":measures[
          "after_v1G_original_FP32_all_twelve"]["real_fp32_teacher_FULL_8token_match"],
       "original_42M_production_model_accepted":False,
    }
    manifest_size=3072
    serialized=(json.dumps(recipe,sort_keys=True,indent=2,
                           ensure_ascii=False)+"\n").encode()
    if len(serialized)>manifest_size:
        raise AssertionError("Oscar v1G actual model manifest physical budget exceeded")
    manifest.write_bytes(serialized+b" "*(manifest_size-len(serialized)))
    bytes_total=sum(p.stat().st_size for p in (source_out,gt_out,manifest))
    rate=8*bytes_total/EXPECTED_ORIGINAL_UNIQUE
    if bytes_total>MAX_PAID:
        raise AssertionError("Oscar v1G independently pretrained 42M full source-plus-recurrent-GT fails 0.7 bpp rate")
    report={
      **precommit,
      "reloaded_ACTUAL_8layer_512_safetensors_FP16":True,
      "independently_pretrained_FP32_original_safetensors_not_shipped":True,
      "true_student_generated_own_prefix_teacher_target_total":count_actual_student_prefixes,
      "true_onpolicy_student_original_FP32_first_mismatch_training_events":
        different_from_teacher_prefix_contexts,
      "training_actual_160_steps":len(steps),
      "original_pretrained_42M_real_recurrent_FP16_changed_elements":
        physically_changed,
      "real_benchmark_original_FP32_vs_gt_only_before":baseline,
      "real_benchmark_original_FP32_vs_gt_only_after":after,
      "honest_repetition_and_teacher_metric_summaries":measures,
      "original_42M_model_real_inference_component_bytes":{
        "one_compressed_CB61_source":source_out.stat().st_size,
        "one_physically_trained_8layer_GT_only_FP16":gt_out.stat().st_size,
        "one_charged_runtime_manifest":manifest.stat().st_size,
        "ALL_source_plus_GT_manifest_bytes":bytes_total,
        "real_original_pretrained_41689600_unique_bpp":rate,
        "actual_0p7_bpp_source_GT_only_manifest_PASS":rate<=.7,
        "original_tokenizer_and_executable_binary_not_included":True},
      "original_FP32_model_RECONSTRUCTED_at_student_runtime":False,
      "one_compressed_source_only_no_duplicated_extra_32k_head":True,
      "no_original_Calibri_12x320_language_skill_conversion_claim":True,
      "real_native_C_CPU_90_NOT_tested":True,
      "original_616_J7_not_tested":True,
      "new_Calibri_43M_S142_unchanged":True,
      "model_accepted":False,
    }
    write_json(out/"OSCAR_V1G_FINAL_TRUE_FP32_42M_STUDENT_ONPOLICY_FULL8_GREEN_TUNNEL.json",report)
    log("OSCAR_V1G_FINAL_REAL_PRETRAINED42M_FP32_ON_POLICY_INTERLAYER",
        real_cpu_optimizer_steps=STEPS,
        real_original_42M_teacher_own_student_prefixes=count_actual_student_prefixes,
        old8_FP32_first_before=before_old["real_fp32_teacher_first_match"],
        old8_FP32_first_after=measures[
           "after_v1G_original_FP32_all_eight_old"]["real_fp32_teacher_first_match"],
        old8_FP32_full_before=before_old["real_fp32_teacher_FULL_8token_match"],
        old8_FP32_full_after=measures[
           "after_v1G_original_FP32_all_eight_old"]["real_fp32_teacher_FULL_8token_match"],
        new_total12_first=measures[
           "after_v1G_original_FP32_all_twelve"]["real_fp32_teacher_first_match"],
        new_total12_full=measures[
           "after_v1G_original_FP32_all_twelve"]["real_fp32_teacher_FULL_8token_match"],
        last_hidden_cos_before=round(before_old["mean_final_decoder_state_cos"],5),
        last_hidden_cos_after=round(measures[
           "after_v1G_original_FP32_all_eight_old"]["mean_final_decoder_state_cos"],5),
        true_source_GT_only_physical_bpp=rate,
        full_model_quality_ACCEPTED=False,
        no_LIVE_CALIBRI_S142_J7=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for n in ("existing_CB59","existing_CB61","existing_v1f_GT_FP16","out"):
        ap.add_argument(n,type=Path)
    run(ap.parse_args())
