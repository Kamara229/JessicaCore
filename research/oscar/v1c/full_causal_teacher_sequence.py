#!/usr/bin/env python3
"""Oscar v1C: REAL full-sequence original pretrained 135M FP32 teacher transfer.

One actual 43M CALIBRI original trained 12layer source+GT3/9+Q2 GT12 and
an independently trained Oscar v1B isolated rank8 route, OFF by default.
Continue from the exactly authenticated v1B physical FP16 checkpoint.
Train only the Oscar teacher-specific route on all eight genuine original
135M next-token logits through real student incremental causal KV; no
synthetic EOS if original teacher did not emit one. Research-only, no
NRA, no LIVE, no untouched QA or complete 616 J7 assertions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s7a_43m_fp32_transfer.actual_43m_fp32_conversion import (
    Stack, validate_parent, selected_S93_existing_train, BASE_36M,
    TOTAL_43M, dump, digest, log,
)
from calibri2.s127.q2_prefill_gate_control import prefill_only_tunnel
from oscar.encoder import MODE_LOCKED, encode_fp32, to_cb61
from oscar.v1.independent_fp32_teacher_distill import (
    ids_for, score_teacher_vs_student,
)
from oscar.v1b.teacher_route_isolated import (
    OscarTeacherRoute, physical_package, PREFIX, RANK, LAYERS, WIDTH,
    RETAINED_S142_EXPECTED_CORRECT_OLD8,
)
from v176.operators import clear

V1B_SHA="11f79c0e9fb4f67e92149a8bdaf083f65a2c135168b557c6c8fb6df0d97aeaee"
V1B_SOURCE_BYTES=123080
STEPS=64
TRAIN=4
CHECK=8
MAX_NEW=8
TEMP=2.0
LR_UP=.0006
LR_DOWN=.00025
SEED=431917
FOCUS_LATE=1.7
EOS_WEIGHT=.12
EOS_MARGIN=1.0


@torch.inference_mode()
def capture_teacher_trajectory(teacher,tok,row):
    """Teacher-generated original *causal next-token* all-vocabulary targets.

    Capture original FP32 entire next distribution at each actual chosen
    position using exactly its greedy history, never original S142 labels.
    If 8 tokens elapse without EOS, do not invent EOS as a target.
    """
    context=ids_for(tok,row["question"])
    actual=context
    emissions=[]
    references=[]
    ends=False
    for step in range(MAX_NEW):
        logits=teacher(input_ids=actual,use_cache=False).logits[0,-1].float().cpu()
        if logits.shape!=(tok.vocab_size,) or not torch.isfinite(logits).all():
            raise AssertionError("Oscar v1C teacher original pretrained FP32 logits invalid")
        chosen=int(logits.argmax())
        emissions.append(chosen)
        references.append(logits.detach().clone())
        if chosen==tok.eos_token_id:
            ends=True
            break
        actual=torch.cat((actual,actual.new_tensor([[chosen]])),dim=-1)
    return {
        "source_original_S142_TRAIN_id":row["TRAIN_id"],
        "original_PROMPT_sha256":digest(row["question"].encode()),
        "teacher_full_token_ids_including_actual_EOS":emissions,
        "teacher_generated_valid_EOS":ends,
        "teacher_text":tok.decode(emissions,skip_special_tokens=True),
        "teacher_real_full_vocab_token_distribution_SHA256":[
            hashlib.sha256(r.numpy().tobytes()).hexdigest() for r in references],
        "teacher_training_distributions":references,
    }


def student_causal_kv_teacher_forcing(student,route,tok,row,targets):
    """Differentiable actual autoregressive KV, not one-shot fake prefill.

    The original S142 full12 Q2 and progressive core update only on the
    initial input as their real S101 inference path; subsequent generated
    tokens use each real prior teacher token with the source KV cache. The
    *separate* new Oscar route covers all 12 layers at EVERY output step.
    Only emitted tokens from the independently trained teacher are forced.
    """
    if not 1<=len(targets)<=MAX_NEW:
        raise AssertionError("Oscar true FP32 target size changed")
    ctx=ids_for(tok,row["question"])
    logits=[]
    past=None
    with student.progressive.prefill_only(student.model),\
         prefill_only_tunnel(student.model,student.full),\
         student.bridge.prompt_scope("S93",row["feature"]),\
         route.teacher_mode():
        for step,target in enumerate(targets):
            inp=ctx if step==0 else ctx.new_tensor([[targets[step-1]]])
            clear(student.sources)
            result=student.model(
                input_ids=inp,past_key_values=past,
                use_cache=True,output_hidden_states=True)
            past=result.past_key_values
            if past is None:
                raise AssertionError("Oscar v1C real original CALIBRI KV cache missing")
            hidden=result.hidden_states[-1][0,-1]
            prediction=result.logits[0,-1]+F.linear(
                student.first(hidden) if step==0 else student.later(hidden),
                student.head)
            if (prediction.shape!=(tok.vocab_size,) or
                not torch.isfinite(prediction).all()):
                raise AssertionError("Oscar v1C physically paid FP32 teacher causal logits invalid")
            logits.append(prediction)
        if (student.full.last_forward_coverage!=tuple(range(LAYERS)) or
            student.progressive.last_prefill_coverage!=tuple(range(LAYERS)) or
            route._last_visit!=tuple(range(LAYERS))):
            raise AssertionError("Oscar v1C actual original 12 layer GT / teacher branch all missing")
    clear(student.sources)
    return logits


def sequence_loss(student_outputs,teacher_outputs,target_tokens,eos):
    if len(student_outputs)!=len(teacher_outputs) or len(target_tokens)!=len(teacher_outputs):
        raise AssertionError("Oscar v1C training sequence/teacher-logit lengths drifted")
    terms=[]
    telemetry=[]
    for idx,(student,teacher,target) in enumerate(zip(
            student_outputs,teacher_outputs,target_tokens)):
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise AssertionError("Non-finite original pretrained whole sequence")
        # Original teacher full-vocabulary KL + supervised teacher's own
        # GREEDY (not hand-invented) next token.
        teacher_dist=F.softmax(teacher/TEMP,dim=-1)
        kl=F.kl_div(F.log_softmax(student[None]/TEMP,-1),
                    teacher_dist[None],reduction="batchmean")*TEMP**2
        ce=F.cross_entropy(student[None],
                           student.new_tensor([target],dtype=torch.long))
        late_weight=FOCUS_LATE if idx>0 else 1.
        if target!=eos:
            eos_gap=(student[eos]-student[target]+EOS_MARGIN)/4.
            premature=F.softplus(eos_gap)*4.
        else:
            premature=student.new_zeros(())
        loss=late_weight*(kl+.20*ce+EOS_WEIGHT*premature)
        terms.append(loss)
        telemetry.append({
          "actual_next_position":idx,
          "teacher_original_F32_id":target,
          "student_teacherforced_argmax":int(student.detach().argmax()),
          "FP32_teacher_soft_KL":float(kl.detach()),
          "teacher_greedy_cross_entropy":float(ce.detach()),
          "wrong_premature_EOS_margin_penalty":float(premature.detach()),
          "teacher_EOS_observed":target==eos
        })
    return torch.stack(terms).mean(),telemetry


@torch.inference_mode()
def record_actual_student_on_teacher_trajectory(student,route,tok,row,
                                                teacher_targets,teacher_logits):
    """Record teacher-forced ranking independently of greedy S98 generation."""
    predictions=student_causal_kv_teacher_forcing(
        student,route,tok,row,teacher_targets)
    records=[]
    for idx,(student_logits,teacher_id,teacher_logit) in enumerate(zip(
            predictions,teacher_targets,teacher_logits)):
        records.append({
          "idx":idx,"teacher_target":int(teacher_id),
          "student_teacherforced_id":int(student_logits.argmax()),
          "teacher_target_rank_in_student":
             int((student_logits>student_logits[int(teacher_id)]).sum())+1,
          "KL_FULL_VOCAB_T2":float(F.kl_div(
             F.log_softmax(student_logits[None]/TEMP,-1),
             F.softmax(teacher_logit[None]/TEMP,-1),
             reduction="batchmean")*TEMP*TEMP),
        })
    return records


def run(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if out.iterdir():
        raise FileExistsError("Oscar v1C results must start in empty immutable directory")
    old_v1b=Path(args.V1B_TRAINED_FP16)
    raw_v1b=old_v1b.read_bytes()
    if digest(raw_v1b)!=V1B_SHA or len(raw_v1b)!=V1B_SOURCE_BYTES:
        raise AssertionError("Oscar v1C must start EXACT known trained FP16 v1B checkpoint")
    original_cb,old_core,old_s93,old_s142,bank,router,copy=validate_parent(args)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    rows=selected_S93_existing_train(tok,router)
    if len(rows)!=CHECK:
        raise AssertionError("Oscar v1C exact original S142 known TRAIN cohort changed")
    pre={
      "stage":"OSCAR_V1C_PRECOMMIT_BEFORE_REAL_PRETRAINED_FP32_TEACHER_LOAD",
      "independent_pretrained_135M_FP32_name":MODEL,
      "pinned_pretrained_135M_FP32_sha":MODEL_SHA,
      "actual_teacher_30layers_576width_134515008":True,
      "actual_student_12layers_320width_43m":True,
      "actual_original_43M_SHA":digest(args.S142_FP16.read_bytes()),
      "original_v1b_independent_teacher_route_FP16_SHA":V1B_SHA,
      "TRAIN_ONLY_four":[x["TRAIN_id"] for x in rows[:TRAIN]],
      "TRAIN_DEBUG_ONLY_four":[x["TRAIN_id"] for x in rows[TRAIN:]],
      "train_causal_KV_steps":STEPS,
      "real_teacher_next_tokens_per_case_max":MAX_NEW,
      "no_invented_EOS_labels_when_teacher_no_EOS":True,
      "all_real_originalS142_GT3_9_Q2_full12_unchanged":True,
      "all_trainable_original_S142_43M_immutable":True,
      "all_training_gradients_ONLY_new_separate_12layer_rank8_Oscar_route":True,
      "real_native_KV_teacher_forcing_not_pure_first_token":True,
      "no_QA_EVAL_or_original_616_J7_replay":True,
    }
    dump(out/"OSCAR_V1C_PRECOMMIT_BEFORE_TEACHER.json",pre)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
        trust_remote_code=False).cpu().eval().requires_grad_(False)
    if (teacher.config.num_hidden_layers!=30 or teacher.config.hidden_size!=576
        or sum(p.numel() for p in teacher.parameters())!=EXPECTED_UNIQUE):
        raise AssertionError("Oscar v1C did not load genuine independent original pretrained FP32 model")
    original_fp32=s30.dense_from_paid(original_cb).eval().requires_grad_(False)
    oscar=encode_fp32(original_fp32.state_dict(),
                      mode=MODE_LOCKED,prior_blob=original_cb.blob)
    new_cb=CB59(oscar.blob)
    if new_cb.meta["original_unique_parameter_count"]!=BASE_36M:
        raise AssertionError("Oscar v1C genuine 43M 12layer source wrong")
    packed_source=to_cb61(oscar.blob)
    baseline=Stack(new_cb,oscar.blob,old_s142,dense=False)
    try:
        baseline_rows=score_teacher_vs_student(
            teacher,baseline,tok,rows,copy,label="V1C_original_S142_default_before")
    finally:
        baseline.close()
    if sum(x["protected_S142_old_full_EOS"] for x in baseline_rows)!=RETAINED_S142_EXPECTED_CORRECT_OLD8:
        raise AssertionError("Oscar v1C no authentic original S142 6/8 first 8 TRAIN baseline")
    # Precommitted 8 old TRAIN-debug samples: only first 4 receive gradient.
    trajectories=[capture_teacher_trajectory(teacher,tok,row) for row in rows]
    dump(out/"OSCAR_V1C_ORIGINAL_GENUINE_FP32_EIGHT_CAUSAL_TEACHER_TRAJECTORIES.json",[
        {k:v for k,v in example.items() if k!="teacher_training_distributions"}
        for example in trajectories
    ])
    if (not all(1<=len(t["teacher_full_token_ids_including_actual_EOS"])<=MAX_NEW
                for t in trajectories)
        or not all(len(t["teacher_training_distributions"])==
                    len(t["teacher_full_token_ids_including_actual_EOS"])
                   for t in trajectories)):
        raise AssertionError("Original pretrained teacher causal trajectory missing")
    original_fp16=load_file(str(old_v1b))
    student=Stack(new_cb,oscar.blob,old_s142,dense=False)
    route=OscarTeacherRoute().install(student.model)
    route.load_fp16(original_fp16)
    try:
        # Preserve no re-training of authentic original CALIBRI+GT3/9/Q2+S95/98.
        for mod in (student.model,student.inner,student.full,student.progressive,
                    student.bridge,student.first,student.later,student.ptr):
            mod.requires_grad_(False)
        route.up.requires_grad_(True)
        route.down.requires_grad_(True)
        first_before=score_teacher_vs_student(
            teacher,student,tok,rows,copy,
            label="V1C_before_original_V1B_FP16_teacher_route_OFF")
        if [x["CALIBRI43M_reloaded_generated_ids"] for x in first_before]!=[
            x["CALIBRI43M_reloaded_generated_ids"] for x in baseline_rows]:
            raise AssertionError("Oscar v1C trained separate v1B route modified default S142 before continuation")
        with route.teacher_mode():
            prior_teacher=score_teacher_vs_student(
                teacher,student,tok,rows,copy,
                label="V1C_before_causal_sequence_training_TEACHER_ROUTE")
        # Check teacher forcing only on first 4 TRAIN; monitoring four
        # old debug prompts cannot influence optimizer/weight selection.
        before_trajectory=[
            record_actual_student_on_teacher_trajectory(student,route,tok,row,
                example["teacher_full_token_ids_including_actual_EOS"],
                example["teacher_training_distributions"])
            for row,example in zip(rows[:TRAIN],trajectories[:TRAIN])
        ]
        save_routes_baseline=route.fp16()
        optimizer=torch.optim.Adam(
           [{"params":[route.up],"lr":LR_UP},
            {"params":[route.down],"lr":LR_DOWN}],weight_decay=0.)
        trace=[]
        for step in range(STEPS):
            ix=step%TRAIN
            original=trajectories[ix]
            optimizer.zero_grad(set_to_none=True)
            student_logits=student_causal_kv_teacher_forcing(
                student,route,tok,rows[ix],
                original["teacher_full_token_ids_including_actual_EOS"])
            loss,per_position=sequence_loss(
                student_logits,original["teacher_training_distributions"],
                original["teacher_full_token_ids_including_actual_EOS"],
                tok.eos_token_id)
            if not torch.isfinite(loss):
                raise AssertionError("Oscar v1C real FP32 pretrained full-sequence objective nonfinite")
            loss.backward()
            if (route.up.grad is None or route.down.grad is None or
                not torch.isfinite(route.up.grad).all() or
                not torch.isfinite(route.down.grad).all()):
                raise AssertionError("Oscar v1C 12layer teacher route not receiving real full-sequence gradients")
            up_g=float(route.up.grad.norm())
            down_g=float(route.down.grad.norm())
            if up_g<=0 or down_g<=0:
                raise AssertionError("Oscar v1C real student teacher-forced causal loss has zero gradient")
            torch.nn.utils.clip_grad_norm_(
                [route.up,route.down],.8,error_if_nonfinite=True)
            optimizer.step()
            event={
                "optimizer_step":step+1,
                "real_original_FP32_TRAIN_id":rows[ix]["TRAIN_id"],
                "original_teacher_generated_target_count":len(per_position),
                "real_multitoken_KV_teacher_forced_loss":float(loss.detach()),
                "real_12layer_new_route_up_gradient":up_g,
                "real_12layer_new_route_down_gradient":down_g,
                "per_original_pretrained_teacher_token_loss":per_position,
            }
            trace.append(event)
            if (step+1)%8==0:
                log("OSCAR_V1C_ACTUAL_INDEPENDENT_FP32_135M_CAUSAL_KV_DISTILL",
                    step=step+1,
                    original_pretrained_full_tokens=len(per_position),
                    loss=round(float(loss.detach()),5),
                    teacher_route_up_grad=round(up_g,5),
                    teacher_route_down_grad=round(down_g,5),
                    early_student_eos=bool(per_position[1]["student_teacherforced_argmax"]==
                        tok.eos_token_id if len(per_position)>1 else False))
        new_paid=route.fp16()
        fp16_changed={k:int(torch.count_nonzero(
            save_routes_baseline[k]!=new_paid[k])) for k in new_paid}
        if any(n==0 for n in fp16_changed.values()):
            raise AssertionError("Oscar v1C no physically changed rank8 FP16 from actual causal teacher")
        new_route_path=out/"OSCAR_V1C_NEW_TRUE_FP32_CAUSAL_12LAYER_TEACHER_ONLY_FP16.safetensors"
        save_file(new_paid,str(new_route_path))
        physically_read=load_file(str(new_route_path))
        if (set(physically_read)!=set(new_paid) or
            any(not torch.equal(physically_read[k],new_paid[k]) for k in new_paid)):
            raise AssertionError("Oscar v1C trained full-sequence FP16 binary reload differs")
    finally:
        route.uninstall()
        student.close()
    # Compare only after physically reloading saved paid weights into a
    # completely independent reconstructed original 43M inference runtime.
    reloaded=Stack(new_cb,oscar.blob,old_s142,dense=False)
    newroute=OscarTeacherRoute().install(reloaded.model)
    newroute.load_fp16(physically_read)
    try:
        original_default_after=score_teacher_vs_student(
            teacher,reloaded,tok,rows,copy,
            label="V1C_after_teacher_sequence_training_DEFAULT_OLD_S142")
        if [x["CALIBRI43M_reloaded_generated_ids"] for x in
            original_default_after]!=[x["CALIBRI43M_reloaded_generated_ids"]
                                     for x in baseline_rows]:
            raise AssertionError("Oscar v1C modified original CALIBRI 8 default full+EOS outputs")
        protected=sum(x["protected_S142_old_full_EOS"] for x in original_default_after)
        if protected!=RETAINED_S142_EXPECTED_CORRECT_OLD8:
            raise AssertionError("Oscar v1C catastrophic forgetting of original S142 6/8 full gold")
        with newroute.teacher_mode():
            after_teacher_route=score_teacher_vs_student(
                teacher,reloaded,tok,rows,copy,
                label="V1C_after_teacher_sequence_training_TEACHER_EXPERIMENTAL")
        after_trajectory=[
            record_actual_student_on_teacher_trajectory(reloaded,newroute,tok,row,
                example["teacher_full_token_ids_including_actual_EOS"],
                example["teacher_training_distributions"])
            for row,example in zip(rows[:TRAIN],trajectories[:TRAIN])
        ]
    finally:
        newroute.uninstall()
        reloaded.close()
    if (digest(args.S142_FP16.read_bytes())!=pre["actual_original_43M_SHA"]
        or digest(old_v1b.read_bytes())!=V1B_SHA):
        raise AssertionError("Oscar v1C original trained 43M S142 or inherited v1B sidecar modified")
    original_full8=[x["CALIBRI43M_reloaded_generated_ids"] for x in baseline_rows]
    summary={
       **pre,
       "real_true_original_FP32_135M_teacher_loaded":True,
       "trained_only_separate_Oscar_12layer_rank8":True,
       "original_FP32_43M_trained_independently_not_claimed":True,
       "pretrained_teacher_original_full_8_REAL_greedy_trajectories":[{
           k:v for k,v in sample.items() if k!="teacher_training_distributions"
       } for sample in trajectories],
       "optimizer_all_64_real_genuine_8token_teacher_distillation":trace,
       "original_V1B_12layer_rank8_FP16_changed_elements":fp16_changed,
       "previous_baseline_original_S142_all_8_full_answers":baseline_rows,
       "original_default_after_REAL_full_EOS":original_default_after,
       "pretrained_v1b_teacher_route_before":prior_teacher,
       "v1c_causally_trained_teacher_route_after":after_teacher_route,
       "teacher_forced_real_KV_train4_before":before_trajectory,
       "teacher_forced_real_KV_train4_after":after_trajectory,
       "old_S142_8_full_generations_exactly_retained":
           original_full8==[x["CALIBRI43M_reloaded_generated_ids"]
                            for x in original_default_after],
       "old_S142_protected_6_of8_gold_retained":protected,
       "real_pretrained_teacher_first_match_before":sum(
           x["student_first_matches_original_135M_FP32"] for x in prior_teacher),
       "real_pretrained_teacher_first_match_after":sum(
           x["student_first_matches_original_135M_FP32"] for x in after_teacher_route),
       "real_pretrained_teacher_full_EOS_match_before":sum(
           x["full_EOS_exact_original_135M_match"] for x in prior_teacher),
       "real_pretrained_teacher_full_EOS_match_after":sum(
           x["full_EOS_exact_original_135M_match"] for x in after_teacher_route),
       "actual_live_generation_EOS_before":sum(
           tok.eos_token_id in x["CALIBRI43M_reloaded_generated_ids"]
           for x in prior_teacher),
       "actual_live_generation_EOS_after":sum(
           tok.eos_token_id in x["CALIBRI43M_reloaded_generated_ids"]
           for x in after_teacher_route),
       "actual_teacher_EOS_8token_limit":sum(t["teacher_generated_valid_EOS"]
                                             for t in trajectories),
       "new_blind_QA_or_616_J7_rechecked":False,
       "teacher_generated_full_sequence_quality_verified":False,
       "model_accepted":False,
    }
    physical=physical_package(out,args,packed_source,new_route_path,summary)
    summary["complete_physical_source_bpp"]={
       "original_43M_nominal":physical["nominal_43M_bpp"],
       "conservative_excluding_6p88M":physical[
          "conservative_excluding_untrained_generated_growth_bpp"],
       "all_actual_files_bytes":physical["full_physical_inference_bytes"],
       "nominal_under_0p7":physical["nominal_bpp_le_0p7"],
       "conservative_under_0p7":physical["conservative_bpp_le_0p7"]
    }
    dump(out/"OSCAR_V1C_FINAL_TRUE_FP32_135M_FULL_SEQUENCE_TO_43M_GT12_RESULT.json",
         summary)
    log("OSCAR_V1C_FINAL_REAL_PRETRAINED_135M_CAUSAL_SEQUENCE_GT12",
       steps=STEPS,
       trained_teacher_full_tokens_max=MAX_NEW,
       real_fp32_teacher_first_before=summary["real_pretrained_teacher_first_match_before"],
       real_fp32_teacher_first_after=summary["real_pretrained_teacher_first_match_after"],
       real_fp32_teacher_full_EOS_before=summary["real_pretrained_teacher_full_EOS_match_before"],
       real_fp32_teacher_full_EOS_after=summary["real_pretrained_teacher_full_EOS_match_after"],
       old_S142_all8_exact=summary["old_S142_8_full_generations_exactly_retained"],
       old_S142_6gold_full=summary["old_S142_protected_6_of8_gold_retained"],
       bpp_nominal=physical["nominal_43M_bpp"],
       bpp_conservative=physical[
          "conservative_excluding_untrained_generated_growth_bpp"],
       pretrained_teacher_not_at_inference=True,
       independent_blind_QA_not_run=True,
       production_accepted=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for field in ("S93_CB61","S93_sidecar","S142_FP16","S94_bank",
                  "S94_report","S95_router","S95_report",
                  "S98_copy_net","S98_report","V1B_TRAINED_FP16","out"):
        p.add_argument(field,type=Path)
    run(p.parse_args())
