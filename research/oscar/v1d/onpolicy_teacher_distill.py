#!/usr/bin/env python3
"""Oscar v1D: train on real on-policy student-generated prefixes, not gold ones.

* Teacher: genuine original independently pretrained FP32 SmolLM2 135M
  revision-pinned, 30 decoder layers x width576, 134,515,008 real parameters.
* Student: original trained S142 43.26M CALIBRI 12x320, physically paid
  Oscar-compiled 36.38M K32V8 source with GT3/9, GT12 Q2, original S94-98.
* Only Oscar v1B already-trained 12layer rank8 FP16 route changes. The
  route remains OFF by default; original S142 immutable, no teacher at runtime.
* Optimizer targets actual pretrained teacher logits conditioned on the
  student's generated prefixes. An actual premature EOS terminates rollout;
  never invent EOS or splice secret QA labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM,AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s7a_43m_fp32_transfer.actual_43m_fp32_conversion import (
    Stack,validate_parent,selected_S93_existing_train,BASE_36M,TOTAL_43M,
    dump,digest,log,
)
from oscar.encoder import MODE_LOCKED,encode_fp32,to_cb61
from oscar.v1.independent_fp32_teacher_distill import (
    ids_for,real_student_first_logits,score_teacher_vs_student,
)
from oscar.v1b.teacher_route_isolated import (
    OscarTeacherRoute,physical_package,PREFIX,
    RETAINED_S142_EXPECTED_CORRECT_OLD8,
)
from oscar.v1c.full_causal_teacher_sequence import (
    student_causal_kv_teacher_forcing,
)
from v176.operators import clear

V1B_SHA="11f79c0e9fb4f67e92149a8bdaf083f65a2c135168b557c6c8fb6df0d97aeaee"
V1B_BYTES=123080
TOTAL_STEPS=48
TRAIN_N=4
DEBUG_N=8
MAX_STUDENT_PREFIX=4
T=2.0
SEED=431118
LR_UP=.00055
LR_DOWN=.00018
FIRST_WEIGHT=3.0
FIRST_ANCHOR_WEIGHT=1.6
LATER_WEIGHT=1.3
EOS_PENALTY_WEIGHT=.4


@torch.inference_mode()
def actual_student_policy(student,route,tok,row,copy):
    """Actual S142 KV autoregressive S98-controlled student output.

    Don't replace generated tokens with teacher prefix. This is the
    exposure-bias correction and defines all subsequent teacher contexts.
    """
    with route.teacher_mode():
        emitted,override_meta=student.tokens(tok,row["question"],
                                              row["feature"],copy)
    if not emitted:
        raise AssertionError("Oscar v1D native original 43M generated no tokens")
    # Query the real independent teacher at steps BEFORE student EOS,
    # including the EOS prediction position itself (so EOS can be fixed).
    selected=[]
    for token in emitted[:MAX_STUDENT_PREFIX]:
        selected.append(int(token))
        if int(token)==tok.eos_token_id:
            break
    return selected,{
        "actual_43M_complete_greedy_tokens":list(map(int,emitted)),
        "actual_on_policy_prefix_tokens":selected,
        "S98_native_neural_copy_meta":override_meta,
    }


@torch.inference_mode()
def teacher_on_actual_student_prefixes(teacher,tok,row,actual_student_tokens):
    """Get real 135M FP32 source distribution for each student's OWN prefix.

    These teacher logits are NOT identical to the old v1C teacher-forced
    references when the current 43M branch chooses an erroneous token.
    """
    prompt=ids_for(tok,row["question"])
    full=prompt
    targets=[]
    for i,student_emitted in enumerate(actual_student_tokens):
        logits=teacher(input_ids=full,use_cache=False).logits[0,-1].float().cpu()
        if logits.shape!=(tok.vocab_size,) or not torch.isfinite(logits).all():
            raise AssertionError("Oscar original 135M FP32 real-on-policy logits invalid")
        teacher_target=int(logits.argmax())
        targets.append({
            "step":i,
            "source_actual_student_token_id":int(student_emitted),
            "original_teacher_target_id_on_student_prefix":teacher_target,
            "teacher_EXPECTED_EOS":teacher_target==tok.eos_token_id,
            "actual_student_emitted_EOS":int(student_emitted)==tok.eos_token_id,
            "teacher_full_vocabulary_logits_SHA256":hashlib.sha256(
                logits.numpy().tobytes()).hexdigest(),
            "whole_vocab_target":logits,
        })
        if int(student_emitted)!=tok.eos_token_id:
            full=torch.cat((full,full.new_tensor([[int(student_emitted)]])),dim=-1)
        elif i!=len(actual_student_tokens)-1:
            raise AssertionError("Oscar v1D used post-EOS on-policy tokens")
    return targets


def loss_against_real_student_policy(
        student_logits,teacher_targets,baseline_first_logits,eos_id):
    if len(student_logits)!=len(teacher_targets):
        raise AssertionError("Oscar v1D teacher/student genuine prefix coverage changed")
    if len(student_logits)>MAX_STUDENT_PREFIX or not student_logits:
        raise AssertionError("Oscar v1D unexpected on-policy original student sample count")
    weights=[]
    losses=[]
    seen=Counter()
    snapshot=[]
    for i,(current,target) in enumerate(zip(student_logits,teacher_targets)):
        reference=target["whole_vocab_target"]
        target_id=int(target["original_teacher_target_id_on_student_prefix"])
        if not torch.isfinite(current).all():
            raise AssertionError("Oscar real pretrained teacher route logits nonfinite")
        dist=F.kl_div(F.log_softmax(current[None]/T,dim=-1),
                      F.softmax(reference[None]/T,dim=-1),
                      reduction="batchmean")*T*T
        ce=F.cross_entropy(current[None],
                            current.new_tensor([target_id],dtype=torch.long))
        if target_id!=eos_id:
            eos_penalty=F.softplus(
                current[eos_id]-current[target_id]+1.0)
        else:
            eos_penalty=current.new_zeros(())
        # Prevent repeated "0" tokens from dominating the entire sequence.
        frequency_weight=1.0/(1+seen[target_id])
        seen[target_id]+=1
        positional=FIRST_WEIGHT if i==0 else LATER_WEIGHT*frequency_weight
        scalar=dist+.17*ce+EOS_PENALTY_WEIGHT*eos_penalty
        if i==0:
            frozen=F.kl_div(F.log_softmax(current[None]/T,-1),
                            F.softmax(baseline_first_logits[None]/T,-1),
                            reduction="batchmean")*T*T
            scalar=scalar+FIRST_ANCHOR_WEIGHT*frozen
        else:
            frozen=current.new_zeros(())
        losses.append(scalar*positional)
        weights.append(positional)
        snapshot.append({
          "generated_student_position":i,
          "student_actual_on_policy_token":target["source_actual_student_token_id"],
          "original_pretrained_FP32_teacher_target":target_id,
          "teacher_target_rank_under_student":
              int((current.detach()>current.detach()[target_id]).sum())+1,
          "original_FP32_full_vocab_soft_KL":float(dist.detach()),
          "real_teacher_argmax_CE":float(ce.detach()),
          "EOS_correction":float(eos_penalty.detach()),
          "original_v1B_first_token_anchor_KL":float(frozen.detach()),
          "frequency_weight":frequency_weight,
          "position_objective_weight":positional,
        })
    return torch.stack(losses).sum()/sum(weights),snapshot


def run(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar v1D output must be initially EMPTY immutable research directory")
    v1b_path=Path(args.OSCAR_V1B_FP16)
    if (v1b_path.stat().st_size!=V1B_BYTES or
        digest(v1b_path.read_bytes())!=V1B_SHA):
        raise AssertionError("Oscar v1D refuses invalid original trained v1B preserved FP16 source")
    original,oldraw,old_s93,s142,bank,router,copy=validate_parent(args)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    rows=selected_S93_existing_train(tok,router)
    if len(rows)!=DEBUG_N:
        raise AssertionError("Oscar original 8 old disclosed TRAIN debug prompts changed")
    precommit={
      "experiment":"Oscar v1D REAL original FP32 135M teacher conditioned on 43M STUDENT GENERATED prefixes",
      "original_teacher":"HuggingFaceTB/SmolLM2-135M-Instruct",
      "original_teacher_revision":MODEL_SHA,
      "original_teacher_unique_params":EXPECTED_UNIQUE,
      "real_original_S142_FP16_SHA256":digest(args.S142_FP16.read_bytes()),
      "original_trained_v1b_FP16_SHA256":V1B_SHA,
      "do_NOT_restart_from_v1c_repeated_zero_model":True,
      "training_original_S142_TRAIN_ids":[r["TRAIN_id"] for r in rows[:TRAIN_N]],
      "monitor_old_TRAIN_DEBUG_only":[r["TRAIN_id"] for r in rows[TRAIN_N:]],
      "max_real_student_prefix":MAX_STUDENT_PREFIX,
      "true_student_policy_token_source":"Original actual S142 S98-neural-copy KV greedy with separate Oscar v1B teacher mode",
      "source_teacher_next_vocab_queries":"true original FP32 on student-produced tokens, NOT teacher-forced gold prefixes",
      "total_actual_original_teacher_distillation_steps":TOTAL_STEPS,
      "original_GreenTunnel_12layers_in_student":True,
      "original_43M_S142_and_GreenTunnel_frozen":True,
      "teacher_mode_disabled_by_default":True,
      "full_trained_616_J7_not_tested":True,
      "independent_new_QA_EVAL_never_used":True,
      "no_LIVE_main_Render_Supabase_APK_JessicaChat_NRA":True,
    }
    dump(out/"OSCAR_V1D_PRECOMMIT_BEFORE_TEACHER_AND_STUDENT_LOAD.json",precommit)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
        trust_remote_code=False).cpu().eval().requires_grad_(False)
    if (teacher.config.num_hidden_layers!=30 or teacher.config.hidden_size!=576 or
        sum(p.numel() for p in teacher.parameters())!=EXPECTED_UNIQUE):
        raise AssertionError("Oscar v1D not original independently pretrained real FP32 teacher")
    dense=s30.dense_from_paid(original).eval().requires_grad_(False)
    encoded=encode_fp32(dense.state_dict(),mode=MODE_LOCKED,
                        prior_blob=original.blob)
    paid=CB59(encoded.blob)
    if paid.meta["original_unique_parameter_count"]!=BASE_36M:
        raise AssertionError("Oscar actual 43M source was not recompiled from FP32")
    paid_cb61=to_cb61(encoded.blob)
    baseline=Stack(paid,encoded.blob,s142,dense=False)
    try:
        protected_before=score_teacher_vs_student(
            teacher,baseline,tok,rows,copy,
            label="V1D_original_training_S142_default_before")
    finally:
        baseline.close()
    if sum(int(x["protected_S142_old_full_EOS"]) for x in protected_before)!=RETAINED_S142_EXPECTED_CORRECT_OLD8:
        raise AssertionError("Oscar v1D 43M S142 genuine 6/8 original TRAIN gold baseline lost")
    # Restore actual v1B FP16 learned by an original genuine pretrained 135M teacher.
    old_v1b=load_file(str(v1b_path))
    student=Stack(paid,encoded.blob,s142,dense=False)
    route=OscarTeacherRoute().install(student.model)
    route.load_fp16(old_v1b)
    for m in (student.model,student.inner,student.full,student.progressive,
              student.bridge,student.first,student.later,student.ptr):
        m.requires_grad_(False)
    route.up.requires_grad_(True)
    route.down.requires_grad_(True)
    previous_fp16=route.fp16()
    try:
        with route.teacher_mode():
            teacher_route_before=score_teacher_vs_student(
                teacher,student,tok,rows,copy,
                label="V1D_original_v1B_135M_teacher_route_before_ON_POLICY")
        before_first=sum(int(x["student_first_matches_original_135M_FP32"])
                         for x in teacher_route_before)
        if before_first!=6:
            raise AssertionError("Oscar v1D source v1B must be the authentic 6/8 pretrained FP32 first-token control")
        anchors=[]
        with torch.inference_mode():
            for row in rows[:TRAIN_N]:
                with route.teacher_mode():
                    logits=real_student_first_logits(student,tok,row).detach().float().cpu()
                anchors.append(logits.clone())
        optimizer=torch.optim.Adam([
            {"params":[route.up],"lr":LR_UP},
            {"params":[route.down],"lr":LR_DOWN},
        ],weight_decay=0.)
        trace=[]
        all_real_teacher_prefix_queries=0
        late_corrections=0
        for step in range(TOTAL_STEPS):
            which=step%TRAIN_N
            row=rows[which]
            # Native original S98 branch generation (true MODEL policy).
            onpolicy,meta=actual_student_policy(student,route,tok,row,copy)
            fp32_teacher=teacher_on_actual_student_prefixes(
                teacher,tok,row,onpolicy)
            if len(fp32_teacher)!=len(onpolicy) or len(onpolicy)>MAX_STUDENT_PREFIX:
                raise AssertionError("Oscar authentic on-policy prefix teacher count corrupted")
            all_real_teacher_prefix_queries+=len(fp32_teacher)
            late_corrections+=sum(int(x["actual_student_emitted_EOS"] and
                                     not x["teacher_EXPECTED_EOS"]) for x in fp32_teacher)
            clear(student.sources)
            optimizer.zero_grad(set_to_none=True)
            # Differentiable actual original S142 real KV, first/later,
            # GT3/9, Q2-GT12, original S142 48 active progressive.
            logits=student_causal_kv_teacher_forcing(
                student,route,tok,row,onpolicy)
            loss,per_token=loss_against_real_student_policy(
                logits,fp32_teacher,anchors[which],tok.eos_token_id)
            if not torch.isfinite(loss):
                raise AssertionError("Oscar v1D original FP32 teacher full on-policy loss not finite")
            loss.backward()
            if route.up.grad is None or route.down.grad is None:
                raise AssertionError("Oscar v1D actual original 135M teacher did not train separate rank8 route")
            upnorm=float(route.up.grad.norm())
            downnorm=float(route.down.grad.norm())
            if not upnorm>0 or not downnorm>0:
                raise AssertionError("Oscar v1D actual original 135M teacher rank8 path gradient is zero")
            torch.nn.utils.clip_grad_norm_(
                (route.up,route.down),.8,error_if_nonfinite=True)
            optimizer.step()
            trace.append({
               "real_optimizer_step":step+1,
               "actual_old_TRAIN_id":row["TRAIN_id"],
               "genuinely_student_generated_prefix":onpolicy,
               "student_actual_native_S98_copy_meta":meta["S98_native_neural_copy_meta"],
               "independent_FP32_teacher_own_student_prefix_targets":[
                   {k:v for k,v in entry.items() if k!="whole_vocab_target"}
                   for entry in fp32_teacher],
               "fully_original_F32_on_policy_training_loss":float(loss.detach()),
               "new_route_all12_UP_grad_norm":upnorm,
               "new_route_all12_DOWN_grad_norm":downnorm,
               "causal_teacher_distill_at_each_student_position":per_token,
            })
            if (step+1)%8==0:
                log("OSCAR_V1D_TRUE_135M_FP32_ON_STUDENT_GENERATED_PREFIXES",
                    step=step+1,
                    genuine_prefixes=len(onpolicy),
                    total_FP32_student_prefix_queries=all_real_teacher_prefix_queries,
                    premature_student_EOS_events=late_corrections,
                    loss=round(float(loss.detach()),5),
                    UP_grad=round(upnorm,4),DOWN_grad=round(downnorm,4),
                    full_original_S142_frozen=True)
        new_fp16=route.fp16()
        changed={k:int(torch.count_nonzero(new_fp16[k]!=previous_fp16[k]))
                 for k in new_fp16}
        if len(changed)!=2 or not all(value>0 for value in changed.values()):
            raise AssertionError("Oscar v1D failed to physically change BOTH FP16 rank8 12layer tensors")
        fp16_path=out/"OSCAR_V1D_REAL_135M_TEACHER_ONPOLICY_12LAYER_FP16.safetensors"
        save_file(new_fp16,str(fp16_path))
        verified=load_file(str(fp16_path))
        if (set(verified)!=set(new_fp16) or
            any(not torch.equal(verified[k],new_fp16[k]) for k in verified)):
            raise AssertionError("Oscar v1D actual separate FP16 route physical save/reload failed")
        dump(out/"OSCAR_V1D_INCREMENTAL_ONPOLICY_REAL_TRAINING_48STEPS.json",trace)
    finally:
        route.uninstall()
        student.close()
    restored=Stack(paid,encoded.blob,s142,dense=False)
    loaded_route=OscarTeacherRoute().install(restored.model)
    loaded_route.load_fp16(verified)
    try:
        default_after=score_teacher_vs_student(
            teacher,restored,tok,rows,copy,
            label="V1D_AFTER_trueFP32_ON_POLICY_default_old_S142_OFF")
        with loaded_route.teacher_mode():
            generated_after=score_teacher_vs_student(
                teacher,restored,tok,rows,copy,
                label="V1D_AFTER_trueFP32_ON_POLICY_OSCAR_TEACHER_ON")
        if [x["CALIBRI43M_reloaded_generated_ids"] for x in protected_before]!=[
            x["CALIBRI43M_reloaded_generated_ids"] for x in default_after]:
            raise AssertionError("Oscar v1D catastrophically altered the ORIGINAL 43M S142 complete-token default outputs")
        protected=sum(int(x["protected_S142_old_full_EOS"]) for x in default_after)
        if protected!=RETAINED_S142_EXPECTED_CORRECT_OLD8:
            raise AssertionError("Oscar v1D lost originally correct 6/8 protected CALIBRI old TRAIN answers")
    finally:
        loaded_route.uninstall()
        restored.close()
    if (digest(args.S142_FP16.read_bytes())!=precommit["real_original_S142_FP16_SHA256"] or
        digest(v1b_path.read_bytes())!=V1B_SHA):
        raise AssertionError("Oscar v1D immutable original CALIBRI or inherited v1B source mutated")
    teacher_first_before=sum(int(x["student_first_matches_original_135M_FP32"])
                             for x in teacher_route_before)
    teacher_first_after=sum(int(x["student_first_matches_original_135M_FP32"])
                            for x in generated_after)
    teacher_full_before=sum(int(x["full_EOS_exact_original_135M_match"])
                           for x in teacher_route_before)
    teacher_full_after=sum(int(x["full_EOS_exact_original_135M_match"])
                          for x in generated_after)
    summary={
      **precommit,
      "complete":True,
      "real_grad_steps":TOTAL_STEPS,
      "true_original_135M_FP32_teacher_queried_on_student_emitted_prefixes":
          all_real_teacher_prefix_queries,
      "real_student_early_EOS_correction_train_events":late_corrections,
      "genuinely_original_pretrained_FP32_not_a_Calibri_43M_derived_mirror":True,
      "original_S142_default_all8_generated_complete_tokens_reproduced":True,
      "original_S142_default_6_of8_gold_correct_retained":protected,
      "unseen_456plus128plus32_original_Guard_not_replayed":True,
      "teacher_route_original_v1B_FP16_source_SHA256":V1B_SHA,
      "new_Oscar_v1D_saved_FP16_SHA256":digest(fp16_path.read_bytes()),
      "actual_original_rank8_FP16_changed_cells":changed,
      "old_student_DEFAULT_before":protected_before,
      "old_student_DEFAULT_after_physical_reloaded":default_after,
      "actual_OSCAR_v1B_first_token_teacher_route_before":teacher_route_before,
      "actual_OSCAR_v1D_onpolicy_teacher_route_after":generated_after,
      "teacher_first_token_matches_before":teacher_first_before,
      "teacher_first_token_matches_after":teacher_first_after,
      "real_autoregressive_full_output_EOS_matches_before":teacher_full_before,
      "real_autoregressive_full_output_EOS_matches_after":teacher_full_after,
      "incremental_real_onpolicy_48_step_train_trace_in_separate_file":True,
      "v1D_independent_heldout_QA_tested":False,
      "real_full_orig_616_J7_not_tested":True,
      "v1D_teacher_token_language_skills_restored_NOT_CLAIMED":True,
      "model_accepted":False,
    }
    # Do not ship TRAIN/debug traces or teacher probabilities inside runtime.
    runtime_receipt={
       "algorithm":"Oscar v1D",
       "teacher_original_135M_revision":MODEL_SHA,
       "teacher_is_independently_pretrained":True,
       "original_S142_trained_43M_immutable_sha256":
           precommit["real_original_S142_FP16_SHA256"],
       "original_trained_v1B_12layer_rank8_SHA256":V1B_SHA,
       "updated_separate_12layer_rank8_FP16_SHA256":digest(fp16_path.read_bytes()),
       "real_original_135M_FP32_queries_on_student_emitted_prefixes":
           all_real_teacher_prefix_queries,
       "actual_old_8_original_S142_default_complete_outputs_kept":True,
       "full_true_pretrained_teacher_8debug_matches":teacher_full_after,
       "full_original_616_J7_and_new_blind_QA_not_tested":True,
       "teacher_FP32_NOT_shipped_in_student_inference":True,
       "independent_training_traces_not_shipped_in_runtime":True,
       "quality_not_proven_model_not_promoted":True,
    }
    paid_output=physical_package(out,args,paid_cb61,fp16_path,runtime_receipt)
    summary["actual_physically_paid_runtime"]={
      "complete_bytes":paid_output["full_physical_inference_bytes"],
      "nominal_43p26M_bpp":paid_output["nominal_43M_bpp"],
      "conservative_excluding_6p88M_bpp":
          paid_output["conservative_excluding_untrained_generated_growth_bpp"],
      "nominal_le_0p7":paid_output["nominal_bpp_le_0p7"],
      "conservative_le_0p7":paid_output["conservative_bpp_le_0p7"],
    }
    dump(out/"OSCAR_V1D_FINAL_INDEPENDENT_FP32_135M_TEACHER_ONPOLICY_REAL_43M_RESULT.json",
         summary)
    log("OSCAR_V1D_FINAL_REAL_135M_FP32_STUDENT_PREFIX_DISTILL",
        real_steps=TOTAL_STEPS,
        actual_original_FP32_teacher_prefix_queries=all_real_teacher_prefix_queries,
        teacher_first_before=teacher_first_before,
        teacher_first_after=teacher_first_after,
        teacher_full_before=teacher_full_before,
        teacher_full_after=teacher_full_after,
        original_S142_8full_exact=True,
        original_S142_6gold_kept=protected,
        nominal_full_paid_bpp=paid_output["nominal_43M_bpp"],
        conservative_full_paid_bpp=paid_output[
            "conservative_excluding_untrained_generated_growth_bpp"],
        full_616_J7_not_retested=True,
        independent_unseen_QA_not_run=True,
        model_accepted=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("S93_CB61","S93_sidecar","S142_FP16","S94_bank",
                 "S94_report","S95_router","S95_report",
                 "S98_copy_net","S98_report","OSCAR_V1B_FP16","out"):
        parser.add_argument(name,type=Path)
    run(parser.parse_args())
