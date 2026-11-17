#!/usr/bin/env python3
"""Oscar v1B — genuine original 135M FP32 teacher, SEPARATE preserved 43M route.

V1 actually damaged 6/8 original correct S142 TRAIN outputs. Here original
S142 trained 43M, GT3/9 and all12 Q2 recurrence + S94/S95/S98 are immutable.
New 12×rank8 Oscar residual is off by default and genuinely trained only
under an explicit isolated teacher-only mode; both DOWN and UP physically
stored as FP16. No QA/general language-model equivalence claim.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s7a_43m_fp32_transfer.actual_43m_fp32_conversion import (
    Stack,validate_parent,selected_S93_existing_train,BASE_36M,TOTAL_43M,
    dump,digest,log,
)
from oscar.v1.independent_fp32_teacher_distill import (
    ids_for,real_student_first_logits,score_teacher_vs_student
)
from oscar.encoder import encode_fp32,to_cb61,MODE_LOCKED
from direct_codebook_v59 import CB59
from v176.operators import clear

RANK=8
WIDTH=320
LAYERS=12
STEPS=48
TRAIN=4
NUM=8
TEMP=2.0
SEED=431817
PREFIX="oscar.teacher_route."
RETAINED_S142_EXPECTED_CORRECT_OLD8=6


class OscarTeacherRoute(nn.Module):
    """12 independent FP16-paid teacher-only residuals, default ZERO-path.

    This leaves the original trained S142 GT3/9, full-12 recurrent Q2 GT and
    progressive rank896 physically untouched and always active with original
    S95/S98. The new adaptive 12layer residual is only active inside an
    explicit experimental branch; no implicit prompt/data-based switching.
    """
    def __init__(self):
        super().__init__()
        generator=torch.Generator(device="cpu").manual_seed(SEED)
        self.down=nn.Parameter(
            torch.randn((LAYERS,RANK,WIDTH),generator=generator)*.02)
        self.up=nn.Parameter(torch.zeros((LAYERS,WIDTH,RANK)))
        self._active=False
        self._handles=[]
        self._visited=[]
        self._last_visit=()

    def install(self,model):
        if self._handles or len(model.model.layers)!=LAYERS:
            raise AssertionError("Oscar v1B requires 12 original trained S142 layers once")
        def start(_model,_args):
            self._visited=[]
        def done(_model,_args,out):
            if self._active and self._visited!=list(range(LAYERS)):
                raise AssertionError("Oscar teacher branch did not reach all 12 decoder layers")
            if self._active:
                self._last_visit=tuple(self._visited)
            self._visited=[]
            return out
        self._handles.append(model.register_forward_pre_hook(start))
        for i,layer in enumerate(model.model.layers):
            def add_teacher(_layer,_args,output,*,idx=i):
                if not self._active:
                    return output
                h=output[0] if isinstance(output,tuple) else output
                if not isinstance(h,torch.Tensor) or h.shape[-1]!=WIDTH:
                    raise AssertionError("Oscar teacher 12layer hidden width changed")
                if len(self._visited)!=idx:
                    raise AssertionError("Oscar teacher hook full 12layer physical order drift")
                self._visited.append(idx)
                projected=F.gelu(F.linear(F.layer_norm(h,(WIDTH,)),self.down[idx]))
                updated=h+F.linear(projected,self.up[idx])
                return (updated,*output[1:]) if isinstance(output,tuple) else updated
            self._handles.append(layer.register_forward_hook(add_teacher))
        self._handles.append(model.register_forward_hook(done))
        return self

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles=[]
        self._active=False

    @contextmanager
    def teacher_mode(self):
        if self._active:
            raise AssertionError("Oscar teacher route must not nest/bleed into old S142")
        self._active=True
        try:
            yield
        finally:
            self._active=False
            self._visited=[]

    def fp16(self):
        return {
           PREFIX+"down_fp16":self.down.detach().cpu().half().contiguous(),
           PREFIX+"up_fp16":self.up.detach().cpu().half().contiguous(),
        }

    def load_fp16(self,payload):
        if set(payload)!={PREFIX+"down_fp16",PREFIX+"up_fp16"}:
            raise AssertionError("Oscar v1B missing real 12layer fully paid rank8 FP16")
        for key,part,shape in [
            (PREFIX+"down_fp16",self.down,(LAYERS,RANK,WIDTH)),
            (PREFIX+"up_fp16",self.up,(LAYERS,WIDTH,RANK))
        ]:
            val=payload[key]
            if (val.shape!=shape or val.dtype!=torch.float16
                or not torch.isfinite(val).all()):
                raise AssertionError("Oscar v1B physical adaptive teacher-only FP16 ABI invalid")
            with torch.no_grad():
                part.copy_(val.float())

    def nparams(self):
        return sum(v.numel() for v in self.parameters())


def physical_package(out,args,oscar_cb61,teacher_route_fp16,report):
    target=out/"OSCAR_V1B_ACTUAL_PHYSICAL_INFERENCE_PACKAGE"
    target.mkdir(parents=True,exist_ok=True)
    original_components={
      "OSCAR_V1B_REAL_43M_CORE_COMPRESSED.s61":oscar_cb61,
      "OSCAR_V1B_ORIGINAL_S142_UNCHANGED_43M_GT12.safetensors":args.S142_FP16,
      "OSCAR_V1B_SEPARATE_RANK8_TEACHER_BRANCH_12LAYERS.safetensors":teacher_route_fp16,
      "OSCAR_V1B_ORIGINAL_S94_BANK.bin":args.S94_bank,
      "OSCAR_V1B_ORIGINAL_S95_ROUTER.safetensors":args.S95_router,
      "OSCAR_V1B_ORIGINAL_S98_COPY.safetensors":args.S98_copy_net,
    }
    files={}
    for name,data in original_components.items():
        dest=target/name
        if isinstance(data,bytes):
            dest.write_bytes(data)
        else:
            shutil.copyfile(data,dest)
        files[name]={"physical_bytes":dest.stat().st_size,
                     "sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}
    manifest={
      "algorithm":"Oscar v1B",
      "original_fp32_teacher_is_independently_pretrained_135M":True,
      "source_student_Calibri_12layer_original_S142_unchanged":True,
      "student_new_original_teacher_only_12layer_rank8_GreenTunnel_paid_FP16":True,
      "new_teacher_branch_default_disabled_in_existing_live_routes":True,
      "new_teacher_route_requires_explicit_research_scope":True,
      "teacher_FP32_not_at_inference":True,
      "all_original_S142_source_37M_and_other_mechanisms_immutable":True,
      "no_new_EVAL_or_QA":True,
      "protected_456plus128plus32_not_rechecked":True,
      "no_native_CPU_90pct_speed_measurement":True,
      "trained_route_model_accepted":False,
      "actual_full_original_and_teacher_benchmark":report,
      "physical_exported_runtime_files":files,
      "physical_manifest_bytes":0,
      "full_physical_inference_bytes":0,
      "nominal_43M_bpp":0.,
      "conservative_excluding_untrained_generated_growth_bpp":0.,
      "nominal_bpp_le_0p7":False,
      "conservative_bpp_le_0p7":False,
    }
    path=target/"OSCAR_V1B_REAL_RUNTIME_MANIFEST.json"
    initial=(json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    charged=len(initial)+1024
    total=charged+sum(x["physical_bytes"] for x in files.values())
    manifest["physical_manifest_bytes"]=charged
    manifest["full_physical_inference_bytes"]=total
    manifest["nominal_43M_bpp"]=8*total/TOTAL_43M
    manifest["conservative_excluding_untrained_generated_growth_bpp"]=8*total/BASE_36M
    manifest["nominal_bpp_le_0p7"]=manifest["nominal_43M_bpp"]<=.7
    manifest["conservative_bpp_le_0p7"]=(manifest[
        "conservative_excluding_untrained_generated_growth_bpp"]<=.7)
    value=(json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    if len(value)>charged:
        raise AssertionError("Oscar v1B fixed physical manifest reservation inadequate")
    path.write_bytes(value+b" "*(charged-len(value)))
    if (path.stat().st_size!=charged or
        sum(x["physical_bytes"] for x in files.values())+charged!=total):
        raise AssertionError("Oscar v1B missing paid FP16 teacher branch / manifest bytes")
    if not manifest["nominal_bpp_le_0p7"]:
        raise AssertionError("Oscar v1B even structural model has >0.7bpp")
    return manifest


def run(a):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    original_cb,source,old_s93,old_s142,bank,router,copy=validate_parent(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    rows=selected_S93_existing_train(tok,router)
    if len(rows)!=NUM:
        raise AssertionError("Oscar v1B requires original 8 disclosed S142 S93 TRAIN cases")
    plan={
      "stage":"Oscar v1B new independent teacher route, old 43M source frozen",
      "teacher":"REAL ORIGINAL PRETRAINED 135M FP32 "+MODEL,
      "teacher_revision":MODEL_SHA,
      "student_architecture_original_S142":TOTAL_43M,
      "original_source_CB61_SHA":digest(a.S93_CB61.read_bytes()),
      "original_S142_all_GT12_sidecar_SHA":digest(a.S142_FP16.read_bytes()),
      "new_teacher12_layer_rank":RANK,
      "new_two_FP16_matrices_per_layer":True,
      "new_up_initialization":"EXACT ZERO",
      "original_GT3_9_and_Q2_12_and_S94_95_98_unmodified":True,
      "teacher_only_route_disabled_by_default":True,
      "TRAIN_first4":[x["TRAIN_id"] for x in rows[:TRAIN]],
      "TRAIN_debug_second4":[x["TRAIN_id"] for x in rows[TRAIN:]],
      "steps":STEPS,"new_Blind_eval":False,
      "protected_original_8_case_gold_recorded_before_training":True,
      "no_original_616_J7_unless_executed":True,
    }
    dump(out/"OSCAR_V1B_PRECOMMIT_BEFORE_MODEL_LOAD.json",plan)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
        trust_remote_code=False).cpu().eval().requires_grad_(False)
    if (teacher.config.num_hidden_layers!=30 or
        teacher.config.hidden_size!=576 or
        sum(p.numel() for p in teacher.parameters())!=EXPECTED_UNIQUE):
        raise AssertionError("Not true independent originally PRETRAINED FP32 135M")
    mirror=s30.dense_from_paid(original_cb).eval().requires_grad_(False)
    encoded=encode_fp32(mirror.state_dict(),mode=MODE_LOCKED,
                        prior_blob=original_cb.blob)
    new_source=CB59(encoded.blob)
    if new_source.meta["original_unique_parameter_count"]!=BASE_36M:
        raise AssertionError("Original 43M core not genuinely recoded using Oscar")
    packed=to_cb61(encoded.blob)
    baseline=Stack(new_source,encoded.blob,old_s142,dense=False)
    try:
        old=score_teacher_vs_student(teacher,baseline,tok,rows,copy,
                                      label="S142_43M_original_protected_before")
        if sum(x["protected_S142_old_full_EOS"] for x in old)!=RETAINED_S142_EXPECTED_CORRECT_OLD8:
            raise AssertionError("Original S142 43M pretrained TRAIN debug 6/8 baseline changed")
    finally:
        baseline.close()
    student=Stack(new_source,encoded.blob,old_s142,dense=False)
    # Additional donor route installs AFTER all originally trained internal
    # S142/GT3+9/Q2/progressive 12 hooks. OFF default does literal NO math.
    donor=OscarTeacherRoute().install(student.model)
    if donor.nparams()!=LAYERS*WIDTH*RANK*2:
        raise AssertionError("Unpaid teacher-specific 12layer residual shape")
    if torch.count_nonzero(donor.up).item()!=0:
        raise AssertionError("Oscar v1B teacher route zero baseline not exact")
    # Freeze every last original trained S142 model component.
    for module in [student.model,student.inner,student.full,
                   student.progressive,student.bridge,
                   student.first,student.later,student.ptr]:
        module.requires_grad_(False)
    donor.up.requires_grad_(True)
    donor.down.requires_grad_(True)
    old_original_S142_checksum=digest(a.S142_FP16.read_bytes())
    teacher_logits=[]
    with torch.inference_mode():
        for row in rows[:TRAIN]:
            logit=teacher(input_ids=ids_for(tok,row["question"]),
                          use_cache=False).logits[0,-1].detach().float().cpu()
            teacher_logits.append(logit)
    opt=torch.optim.Adam([{"params":[donor.up],"lr":0.002},
                          {"params":[donor.down],"lr":0.0005}])
    trace=[]
    initial_down=donor.down.detach().clone()
    try:
        # Initial zero teacher residual must preserve all actual original
        # S142 generated greedy full+EOS before any training.
        with donor.teacher_mode():
            zero=score_teacher_vs_student(teacher,student,tok,rows,copy,
                                         label="TEACHER_ROUTE_EXACT_ZERO_INITIAL")
        for orig,test in zip(old,zero):
            if orig["CALIBRI43M_reloaded_generated_ids"]!=test["CALIBRI43M_reloaded_generated_ids"]:
                raise AssertionError("Oscar zero route did not preserve the original complete S142 response")
        for step in range(STEPS):
            i=step%TRAIN
            clear(student.sources)
            opt.zero_grad(set_to_none=True)
            with donor.teacher_mode():
                logits=real_student_first_logits(student,tok,rows[i])
            tlogit=teacher_logits[i]
            soft_kl=F.kl_div(F.log_softmax(logits[None]/TEMP,-1),
                            F.softmax(tlogit[None]/TEMP,-1),
                            reduction="batchmean")*TEMP*TEMP
            top_id=int(tlogit.argmax())
            first=F.cross_entropy(logits[None],torch.tensor([top_id]))
            loss=soft_kl+0.15*first
            if not torch.isfinite(loss):
                raise AssertionError("Oscar v1B true pretrained teacher KL nonfinite")
            loss.backward()
            if donor.up.grad is None or not torch.isfinite(donor.up.grad).all():
                raise AssertionError("Oscar v1B separate FP32 teacher's 12-layer UP not receiving gradients")
            grad=float(donor.up.grad.norm())
            down_grad=(float(donor.down.grad.norm()) if donor.down.grad is not None else 0.)
            if grad<=0:
                raise AssertionError("Oscar v1B no real FP32 teacher gradient on new route")
            torch.nn.utils.clip_grad_norm_([donor.up,donor.down],1.0,error_if_nonfinite=True)
            opt.step()
            row_result={
                "step":step+1,"source_train_debug":rows[i]["TRAIN_id"],
                "real_FP32_pretrained_teacher_soft_KL":float(soft_kl.detach()),
                "real_FP32_pretrained_teacher_first_CE":float(first.detach()),
                "loss":float(loss.detach()),
                "original_43M_S142_weights_frozen":True,
                "new_Oscar_12layer_teacher_UP_gradient":grad,
                "new_Oscar_12layer_teacher_DOWN_gradient":down_grad,
            }
            trace.append(row_result)
            if (step+1)%8==0:
                log("OSCAR_V1B_REAL_PRETRAINED_FP32_TEACHER_TO_ISOLATED_12_GT_TRAIN",
                    step=step+1,loss=round(float(loss.detach()),4),
                    up_grad=round(grad,4),down_grad=round(down_grad,4))
        if not torch.any(donor.up.detach()!=0):
            raise AssertionError("Oscar v1B did not really train any 12layer new source")
        if not torch.any(donor.down.detach()!=initial_down):
            raise AssertionError("Oscar v1B did not train any 12layer DOWN source")
        live_fp16=donor.fp16()
        sidepath=out/"OSCAR_V1B_PRETRAINED_135M_TEACHER_NEW_ISOLATED_12LAYER_FP16.safetensors"
        save_file(live_fp16,str(sidepath))
        verified=load_file(str(sidepath))
        if (set(verified)!=set(live_fp16)
            or any(not torch.equal(verified[k],live_fp16[k]) for k in live_fp16)
            or not torch.count_nonzero(verified[PREFIX+"up_fp16"]).item()):
            raise AssertionError("Oscar v1B actual trainable FP16 teacher-route not physically serialized")
    finally:
        donor.uninstall()
        student.close()
    if digest(a.S142_FP16.read_bytes())!=old_original_S142_checksum:
        raise AssertionError("Oscar v1B mutated actual S142 source checkpoint!")
    reloaded=Stack(new_source,encoded.blob,old_s142,dense=False)
    route=OscarTeacherRoute().install(reloaded.model)
    route.load_fp16(verified)
    try:
        default=score_teacher_vs_student(teacher,reloaded,tok,rows,copy,
                      label="AFTER_OSCAR_TEACHER_TRAIN_DEFAULT_S142_UNMODIFIED")
        with route.teacher_mode():
            new=score_teacher_vs_student(teacher,reloaded,tok,rows,copy,
                           label="AFTER_OSCAR_TEACHER_TRAIN_ISOLATED_ROUTE")
        same=[x["CALIBRI43M_reloaded_generated_ids"]==
              y["CALIBRI43M_reloaded_generated_ids"]
              for x,y in zip(old,default)]
        if not all(same):
            raise AssertionError("Oscar v1B changed any original S142 default 8 actual generated responses")
        orig_correct=sum(int(r["protected_S142_old_full_EOS"]) for r in old)
        default_correct=sum(int(r["protected_S142_old_full_EOS"]) for r in default)
        if orig_correct!=default_correct or orig_correct!=RETAINED_S142_EXPECTED_CORRECT_OLD8:
            raise AssertionError("Oscar v1B default mode did not preserve 6 authentic original gold responses")
    finally:
        route.uninstall()
        reloaded.close()
    actual={
        **plan,"stage":"OSCAR_V1B_ORIGINAL_PRETRAINED_FP32_TEACHER_PRESERVED_43M_DEFAULT",
        "real_training_done":True,
        "original_authenticated_original_S142_FP16_SHA":old_original_S142_checksum,
        "new_teacher_FP16_sidecar_sha":digest(sidepath.read_bytes()),
        "paid_new_FP16_teacher_route_tensor_shapes":
             {k:list(v.shape) for k,v in verified.items()},
        "original_full_8_answers_identical_after_training":all(same),
        "original_S142_first8_debug_full_gold_before":orig_correct,
        "original_S142_first8_debug_full_gold_after":default_correct,
        "teacher_first_token_rank_old":[r["teacher_first_token_rank_in_student"] for r in old],
        "teacher_first_token_rank_new":[r["teacher_first_token_rank_in_student"] for r in new],
        "teacher_first_matches_before":sum(int(x[
             "student_first_matches_original_135M_FP32"]) for x in old),
        "teacher_first_matches_teacher_route_after":sum(int(x[
             "student_first_matches_original_135M_FP32"]) for x in new),
        "teacher_output_full_EOS_after":sum(int(x[
             "full_EOS_exact_original_135M_match"]) for x in new),
        "original_full8_CPU_default_exact_identity_guard":True,
        "actual_optimizer_steps":trace,
        "old_original_S142_default_before":old,
        "old_original_S142_default_after":default,
        "new_experimental_teacher_route_after":new,
        "all_456plus128_original_J7_after_not_retested":True,
        "new_open_independent_43M_FP32_pretrained_checkpoint_tested":False,
        "model_deployed_or_accepted":False,
    }
    physical=physical_package(out,a,packed,sidepath,actual)
    actual["paid_full_source_report"]={
      "original_43M_nominal_bpp":physical["nominal_43M_bpp"],
      "conservative_original_36M_bpp":physical[
         "conservative_excluding_untrained_generated_growth_bpp"],
      "complete_bytes":physical["full_physical_inference_bytes"],
      "teacher_not_shipped":True,
      "passes_nominal":physical["nominal_bpp_le_0p7"],
      "passes_conservative":physical["conservative_bpp_le_0p7"],
    }
    dump(out/"OSCAR_V1B_FINAL_TRUE_PRETRAINED_FP32_ISOLATED_TEACHER_BRANCH_RESULT.json",actual)
    log("OSCAR_V1B_FINAL_REAL_PRETRAINED_FP32_TEACHER_WITHOUT_S142_FORGETTING",
        actual_real_optimizer_steps=STEPS,
        new_route_off_default_replay_full_8_exact=sum(same),
        old_original_gold_full_preserved=default_correct,
        teacher_first_before=actual["teacher_first_matches_before"],
        teacher_first_after=actual["teacher_first_matches_teacher_route_after"],
        bpp_nominal=physical["nominal_43M_bpp"],
        bpp_conservative=physical["conservative_excluding_untrained_generated_growth_bpp"],
        original_J7_456_128_not_rerun=True,
        independent_heldout_QA_not_run=True,
        quality_accepted=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("S93_CB61","S93_sidecar","S142_FP16",
                 "S94_bank","S94_report","S95_router","S95_report",
                 "S98_copy_net","S98_report","out"):
        p.add_argument(name,type=Path)
    run(p.parse_args())
