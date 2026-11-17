#!/usr/bin/env python3
"""Oscar v1: independent genuine pretrained original FP32 135M -> existing 43M GT12.

Unlike v0 SAME-SOURCE FP32 inversion, the training teacher here really is
the pretrained original SmolLM2 135M full 30-layer 576-width FP32 checkpoint.
Only the physical paid original 43M S142 adapter and Q2 full-12 GT are
updated. Original S87 trained GT3/9, core, S94 bank, S95 route, S98 copy
remain immutable. Research-only TRAIN demonstration, no blind QA/J7 claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM,AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s126.compact_rank4_q2 import export_compact_q2, PREFIX
from calibri2.s132.progressive_core import IDS_KEY,WEIGHTS_KEY
from calibri2.s7a_43m_fp32_transfer.actual_43m_fp32_conversion import (
    Stack,validate_parent,selected_S93_existing_train,BASE_36M,TOTAL_43M,digest,dump,log
)
from oscar.encoder import encode_fp32,to_cb61,MODE_LOCKED
from v176.operators import clear

TRAIN_N=4
CHECK_N=8
STEPS=16
T=2.
LR_UP=0.0005
LR_Q2=0.0003
MAX_NEW=8
MAX_NOMINAL_BPP=.7


def ids_for(tok,question):
    txt=tok.apply_chat_template(
        [{"role":"user","content":question}],tokenize=False,
        add_generation_prompt=True)
    return torch.tensor([tok.encode(txt,add_special_tokens=False)],dtype=torch.long)


@torch.inference_mode()
def original_fp32_full_greedy(teacher,tok,question):
    ids=ids_for(tok,question)
    output=[]
    eos=False
    for _ in range(MAX_NEW):
        logits=teacher(input_ids=ids,use_cache=False).logits[0,-1]
        token=int(logits.argmax())
        if token==tok.eos_token_id:
            eos=True
            break
        output.append(token)
        ids=torch.cat((ids,ids.new_tensor([[token]])),1)
    return {"ids":output,
            "full_token_ids_with_EOS":output+([tok.eos_token_id] if eos else []),
            "text":tok.decode(output,skip_special_tokens=True),
            "EOS":eos}


def real_student_first_logits(stu,tok,row):
    ctx=ids_for(tok,row["question"])
    clear(stu.sources)
    with stu.progressive.prefill_only(stu.model),\
         stu.bridge.prompt_scope("S93",row["feature"]),\
         __import__("calibri2.s127.q2_prefill_gate_control",
             fromlist=["prefill_only_tunnel"]).prefill_only_tunnel(stu.model,stu.full):
        output=stu.model(input_ids=ctx,past_key_values=None,
                         use_cache=True,output_hidden_states=True)
        hidden=output.hidden_states[-1][0,-1]
        logits=output.logits[0,-1]+F.linear(stu.first(hidden),stu.head)
    if (stu.full.last_forward_coverage!=tuple(range(12)) or
        stu.progressive.last_prefill_coverage!=tuple(range(12))):
        raise AssertionError("Oscar v1 genuine all-12 S142 GT/adapter transformer hooks missing")
    return logits


@torch.inference_mode()
def score_teacher_vs_student(teacher,stu,tok,rows,copy,*,label):
    result=[]
    for idx,row in enumerate(rows):
        tlogits=teacher(input_ids=ids_for(tok,row["question"]),
                        use_cache=False).logits[0,-1].float()
        slogits=real_student_first_logits(stu,tok,row).float()
        source=original_fp32_full_greedy(teacher,tok,row["question"])
        generated,_=stu.tokens(tok,row["question"],row["feature"],copy)
        dist=F.kl_div(F.log_softmax(slogits[None]/T,-1),
                      F.softmax(tlogits[None]/T,-1),
                      reduction="batchmean")*T*T
        teacher_first=int(tlogits.argmax())
        teacher_first_rank=int((slogits>slogits[teacher_first]).sum().item())+1
        row_data={
          "TRAIN_debug_index":idx,
          "original_prior_training_id":row["TRAIN_id"],
          "TRAIN_question_sha256":digest(row["question"].encode()),
          "FP32_original_model_is_genuinely_independently_pretrained":True,
          "real_FP32_135M_30layer_answer":source,
          "CALIBRI43M_reloaded_generated_ids":generated,
          "CALIBRI43M_reloaded_text":tok.decode(generated,skip_special_tokens=True),
          "full_EOS_exact_original_135M_match":
              generated==source["full_token_ids_with_EOS"],
          "S142_original_43M_expected_gold":row["expected"],
          "protected_S142_old_full_EOS":generated==row["expected"],
          "teacher_FP32_first_token_id":teacher_first,
          "student_first_token_id":int(slogits.argmax()),
          "student_first_matches_original_135M_FP32":
              int(slogits.argmax())==teacher_first,
          "teacher_first_token_rank_in_student":teacher_first_rank,
          "whole_vocab_soft_KL_at_T2":float(dist.item()),
          "teacher_independent_model_not_same_source":True,
          "stage":label
        }
        result.append(row_data)
        log("OSCAR_V1_trueFP32_135M_vs_real43M_GT12",
            phase=label,prompt=idx+1,
            teacher=source["text"][:48],
            student=row_data["CALIBRI43M_reloaded_text"][:48],
            teacher_first_rank=teacher_first_rank,
            kl=round(float(dist.item()),5),
            old_answer_retained=row_data["protected_S142_old_full_EOS"])
    return result


def paid_output(out,args,raw_source,sidecar,answer_report):
    target=out/"actual_physical_Oscar_v1_model"
    target.mkdir(parents=True,exist_ok=True)
    sources={
      "OSCAR_REAL_43M_12layer_core_source.s61":raw_source,
      "OSCAR_S142_TRAINED_SIDE_GT12_FP16.safetensors":sidecar,
      "OSCAR_S94_ALL_THREE_PROFILES.bin":args.S94_bank,
      "OSCAR_S95_REAL_ROUTER.safetensors":args.S95_router,
      "OSCAR_S98_REAL_COPY.safetensors":args.S98_copy_net
    }
    component={}
    for name,source in sources.items():
        p=target/name
        if isinstance(source,bytes):
            p.write_bytes(source)
        else:
            shutil.copyfile(source,p)
        component[name]={"bytes":p.stat().st_size,
                         "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    metadata={
      "stage":"Oscar v1 real ORIGINAL pretrained independent FP32 135M → CALIBRI 43M full12 GT",
      "external_teacher":"HuggingFaceTB/SmolLM2-135M-Instruct",
      "teacher_SHA":MODEL_SHA,
      "teacher_unique_params":EXPECTED_UNIQUE,
      "student_S142_12layer_43p26176M_structure":TOTAL_43M,
      "student_base_36p38048M_actual_trained_core":BASE_36M,
      "student_zero_seed_generated_params":TOTAL_43M-BASE_36M,
      "real_FP32_teacher_NOT_shipped":True,
      "FP32_teacher_to_12layer_adaptor_REAL_gradient_steps":STEPS,
      "student_original_trained_GT3_9_unchanged":True,
      "new_student_Q2_rank4_GT12_quantized_in_real_sidecar":True,
      "student_S142_48active_channels_each_of_12_updated":True,
      "all_original_sidecars_physically_carried":True,
      "no_NRA_no_LIVE_no_existing_Jessica_chat":True,
      "no_unseen_independent_QA_scored":True,
      "full_456plus128_J7_not_replayed":True,
      "native_C_CPU_speed_test_not_performed":True,
      "no_model_quality_or_deployment_acceptance":True,
      "full_actual_student_and_teacher_comparison":answer_report,
      "literal_inference_files":component,
      "literal_manifest_bytes":0,
      "complete_real_inference_bytes":0,
      "nominal_bpp":0.,
      "conservative_excluding_zero_seed_growth_bpp":0.,
      "nominal_under_0p7":False,
      "conservative_under_0p7":False,
      "candidate_accepted":False,
    }
    manifest=target/"OSCAR_V1_ACTUAL_RUNTIME_MANIFEST.json"
    initial=(json.dumps(metadata,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    reserved=len(initial)+1024
    allbytes=reserved+sum(x["bytes"] for x in component.values())
    metadata["literal_manifest_bytes"]=reserved
    metadata["complete_real_inference_bytes"]=allbytes
    metadata["nominal_bpp"]=8*allbytes/TOTAL_43M
    metadata["conservative_excluding_zero_seed_growth_bpp"]=8*allbytes/BASE_36M
    metadata["nominal_under_0p7"]=metadata["nominal_bpp"]<=MAX_NOMINAL_BPP
    metadata["conservative_under_0p7"]=(metadata["conservative_excluding_zero_seed_growth_bpp"]
                                         <=MAX_NOMINAL_BPP)
    data=(json.dumps(metadata,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    if len(data)>reserved:
        raise AssertionError("Oscar v1 full physical manifest reservation insufficient")
    manifest.write_bytes(data+b" "*(reserved-len(data)))
    if manifest.stat().st_size!=reserved or allbytes!=(
        sum(x["bytes"] for x in component.values())+reserved):
        raise AssertionError("Oscar v1 complete physical decoder sidecars not counted")
    if not metadata["nominal_under_0p7"]:
        raise AssertionError("Oscar v1 nominal full 43M physical source exceeds 0.7 bpp")
    return metadata


def run(args):
    torch.manual_seed(431171)
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    parent,raw,s93,s142,bank,router,copy=validate_parent(args)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    rows=selected_S93_existing_train(tok,router)
    if len(rows)!=CHECK_N:
        raise AssertionError("Oscar v1 eight known original S142 TRAIN rows missing")
    precommit={
      "phase":"OSCAR_V1_PRECOMMIT_BEFORE_REAL_FP32_teacher_loading",
      "teacher_name":MODEL,"teacher_pinned_SHA":MODEL_SHA,
      "teacher_expected_original_trained_unique_parameters":EXPECTED_UNIQUE,
      "old_Calibri_S142_research_FPS_SOURCE":args.S142_FP16.name,
      "actual_student_12layer":True,
      "TRAIN_only_teacher_pairs_first4":[x["TRAIN_id"] for x in rows[:TRAIN_N]],
      "TRAIN_debug_only_other4":[x["TRAIN_id"] for x in rows[TRAIN_N:]],
      "train_question_sha256":[digest(x["question"].encode()) for x in rows[:TRAIN_N]],
      "optimizer_steps":STEPS,
      "old_S142_full456_and_new128_not_run":True,
      "original_pretrained_teacher_not_43M_FP32_checkpoint":True,
      "independent_test_never_opened":True,
      "no_main_live_render_supabase":True,
    }
    dump(out/"OSCAR_V1_PRECOMMIT.json",precommit)
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
        trust_remote_code=False).cpu().eval()
    teacher.requires_grad_(False)
    if (teacher.config.num_hidden_layers!=30 or teacher.config.hidden_size!=576 or
        sum(p.numel() for p in teacher.parameters())!=EXPECTED_UNIQUE or
        teacher.config.vocab_size!=tok.vocab_size):
        raise AssertionError("Oscar v1 teacher is not independently PRETRAINED full FP32 135M")
    mirror=s30.dense_from_paid(parent).eval().requires_grad_(False)
    oscar=encode_fp32(mirror.state_dict(),mode=MODE_LOCKED,prior_blob=parent.blob)
    archive=CB59(oscar.blob)
    if archive.meta["original_unique_parameter_count"]!=BASE_36M:
        raise AssertionError("Oscar v1 independently repacked 43M core source missing")
    oscar_cb61=to_cb61(oscar.blob)
    original=Stack(archive,oscar.blob,s142,dense=False)
    try:
        old_results=score_teacher_vs_student(teacher,original,tok,rows,copy,
                                             label="BEFORE_ACTUAL_135M_DISTILLATION")
        if sum(int(r["protected_S142_old_full_EOS"]) for r in old_results)!=6:
            raise AssertionError("Oscar v1 original 43M S142 reference must reproduce its previously measured 6/8 gold TRAIN, NOT falsely claim 8/8")
    finally:
        original.close()
    student=Stack(archive,oscar.blob,s142,dense=False)
    # The original S93 centroid/codebooks and trained GT3/9 controllers are
    # all still frozen; S142 physical up channels + existing paid Q2 train.
    for module in [student.model,student.inner,student.bridge,
                   student.first,student.later,student.ptr]:
        module.requires_grad_(False)
    student.progressive.down.requires_grad_(False)
    student.progressive.up.requires_grad_(True)
    student.full.encoder.requires_grad_(True)
    student.full.decoder.requires_grad_(True)
    student.full.gates.requires_grad_(True)
    student.full.mix_logits.requires_grad_(True)
    teacher_train=[]
    with torch.inference_mode():
        for row in rows[:TRAIN_N]:
            logits=teacher(input_ids=ids_for(tok,row["question"]),
                           use_cache=False).logits[0,-1].detach().cpu().float()
            teacher_train.append(logits)
    fp32_teacher_hashes=[hashlib.sha256(x.numpy().tobytes()).hexdigest()
                         for x in teacher_train]
    param_group=[
        {"params":[student.progressive.up],"lr":LR_UP},
        {"params":list(student.full.parameters()),"lr":LR_Q2},
    ]
    opt=torch.optim.Adam(param_group)
    initial_up=student.progressive.up.detach().cpu().clone()
    initial_gate=student.full.gates.detach().cpu().clone()
    trace=[]
    try:
        for step in range(STEPS):
            idx=step%TRAIN_N
            row=rows[idx]
            tlogits=teacher_train[idx]
            clear(student.sources)
            opt.zero_grad(set_to_none=True)
            slogits=real_student_first_logits(student,tok,row)
            dist=F.kl_div(F.log_softmax(slogits[None]/T,dim=-1),
                          F.softmax(tlogits[None]/T,dim=-1),
                          reduction="batchmean")*T*T
            teacher_top1=int(tlogits.argmax())
            first_ce=F.cross_entropy(slogits[None],torch.tensor([teacher_top1]))
            # Small original TRAIN identity retention rehearsal only;
            # full protected J7 all456+128 still must be separately checked.
            protected_first_ce=F.cross_entropy(
                slogits[None],torch.tensor([row["expected"][0]]))
            loss=dist+0.15*first_ce+0.035*protected_first_ce
            if not torch.isfinite(loss):
                raise AssertionError("Oscar v1 true FP32 teacher distill produced nonfinite loss")
            loss.backward()
            grads={
              "real_12layer_progressive_up_grad_norm":
                  float(student.progressive.up.grad.norm()),
              "real_full12_GreenTunnel_gate_grad_norm":
                  float(student.full.gates.grad.norm()),
            }
            if not all(v>0 and torch.isfinite(torch.tensor(v)) for v in grads.values()):
                raise AssertionError("Oscar v1 genuine teacher FP32 gradient did not reach both trained original mechanisms")
            torch.nn.utils.clip_grad_norm_(
                [student.progressive.up,*student.full.parameters()],1.0,
                error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                student.full.gates.clamp_(-0.25,.25)
            record={
              "step":step+1,"original_pretrained_teacher_TRAIN_case":idx,
              "original_pretrained_teacher_logits_SHA256":fp32_teacher_hashes[idx],
              "cross_model_FP32_teacher_soft_KL":float(dist.detach()),
              "cross_model_FP32_teacher_first_token_CE":float(first_ce.detach()),
              "S142_original_first_token_rehearsal":float(protected_first_ce.detach()),
              "actual_real_FP32_135M_teacher_loss":float(loss.detach()),
              **grads,
            }
            trace.append(record)
            if (step+1)%4==0:
                log("OSCAR_V1_ACTUAL_135M_to_43M_joint_full12_GT_TRAIN_STEP",
                    step=step+1,loss=round(record[
                      "actual_real_FP32_135M_teacher_loss"],5),
                    active_adaptor_grad=round(grads[
                      "real_12layer_progressive_up_grad_norm"],4),
                    shared_full12_GT_grad=round(grads[
                      "real_full12_GreenTunnel_gate_grad_norm"],4))
            clear(student.sources)
        update_up=(student.progressive.up.detach().cpu()-initial_up)
        if not torch.any(update_up!=0):
            raise AssertionError("Oscar v1 optimizer did not really update original trained 43M active adapter weights")
        updated=dict(s142)
        updated.update(student.progressive.export_active_fp16())
        updated.update(export_compact_q2(student.full))
        if set(updated)!=set(s142) or len(updated)!=36:
            raise AssertionError("Oscar v1 tried to add unpaid student GT or alter original source")
        changed=[key for key in updated if not torch.equal(updated[key],s142[key])]
        if WEIGHTS_KEY not in changed:
            raise AssertionError("Oscar v1 FP16 rounding erased all real original 43M trained adapter updates")
        for key in s142:
            if key not in {WEIGHTS_KEY} | set(
                    k for k in s142 if k.startswith(PREFIX)):
                if not torch.equal(s142[key],updated[key]):
                    raise AssertionError("Oscar v1 changed original frozen GT3/9 or S142 controllers: "+key)
        sidepath=out/"OSCAR_V1_REAL_135M_TEACHER_TRAINED_43M_FP16_GT12.safetensors"
        save_file(updated,str(sidepath))
        restored=load_file(str(sidepath))
        if any(not torch.equal(updated[k],restored[k]) for k in updated):
            raise AssertionError("Oscar v1 physical FP16 packed-GT12 trainer reload failed")
        log("OSCAR_V1_ACTUAL_TRAIN_COMPLETED_PHYSICALLY",
            real_steps=STEPS,changed_sidecar_tensors=changed,
            learned_active_fp16_coeffs_updated=
                int(torch.count_nonzero(restored[WEIGHTS_KEY]!=s142[WEIGHTS_KEY])),
            true_full12_GT_original_FP32_delta=float(
                (student.full.gates.detach().cpu()-initial_gate).norm()))
    finally:
        student.close()
    reload_student=Stack(archive,oscar.blob,restored,dense=False)
    try:
        after=score_teacher_vs_student(teacher,reload_student,tok,rows,copy,
                                        label="AFTER_ACTUAL_135M_DISTILLATION_RELOAD")
    finally:
        reload_student.close()
    results={
      **precommit,"model_origin":"REAL INDEPENDENT ORIGINAL HF TRAINED SmolLM2-135M-Instruct FP32",
      "completed_real_FP32_teacher_30layer_to_Calibri_12layer":True,
      "source_student_unique":BASE_36M,
      "student_structural_params":TOTAL_43M,
      "training_steps_real":STEPS,
      "actual_teacher_logits_SHA256":fp32_teacher_hashes,
      "actual_real_training_trace":trace,
      "actual_changed_S142_physically_saved_sidecar_keys":changed,
      "full_teacher_old_vs_candidate_generated":list(zip(old_results,after)),
      "old_FP32_teacher_first_token_matches":sum(int(x["student_first_matches_original_135M_FP32"])
                                                 for x in old_results),
      "new_FP32_teacher_first_token_matches":sum(int(x["student_first_matches_original_135M_FP32"])
                                                 for x in after),
      "old_S142_full_EOS_retained_count":sum(int(x["protected_S142_old_full_EOS"]) for x in after),
      "old_s142_retention_protected_full456plus128_not_run":True,
      "independent_eval_not_run":True,
      "FP32_teacher_full_behavior_copied_on_heldout_not_proven":True,
      "candidate_accepted":False,
    }
    phys=paid_output(out,args,oscar_cb61,sidepath,results)
    results["real_inference_physical_package"]={
       "nominal_bpp":phys["nominal_bpp"],
       "conservative_bpp":phys["conservative_excluding_zero_seed_growth_bpp"],
       "nominal_under_0p7":phys["nominal_under_0p7"],
       "conservative_under_0p7":phys["conservative_under_0p7"],
       "complete_inference_bytes":phys["complete_real_inference_bytes"],
    }
    dump(out/"OSCAR_V1_INDEPENDENT_TRAINED_FP32_TEACHER_TO_43M_RESULT.json",results)
    log("OSCAR_V1_FINAL_ACTUAL_INDEPENDENT_FP32_135M_TO_43M",
        teacher="original HF pretrained 30x576",
        student="actual original CALIBRI GT12 12x320",
        trained_steps=STEPS,
        before_FP32_first_token_matches=results["old_FP32_teacher_first_token_matches"],
        after_FP32_first_token_matches=results["new_FP32_teacher_first_token_matches"],
        old_original_S142_correct_full_remaining=results["old_S142_full_EOS_retained_count"],
        max_old=CHECK_N,
        nominal_bpp=phys["nominal_bpp"],
        conservative_bpp=phys["conservative_excluding_zero_seed_growth_bpp"],
        blind_QA_not_run=True,
        deployment_accepted=False)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for arg in ("S93_CB61","S93_sidecar","S142_FP16","S94_bank",
                "S94_report","S95_router","S95_report",
                "S98_copy_net","S98_report","out"):
        ap.add_argument(arg,type=Path)
    run(ap.parse_args())
