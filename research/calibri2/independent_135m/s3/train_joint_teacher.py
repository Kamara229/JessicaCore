#!/usr/bin/env python3
"""S3: actual real-pretrained-FP32-teacher -> paid CB59 core + S87/S126 GT TRAIN.

All teacher weights/logits are TRAIN-only ephemeral. Inference package is
ONE physically packed CB59 source + ONE real FP16 neural sidecar + manifest.
Strict 0.7 bpp is measured including all source and neural weights.
NO NRA, S3 grammatical adapters, Jessica chat, or model promotion.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from direct_codebook_v59 import CB59
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from v176.operators import clear
from v185.codec import replace_discrete
from v196.recover import build_trainable_from_paid_source
from calibri2.independent_135m.s2.gt_capacity import (
    MultiInternalGreenTunnel, pin_positions, pack_modules,
)
from calibri2.independent_135m.arch_no_nra.architecture import (
    FullDepthGreenTunnel, PromptOnlyRouter, ParallelSourceSlots
)

SOURCE_SHA = "50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
TRAIN_STEPS = 48
CYCLE_LENGTH = 16
TRAIN_OUTPUT_TOKENS = 8
TEACHER_TEMP = 2.0
MAX_BPP = 0.7
LR_CENTROIDS = 0.0002
LR_GREEN_TUNNEL = 0.0010
# Frozen previously exposed TRAIN only + newly authored TRAIN before this job.
TRAIN=(
    "Answer with one word: What color is fresh snow?",
    "Finish in one word: Water turns into ice when it ...",
    "Answer with one digit: How much is 2 plus 2?",
    "Answer with one word: What is the opposite of hot?",
    "Answer with one word: What do bees make?",
    "Answer with one word: What is the opposite of up?",
    "Complete briefly: The sun rises in the ...",
    "Answer with one word: What color is a ripe banana?",
    "Answer with one word only. What color is fresh snow?",
    "Answer with one word only. What is the opposite of hot?",
    "Answer with one word only. What color is a ripe banana?",
    "Answer with one digit only. What is 2 plus 2?",
    "Answer with one word only. What is the opposite of up?",
    "Answer with one word only. What do bees make?",
    "Answer with one word only. In what direction does the sun rise?",
    "Answer with one digit only. How many sides does a triangle have?",
    "Answer with one word only. What is the opposite of yes?",
    "Answer with one word only. What is the opposite of black?",
    "Answer with one word only. What color is most grass?",
    "Answer with one word only. What sound does a cat make?",
    "Finish this phrase: A bird can ...",
    "State one word for the opposite of dark.",
    "Complete: The ocean is usually ...",
    "Write the number obtained by adding five and five.",
    "Answer with one word: A young dog is a ...",
    "Continue in one word: Grass grows in the ...",
    "Write the opposite of early as one word.",
    "What is the color of clear blue sky? Use one word.",
    "Complete the phrase: A fish can ...",
    "Write the result of ten divided by two, using a digit.",
    "Finish in one word: It is cold in ...",
    "Provide a one-word synonym for quick.",
)
CHECK_IDS = (0, 2, 3, 4, 6, 7, 18, 31)


def save(path,value):
    Path(path).write_text(json.dumps(
        value,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def emit(stage,**values):
    print(json.dumps({"stage":stage,**values},ensure_ascii=False),flush=True)


def context(tok,prompt):
    src=tok.apply_chat_template(
        [{"role":"user","content":prompt}],tokenize=False,
        add_generation_prompt=True)
    seq=tok.encode(src,add_special_tokens=False)
    if not 1 <= len(seq)<=160:
        raise AssertionError("TRAIN context malformed")
    return torch.tensor([seq],dtype=torch.long)


@torch.inference_mode()
def greedy(model,source,tok,ctx):
    ids=ctx
    output=[]
    eos=False
    for _ in range(TRAIN_OUTPUT_TOKENS):
        if source is not None:
            clear(source)
        logits=model(input_ids=ids,use_cache=False).logits[0,-1]
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError("Nonfinite greedy logits")
        choice=int(logits.argmax())
        if choice==tok.eos_token_id:
            eos=True
            output.append(choice)
            break
        output.append(choice)
        ids=torch.cat((ids,ids.new_tensor([[choice]])),dim=1)
    if source is not None:
        clear(source)
    return {"ids":output,"EOS_seen":eos,
            "text":tok.decode(
                [t for t in output if t!=tok.eos_token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)}


@torch.inference_mode()
def build_teacher_train(fp32,tok,contexts):
    targets=[]
    for idx,ctx in enumerate(contexts):
        answer=greedy(fp32,None,tok,ctx)
        if not answer["ids"]:
            raise AssertionError("Empty trained FP32 reference answer")
        seq=torch.cat((ctx,ctx.new_tensor([answer["ids"][:-1]])),1)
        out=fp32(input_ids=seq,use_cache=False,output_hidden_states=True)
        off=ctx.shape[1]-1
        n=len(answer["ids"])
        logits=out.logits[0,off:off+n,:].detach().half().contiguous()
        hidden={layer:out.hidden_states[layer][0,off:off+n,:].detach()
                     .half().contiguous()
                for layer in (0,10,20,30)}
        if (logits.shape[0]!=n or set(hidden)!={0,10,20,30}
            or not all(x.shape==(n,576) for x in hidden.values())):
            raise AssertionError("Original 30-layer teacher causal target misalignment")
        targets.append({
          "ctx":ctx,
          "gold_generated_ids":list(answer["ids"]),
          "answer":answer,
          "teacher_logits_F16_TRAIN_only":logits,
          "teacher_hidden_F16_TRAIN_only":hidden,
        })
        if idx%8==7:
            emit("S3_teacher_original_FP32_TRAIN_cached",
                 prompts=idx+1,
                 example_first_tokens=answer["text"][:100],
                 last_candidate_seq_length=n)
    return targets


@torch.inference_mode()
def diagnose_real_generation(student,sources,tok,train):
    cases=[]
    for i in CHECK_IDS:
        got=greedy(student,sources,tok,train[i]["ctx"])
        teacher=train[i]["answer"]
        real_id_equal=(got["ids"]==teacher["ids"] and
                       got["EOS_seen"]==teacher["EOS_seen"])
        first_equal=bool(got["ids"] and teacher["ids"] and
                         got["ids"][0]==teacher["ids"][0])
        cases.append({
            "TRAIN_index":i,
            "teacher_FP32_full":teacher,
            "student":got,
            "same_first_token_as_FP32":first_equal,
            "same_full_token_sequence_including_EOS":real_id_equal,
        })
    return {
      "correct_FP32_first_tokens":sum(x["same_first_token_as_FP32"] for x in cases),
      "correct_FP32_full_sequences":sum(x["same_full_token_sequence_including_EOS"] for x in cases),
      "teacher_output_is_NOT_independent_ground_truth":True,
      "TRAIN_only_examples":list(CHECK_IDS),
      "raw_pairs":cases,
    }


def student_step(student,sources,local,full,optimizer,teacher_row):
    clear(sources)
    optimizer.zero_grad(set_to_none=True)
    n=len(teacher_row["gold_generated_ids"])
    seq=torch.cat((teacher_row["ctx"],
                   teacher_row["ctx"].new_tensor([
                     teacher_row["gold_generated_ids"][:-1]])),1)
    off=teacher_row["ctx"].shape[1]-1
    try:
        out=student(input_ids=seq,use_cache=False,
                    output_hidden_states=True)
        logits=out.logits[:,off:off+n,:].reshape(n,-1).float()
        teacher=teacher_row["teacher_logits_F16_TRAIN_only"].float()
        if logits.shape!=teacher.shape:
            raise AssertionError("Original/paid student distillation alignment failed")
        teacher_tokens=torch.tensor(teacher_row["gold_generated_ids"],
                                     dtype=torch.long)
        token_loss=F.cross_entropy(logits,teacher_tokens)
        temp=TEACHER_TEMP
        kl=F.kl_div(F.log_softmax(logits/temp,dim=-1),
                    F.softmax(teacher/temp,dim=-1),
                    reduction="none").sum(-1).mean()*(temp**2)
        hidden_losses=[]
        for layer in (0,10,20,30):
            h=out.hidden_states[layer][0,off:off+n,:]
            ref=teacher_row["teacher_hidden_F16_TRAIN_only"][layer].float()
            hidden_losses.append(
                (1-F.cosine_similarity(h.float(),ref,dim=-1)).mean())
        hidden=torch.stack(hidden_losses).mean()
        loss=.6*token_loss+.3*kl+.7*hidden
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite actual joint GT loss")
        loss.backward()
        all_params=[s.centroids for s in sources.values()]
        paid_grad=sum(s.centroids.grad is not None for s in sources.values())
        if paid_grad!=211:
            raise AssertionError("Not all 211 genuinely paid CB59 centroids trained")
        grads=[p for p in local.parameters()]+[p for p in full.parameters()]
        if (any(block.up.weight.grad is None for block in local.blocks.values()) or
            full.gates.grad is None or not bool(torch.count_nonzero(full.gates.grad))):
            raise AssertionError("No real GT-local or all30-layer gradient")
        norm=torch.nn.utils.clip_grad_norm_(all_params+grads,
                  1.0,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for s in sources.values():
                s.centroids.clamp_(-16,16)
            for p in grads:
                if not bool(torch.isfinite(p).all()):
                    raise AssertionError("Nonfinite physically paid GT parameters")
        return {"ce":float(token_loss.detach()),
                "kl":float(kl.detach()),
                "hidden_1minusCos":float(hidden.detach()),
                "total":float(loss.detach()),
                "norm_before_clip":float(norm),
                "paid_211_CB59_grad":paid_grad,
                "actual_gt_gate_grad":True}
    finally:
        clear(sources)


def save_full_candidate(source_raw,sources,local,full,router,slots,out,cycle):
    folder=out/f"cycle_{cycle:02d}"
    folder.mkdir(parents=True,exist_ok=True)
    raw,seal=replace_discrete(CB59(source_raw),sources,strict=False)
    if len(raw)!=11229303 or seal["complete_bpp"]>.700:
        raise AssertionError("Modified original trained FP32→CB59 altered physical source geometry")
    path=folder/"S3_PHYSICALLY_TRAINED_ORIGINAL_135M.cb59"
    path.write_bytes(raw)
    model_sidecar=folder/"S3_GREEN_TUNNEL_ALL30_LOCAL5_ROUTER_SLOTS_FP16.safetensors"
    count,tensors=pack_modules(model_sidecar,local,full,router,slots)
    summary={
        "stage":"S3_jnt_teacher_ACTUAL_paid_research_candidate",
        "source":path.name,
        "source_sha256":hashlib.sha256(raw).hexdigest(),
        "source_full_original_211_matrices":seal["all_2d_original_matrices"],
        "original_pretrained_unique_params":EXPECTED_UNIQUE,
        "original_input_S0_sha256":SOURCE_SHA,
        "FP16_saved_sidecar":model_sidecar.name,
        "sidecar_trainable_params_physically_stored":count,
        "sidecar_tensor_fields":tensors,
        "sidecar_bytes":model_sidecar.stat().st_size,
        "source_bytes":len(raw),
        "manifest_bytes":0,
        "fully_paid_bytes":0,
        "all_inclusive_bpp":0.,
        "physically_below_0p7":False,
        "router_is_untrained":True,
        "parallel_slots_are_untrained_and_unarmed":True,
        "no_s1_profile_or_FP32_teacher_or_NRA_adapter_inside_package":True,
        "never_deploy_without_new_blind_EVAL_and_native_CPU":True,
    }
    manifest=folder/"S3_PHYSICAL_INFERENCE_MANIFEST.json"
    for _ in range(25):
        save(manifest,summary)
        b=manifest.stat().st_size
        physical=len(raw)+model_sidecar.stat().st_size+b
        updated=dict(summary,
           manifest_bytes=b,
           fully_paid_bytes=physical,
           all_inclusive_bpp=8*physical/EXPECTED_UNIQUE,
           physically_below_0p7=(8*physical/EXPECTED_UNIQUE<=MAX_BPP))
        if updated==summary:
            break
        summary=updated
    else:
        raise AssertionError("Physical manifest size did not converge")
    save(manifest,summary)
    if (manifest.stat().st_size!=summary["manifest_bytes"] or
        summary["fully_paid_bytes"]!=len(raw)+model_sidecar.stat().st_size+
           manifest.stat().st_size or
        not summary["physically_below_0p7"]):
        raise AssertionError("S3 all-in physical source and trainable GT violate bpp cap")
    return summary


def run(source_path,out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(13531117)
    torch.set_num_threads(2)
    raw=Path(source_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise AssertionError("Only original 135M independently packed S0 source allowed")
    precommit={
        "phase":"actual 135M pretrained-FP32 30-layer GT joint TRAIN only",
        "true_training_steps":TRAIN_STEPS,
        "steps_per_cycle":CYCLE_LENGTH,
        "train_prompts":list(TRAIN),
        "train_prompt_SHA256":[hashlib.sha256(x.encode()).hexdigest()
                               for x in TRAIN],
        "FP32_original_trained_pinned_revision":MODEL_SHA,
        "original_unique_params":EXPECTED_UNIQUE,
        "source_SHA":SOURCE_SHA,
        "internal_local_GT_layer_positions":pin_positions(5),
        "internal_local_GT_rank":24,
        "recurrent_full30_GT_rank":32,
        "parallel_source_slots_rank":16,
        "router_untrained_and_disarmed":True,
        "NRA_present":False,
        "teacher_temperature":TEACHER_TEMP,
        "loss_weights":{"teacher_token_CE":0.6,
                        "full_vocab_KL":0.3,
                        "intermediate_hidden_layer_cosine":0.7},
        "centroid_learning_rate":LR_CENTROIDS,
        "gt_learning_rate":LR_GREEN_TUNNEL,
        "trained_model_quality_accepted_before_reading_EVAL":False,
        "new_blind_EVAL_never_read":True,
        "original_FP32_teacher_model_not_exported":True,
    }
    save(out/"S3_PRECOMMITTED_TRAIN_ONLY_BEFORE_LOADING_TEACHER.json",precommit)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    contexts=[context(tok,text) for text in TRAIN]
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).cpu().eval()
    if sum(x.numel() for x in teacher.parameters())!=EXPECTED_UNIQUE:
        raise AssertionError("Not the real original pretrained FP32 135M")
    references=build_teacher_train(teacher,tok,contexts)
    # Only teacher TRAIN targets are retained, no FP32 teacher needed for student.
    del teacher
    gc.collect()
    archive=CB59(raw)
    student,sources=build_trainable_from_paid_source(archive)
    if len(sources)!=211:
        raise AssertionError("Expected full original 211 2D operator codebooks")
    student.cpu().eval().requires_grad_(False)
    for s in sources.values():
        s.centroids.requires_grad_(True)
    width=student.config.hidden_size
    depth=student.config.num_hidden_layers
    if width!=576 or depth!=30:
        raise AssertionError("Original 135M geometry drift")
    local=MultiInternalGreenTunnel(width,pin_positions(5),24)
    full=FullDepthGreenTunnel(width,depth,32)
    local.install(student)
    full.install(student)
    router=PromptOnlyRouter()
    slots=ParallelSourceSlots(width,rank=16,slots=2)
    optimizer=torch.optim.Adam([
        {"params":[s.centroids for s in sources.values()],
         "lr":LR_CENTROIDS},
        {"params":list(local.parameters())+list(full.parameters()),
         "lr":LR_GREEN_TUNNEL},
    ])
    baseline=diagnose_real_generation(student,sources,tok,references)
    emit("S3_original_COMPRESSED_baseline",
         fp32_first=baseline["correct_FP32_first_tokens"],
         fp32_full=baseline["correct_FP32_full_sequences"],
         output_samples=[r["student"]["text"][:72] for r in baseline["raw_pairs"][:3]])
    records=[]
    quality=[{"cycle":0,"quality":baseline}]
    physical=[]
    max_full=baseline["correct_FP32_full_sequences"]
    max_first=baseline["correct_FP32_first_tokens"]
    accepted=0
    for step in range(TRAIN_STEPS):
        which=(step*7+(step//CYCLE_LENGTH)*3)%len(references)
        t0=time.monotonic()
        result=student_step(student,sources,local,full,optimizer,
                             references[which])
        row={"step":step+1,"cycle":step//CYCLE_LENGTH+1,
             "TRAIN_index":which,
             "wall_seconds":round(time.monotonic()-t0,2),
             **result}
        records.append(row)
        if (step+1)%4==0:
            emit("S3_joint_GT_real_gradient",
                 step=step+1,
                 loss=round(result["total"],4),
                 token_CE=round(result["ce"],4),
                 hidden_error=round(result["hidden_1minusCos"],4),
                 paid_211=result["paid_211_CB59_grad"],
                 rank24_local_blocks=5,rank32_recurrence_all30=True)
        if (step+1)%CYCLE_LENGTH==0:
            cycle=(step+1)//CYCLE_LENGTH
            got=diagnose_real_generation(student,sources,tok,references)
            fulln=got["correct_FP32_full_sequences"]
            firstn=got["correct_FP32_first_tokens"]
            old=quality[-1]["quality"]["raw_pairs"]
            lost=[x["TRAIN_index"] for x,y in zip(old,got["raw_pairs"])
                  if ((x["same_full_token_sequence_including_EOS"] and
                       not y["same_full_token_sequence_including_EOS"]) or
                      (x["same_first_token_as_FP32"] and
                       not y["same_first_token_as_FP32"]))]
            improved=(fulln>max_full or firstn>max_first)
            research_only_pass=improved and not lost
            if research_only_pass:
                accepted+=1
                max_full=max(max_full,fulln)
                max_first=max(max_first,firstn)
            pack=save_full_candidate(raw,sources,local,full,router,slots,
                                      out,cycle)
            quality.append({"cycle":cycle,
              "quality":got,
              "lost_previous_FP32_TRAIN_match_indices":lost,
              "meaningful_TRAIN_teacher_recovery":improved,
              "no_forgetting_guard_candidate_pass":research_only_pass,
              "candidate_model_accepted_for_deployment":False,
            })
            physical.append(pack)
            save(out/"S3_INCREMENTAL_ALL_GRADIENT_STEPS.json",records)
            save(out/"S3_INCREMENTAL_FULL_SEQUENCE_J7_TRAIN.json",quality)
            emit("S3_joint_GT_cycle_complete",
                 cycle=cycle,
                 FP32_TRAIN_first_matches=firstn,
                 FP32_TRAIN_full_sequences=fulln,
                 baseline_first=baseline["correct_FP32_first_tokens"],
                 improved_vs_prior=improved,
                 lost_previous_TRAIN_matches=lost,
                 physical_bpp=pack["all_inclusive_bpp"],
                 physically_paid=pack["physically_below_0p7"])
    best_final=quality[-1]["quality"]
    summary={
      "stage":"S3_real_135M_paid_GT_joint_distillation",
      "completed":True,
      "real_gradient_updates":len(records),
      "full_211_centroid_gradients_per_step":all(
          x["paid_211_CB59_grad"]==211 for x in records),
      "local_GT_trainable_blocks":5,
      "local_GT_rank":24,
      "recurrent_GT_all30_rank":32,
      "original_trained_FP32_pinned_revision":MODEL_SHA,
      "original_S0_physical_source_SHA":SOURCE_SHA,
      "original_unique_params":EXPECTED_UNIQUE,
      "teacher_fp32_trained_only_not_exported":True,
      "baseline":baseline,
      "quality_after_each_full_cycle":quality,
      "physical_after_each_full_cycle":physical,
      "last_cycle_train_CE":records[-1]["ce"],
      "last_cycle_train_full_vocab_KL":records[-1]["kl"],
      "last_cycle_intermediate_hidden_cosine_loss":records[-1]["hidden_1minusCos"],
      "last_cycle_TRAIN_teacher_first_token_matches":best_final[
          "correct_FP32_first_tokens"],
      "last_cycle_TRAIN_teacher_full_output_matches":best_final[
          "correct_FP32_full_sequences"],
      "meaningful_no_forgetting_TRAIN_checkpoints":accepted,
      "candidate_is_NOT_ACCEPTED_FOR_DEPLOYMENT":True,
      "all_inclusive_final_bpp":physical[-1]["all_inclusive_bpp"],
      "new_independent_EVAL_used":False,
      "native_C_SPEED_90pct_verified":False,
      "NRA_present":False,
      "no_Jessica_LIVE_main_Render_Supabase_APK":True,
      "next_step":"new frozen blind QA before accepting this architecture for inference; improve physical codec if GT insufficient",
    }
    save(out/"S3_FORTY_EIGHT_REAL_JOINT_TRAIN_STEPS_FINAL_REPORT.json",summary)
    emit("S3_FINAL_real_48step_joint_GT_core_TRAIN_only",
         fp32_teacher_first_baseline=baseline["correct_FP32_first_tokens"],
         fp32_teacher_first_final=best_final["correct_FP32_first_tokens"],
         fp32_teacher_full_baseline=baseline["correct_FP32_full_sequences"],
         fp32_teacher_full_final=best_final["correct_FP32_full_sequences"],
         bpp=physical[-1]["all_inclusive_bpp"],
         accepted_for_deployment=False,
         original_FP32_restoration_PROVEN=False)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("S0_original_trained_CB59",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.S0_original_trained_CB59,args.output)
