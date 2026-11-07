"""CALIBRI2 S2: fixed capacity ladder on 22M physically packed network.

Start from the verified S1 four-answer, 0.691bpp, 22M checkpoint.
Three predeclared TRAIN-only curricula:
  4 diverse opposing pairs x 384 steps (8 answers);
  8 opposing pairs x 512 steps (16 answers);
  all 24 opposing pairs x 1024 steps (48 answers).
Balanced rehearsal, rotating every active pair, sparse accepted 5bit
index updates plus all 43 FP16 dictionary gradients.
Never use heldout 1024 QA, FP32 teacher, or tests for optimizer selection.
All actual-answer milestones inspect complete greedy generations AND EOS.
Save each stage as a genuine 0.691bpp CB59 physical file. No growth,
no LIVE, no Jessica test. Not a proof of theoretical capacity ceiling.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v174.run import save
from v176.operators import clear
from v185.optim import propose_codes,apply_codes
from v185.codec import replace_discrete
from v186.pairs import build_pairs
from v186.fit import paired_loss
from v200.rollout import probe_greedy_train
from calibri2.s0.bootstrap import CANDIDATES,build_config
from calibri2.s0.train import create_small_paid_model

START_SHA="2b49660ab5084e5a6a43451621a3876c5cc169550d0d1c9c539b357e5e879e58"
PARAMETERS=22368320
PAID_SIZE=1932195
SEED=2021
STEPS=[384,512,1024]
STAGE_NAMES=("4pairs_8answers","8pairs_16answers","24pairs_48answers")
STAGE_PAIR_IDS=(
  (0,6,12,18),
  (0,1,6,7,12,13,18,19),
  tuple(range(24)),
)
LR=(0.008,0.006,0.004)
GREEDY_CAP=8
MAX_BPP=.7
CHECK_EVERY=128

def assess(model,sources,all_pairs,selected_ids,stage,step):
    chosen=[all_pairs[i] for i in selected_ids]
    result=probe_greedy_train(model,sources,chosen,max_new_tokens=GREEDY_CAP)
    correct=first=prefixes=repeats=0
    domains=defaultdict(lambda:{"full_EOS_correct":0,"first_correct":0,"total":0})
    first_hist=Counter()
    rows=[]
    for p in chosen:
        for side in ("left","right"):
            sample=result[p["pair_id"]]["sides"][side]
            predicted=sample["generated_tokens"]
            expected=sample["gold_tokens_with_EOS"]
            ok=sample["exact_with_EOS"]
            fst=bool(predicted) and predicted[0]==expected[0]
            correct+=int(ok)
            first+=int(fst)
            prefixes+=sample["correct_prefix_tokens"]
            repeats+=sample["adjacent_repetition_excess"]
            dom=domains[p["domain"]]
            dom["total"]+=1;dom["full_EOS_correct"]+=int(ok)
            dom["first_correct"]+=int(fst)
            if predicted:first_hist[predicted[0]]+=1
            rows.append({
                "pair_id":p["pair_id"],"domain":p["domain"],"side":side,
                "actual_generated_ids":predicted,"expected_gold_ids":expected,
                "exact_full_and_EOS":ok,"first_token_correct":fst,
                "prefix_gold_correct_tokens":sample["correct_prefix_tokens"],
                "repetition_excess":sample["adjacent_repetition_excess"],
            })
    if len(rows)!=2*len(selected_ids):
        raise AssertionError("TRAIN probe lost original balanced paired questions")
    data={"stage":stage,"step_in_stage":step,"active_pairs":len(selected_ids),
        "active_full_EOS_correct":correct,
        "active_first_correct":first,
        "total_active_graded_questions":len(rows),
        "trained_answer_first_token_diversity":len(first_hist),
        "actual_gold_prefix_tokens":prefixes,
        "repetition_excess":repeats,
        "per_domain":dict(domains),
        "all_answers":rows}
    print(json.dumps({"stage":"CALIBRI2_S2_ACTIVE_TRAIN_REAL_ANSWERS",
        "phase":stage,"step":step,
        "full_correct":correct,"total":len(rows),
        "first_correct":first,"diversity":len(first_hist),
        "by_domain":dict(domains)},ensure_ascii=False),flush=True)
    return data

def export_and_verify(original,model,sources,cfg,all_pairs,out,label):
    packed,info=replace_discrete(original,sources,strict=False)
    output=out/f"CALIBRI2_S2_{label}_22M_paid.cb59"
    output.write_bytes(packed)
    if len(packed)!=PAID_SIZE or info["complete_bpp"]>MAX_BPP:
        raise AssertionError("Physical full-neural model size changed/too large")
    checksum=hashlib.sha256(packed).hexdigest()
    if checksum!=info["sha256"]:
        raise AssertionError("Written archive SHA does not match packed source")
    # Validate unpacked-to-real-source at chosen TRAIN historical anchors.
    restored,restored_sources=create_small_paid_model(CB59(packed),cfg)
    anchors=(0,6,12,18)
    pre=assess(model,sources,all_pairs,anchors,label+"_in_memory",0)
    post=assess(restored,restored_sources,all_pairs,anchors,label+"_reloaded",0)
    for i,(a,b) in enumerate(zip(pre["all_answers"],post["all_answers"])):
        if a["actual_generated_ids"]!=b["actual_generated_ids"]:
            raise AssertionError(f"Physical serialization changed anchor answer {i}")
    del restored,restored_sources
    return {"physical_path":str(output),"sha256":checksum,
         "complete_bytes":len(packed),"physical_bpp":info["complete_bpp"],
         "source_changed":info,
         "full_anchor_greedy_sequences_identical_after_reloading":True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("s1_128_rehearsed_checkpoint",type=Path)
    ap.add_argument("out_dir",type=Path)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    src=args.s1_128_rehearsed_checkpoint.read_bytes()
    if hashlib.sha256(src).hexdigest()!=START_SHA:
        raise AssertionError("Wrong S1 physical 4/4-answer source")
    archive=CB59(src)
    if archive.meta["original_unique_parameter_count"]!=PARAMETERS:
        raise AssertionError("Original 22M architecture changed")
    if len(src)!=PAID_SIZE or 8*len(src)/PARAMETERS>MAX_BPP:
        raise AssertionError("Source no longer physically <=.7bpp")
    cfg=build_config(next(s for s in CANDIDATES if s[0]=="small_320x6"))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    all_pairs=build_pairs(tok,"TRAIN")
    if len(all_pairs)!=24 or len(set(p["pair_id"] for p in all_pairs))!=24:
        raise AssertionError("Only 24 precommitted mutually opposing TRAIN pairs allowed")
    model,sources=create_small_paid_model(archive,cfg)
    source_keys=sorted(sources)
    original_all=tuple(range(24))
    progress=[]
    history=[]
    global_step=0
    for phase,(name,ids,steps,rate) in enumerate(zip(
            STAGE_NAMES,STAGE_PAIR_IDS,STEPS,LR)):
        if len(set(ids))!=len(ids) or not set(ids)<=set(original_all):
            raise AssertionError("Incorrect frozen TRAIN curricula")
        optimizer=torch.optim.Adam([s.centroids for s in sources.values()],lr=rate)
        baseline=assess(model,sources,all_pairs,ids,name,0)
        audits=[baseline]
        accepted=attempted=0
        for step in range(steps):
            # Each active pair has equal frequency within each stage.
            # Keep previous skills in every following curriculum, not a
            # single-task fine-tune which already proved forgetting.
            pair=all_pairs[ids[step%len(ids)]]
            target_name=source_keys[global_step%len(source_keys)]
            source=sources[target_name]
            capture=[]
            def hook(_mod,_args,value):
                if not any(value is v for v in capture):
                    value.retain_grad()
                    capture.append(value)
            handle=source.register_forward_hook(hook)
            clear(sources)
            optimizer.zero_grad(set_to_none=True)
            try:
                loss,detail=paired_loss(model,source,pair)
                before=float(loss.detach())
                loss.backward()
            finally:
                handle.remove()
            if len(capture)!=1 or capture[0].grad is None:
                raise AssertionError("Full neural reconstruction gradient lost")
            gradient=capture[0].grad.detach().cpu().numpy().copy()
            if sum(s.centroids.grad is not None for s in sources.values())!=43:
                raise AssertionError("Not all original paid FP16 codebooks trained")
            grad_norm=float(torch.nn.utils.clip_grad_norm_(
                [s.centroids for s in sources.values()],
                1.,error_if_nonfinite=True))
            optimizer.step()
            with torch.no_grad():
                for s in sources.values():s.centroids.clamp_(-16.,16.)
            clear(sources)
            proposals=adopted=0
            if (step+1)%8==0:
                with torch.inference_mode():
                    current_loss,_=paired_loss(model,source,pair)
                    baseline_loss=float(current_loss)
                    current=source().detach().cpu().numpy().copy()
                candidate=propose_codes(source,gradient,current)
                proposals=candidate["proposed_groups"]
                if proposals>512:
                    raise AssertionError("Unpaid 5bit index edits not allowed")
                if proposals:
                    with torch.no_grad():
                        apply_codes(source,candidate,True)
                    clear(sources)
                    with torch.inference_mode():
                        new_loss,_=paired_loss(model,source,pair)
                    # A single-pair TRAIN loss criterion is an experiment,
                    # NOT a claimed guarantee against forgetting.
                    if np.isfinite(float(new_loss)) and float(new_loss)+1e-7<baseline_loss:
                        adopted=proposals
                    else:
                        with torch.no_grad():apply_codes(source,candidate,False)
            clear(sources)
            attempted+=proposals;accepted+=adopted
            global_step+=1
            history.append({"global_step":global_step,"phase":name,
                "pair_id":pair["pair_id"],"domain":pair["domain"],
                "TRAIN_loss_before":before,"paid_dictionary_grad_norm":grad_norm,
                "opposite_gold_logit_margins":detail["first_token_logit_gaps"],
                "paid_5bit_indices_proposed":proposals,
                "paid_5bit_indices_accepted":adopted})
            if (step+1)%CHECK_EVERY==0:
                audits.append(assess(model,sources,all_pairs,ids,name,step+1))
        if audits[-1]["step_in_stage"]!=steps:
            raise AssertionError("Missing fixed final TRAIN-only milestone")
        physical=export_and_verify(
            archive,model,sources,cfg,all_pairs,args.out_dir,name)
        stage_result={"phase":name,"phase_index":phase,
            "fixed_pair_ids":list(ids),"gradient_steps":steps,
            "learning_rate":rate,"all_steps_cumulative":global_step,
            "train_answers_from_milestones":audits,
            "candidate_5bit_edits":attempted,
            "accepted_5bit_edits":accepted,
            "physical_source":physical}
        progress.append(stage_result)
        save(args.out_dir/f"CALIBRI2_S2_TRAIN_{name}_ACTUAL.json",stage_result)
    # Whole 48 original TRAIN answers, with EOS; not sample selection.
    all_48=assess(model,sources,all_pairs,original_all,"final_all_24pairs",global_step)
    final=progress[-1]["physical_source"]
    report={
        "experiment":"CALIBRI2 S2 capacity ladder and nonforgetting rehearsal",
        "initial_S1_physical_model_sha256":START_SHA,
        "final_S2_physical_model_sha256":final["sha256"],
        "training_22M_unique_physical_parameters_only":PARAMETERS,
        "no_FP32_teacher_comparison_before_135M":True,
        "no_PUBLIC_QA_VALID_TEST_used_for_training_or_model_selection":True,
        "all_3_cumulative_stages_frozen_in_advance":True,
        "stages":progress,
        "actual_48_train_only_full_answer_check":all_48,
        "update_curve":history,
        "total_precommitted_gradient_steps":sum(STEPS),
        "final_paid_bits_per_parameter":final["physical_bpp"],
        "capacity_ceiling_cannot_be_declared_from_single_learning_curve":True,
        "neither_LIVE_nor_Jessica_changed":True,
    }
    save(args.out_dir/"ACTUAL_CALIBRI2_S2_22M_CAPACITY_LADDER.json",report)
    print(json.dumps({
        "stage":"CALIBRI2_S2_PAID_CAPACITY_LADDER_FINISHED",
        "full_gold_48":all_48["active_full_EOS_correct"],
        "first_gold_48":all_48["active_first_correct"],
        "unique_first_tokens":all_48["trained_answer_first_token_diversity"],
        "total_training_steps":sum(STEPS),
        "physical_bpp":final["physical_bpp"],
        "all_stage_full_actual_TRAIN_answers":[
            [s["phase"],
            s["train_answers_from_milestones"][-1]["active_full_EOS_correct"],
            s["train_answers_from_milestones"][-1]["total_active_graded_questions"]]
            for s in progress],
       },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
