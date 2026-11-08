"""S31: paired 36M FP32 training-only old-skill guard, same S26 start as S30.

This experiment directly addresses the observed S30 TRAIN regression 6/8 -> 4/8.
The pilot uses exactly 192 attempted dense FP32 steps with 3:1 new/old
curriculum, adds a rotating old-TRAIN loss to each new-noun step and
precommits a block-level safety rollback if ANY of the 6 initially
correct old TRAIN probe cases become wrong (strict entire answer+EOS).
Rollback includes BOTH weights and AdamW optimizer state.

Critical: these 8 old answers and 16 novel answers are TRAIN-only
probes, NOT a VALID or QA measure. No import or use of S29 frozen QA;
no S26 inspected QA; no model comparison from a future heldout answer.
A guarded TRAIN success cannot prove generalization or FP32 parity.
No Jessica ordinary chat tests or LIVE changes.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch

from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from v186.pairs import build_pairs
from v174.run import save

START_SHA=s30.SOURCE_SHA
TRAIN_STEPS=s30.STEPS
LR=s30.LR
BLOCK_SIZE=24
OLD_GUARD_WEIGHT=0.5
SCHEDULE=("new","new","old","new")
S29_SHA=s30.S29_FROZEN_SHA
assert TRAIN_STEPS%BLOCK_SIZE==0

def full_correct_ids(probe,kind):
    return {row["id"] for row in probe["answers"]
            if row["kind"]==kind and row["strict_correct_full_EOS"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(3030)
    torch.set_num_threads(2)
    blob=a.source.read_bytes()
    if len(blob)!=SOURCE_BYTES or hashlib.sha256(blob).hexdigest()!=START_SHA:
        raise AssertionError("Must start from one sealed 36M S26 archive")
    cb=CB59(blob)
    if cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Different architecture")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    if len(old)!=24 or len(new)!=32:
        raise AssertionError("Incomplete old or new TRAIN")
    model=s30.dense_from_paid(cb)
    logits_diff=s30.double_check_original_logits(cb,model,tok)
    before=s30.train_probe(model,tok,new,old)
    old_baseline=full_correct_ids(before,"historical_TRAIN")
    if len(old_baseline)!=6:
        raise AssertionError("Expected S30 fixed old TRAIN 6/8 baseline")
    source_sha=s30.snapshot_sha(model)
    optim=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.)
    attempted=0
    accepted_blocks=0
    reverted_blocks=0
    actual_grad_steps=0
    training_schedule={"new":0,"old":0}
    trace=[]
    for block in range(TRAIN_STEPS//BLOCK_SIZE):
        weights_before=copy.deepcopy(model.state_dict())
        state_before=copy.deepcopy(optim.state_dict())
        block_grad=0
        for local_step in range(BLOCK_SIZE):
            step=block*BLOCK_SIZE+local_step
            kind=SCHEDULE[step%len(SCHEDULE)]
            idx=training_schedule[kind]
            training_schedule[kind]+=1
            pair=(new if kind=="new" else old)[idx%(len(new) if kind=="new" else len(old))]
            optim.zero_grad(set_to_none=True)
            loss=s30.train_loss(model,pair,kind=="new")
            if kind=="new":
                guard=old[(step*7)%len(old)]
                loss=loss+OLD_GUARD_WEIGHT*s30.train_loss(model,guard,False)
            if not torch.isfinite(loss):
                raise AssertionError("Invalid actual TRAIN loss")
            loss_before=float(loss.detach())
            loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True))
            if not norm>0:raise AssertionError("S31 no real full-FP32 gradient")
            optim.step()
            block_grad+=1
            attempted+=1
            if (local_step+1)==BLOCK_SIZE:
                trace.append({"attempt":attempted,"kind":kind,
                              "TRAIN_loss_before":loss_before,
                              "full_FP32_grad_norm":norm})
        guard_probe=s30.train_probe(model,tok,new,old)
        remaining=full_correct_ids(guard_probe,"historical_TRAIN")
        # Primary task's original 6 exact answers must all remain correct,
        # not just their count. Note this protects only 8 TRAIN probes.
        accept=old_baseline.issubset(remaining)
        if accept:
            accepted_blocks+=1
            actual_grad_steps+=block_grad
        else:
            model.load_state_dict(weights_before)
            optim.load_state_dict(state_before)
            reverted_blocks+=1
        trace.append({
            "block":block+1,"attempted_steps":attempted,
            "accepted":accept,
            "old_exact_FULL_EOS_before":len(old_baseline),
            "old_exact_FULL_EOS_candidate":len(remaining),
            "old_baseline_answer_ids_lost_in_candidate":sorted(old_baseline-remaining),
            "rollback_weights_and_optimizer":not accept,
            "new_train_full_candidate":
                guard_probe["per_kind"]["new_noun_TRAIN"]["full_correct"],
        })
        print(json.dumps({"event":"S31_GUARDED_FP32_TRAIN_BLOCK",
                          **trace[-1]},ensure_ascii=False),flush=True)
        del weights_before,state_before
    after=s30.train_probe(model,tok,new,old)
    retained=full_correct_ids(after,"historical_TRAIN")
    if not old_baseline.issubset(retained):
        raise AssertionError("Old accepted TRAIN full answers lost after rollback")
    after_sha=s30.snapshot_sha(model)
    report={
        "stage":"S31_GUARDED_36M_DENSE_FP32_OLD_TRAIN_REPLAY",
        "S30_control_run_id":42481557658,
        "S30_control_old_TRAIN_6_of_8_to_4_of_8":True,
        "same_S26_source_sha256":START_SHA,
        "matched_initial_FP32_vs_paid_logits_max_abs":logits_diff,
        "one_architecture_36M_unique_parameters":SOURCE_PARAMS,
        "dense_FP32_NOT_under_0_7_bpp":True,
        "S29_frozen_QA_sha_but_QA_NOT_imported":S29_SHA,
        "already_inspected_S26_QA_NOT_used":True,
        "training_only_no_holdout_model_selection":True,
        "script_uses_only_precommitted_TRAIN_guard_outputs":True,
        "attempted_FP32_steps":attempted,
        "real_nonzero_gradient_steps_in_accepted_blocks":actual_grad_steps,
        "accepted_24step_blocks":accepted_blocks,
        "rolled_back_24step_blocks":reverted_blocks,
        "precommitted_blocks":TRAIN_STEPS//BLOCK_SIZE,
        "initial_FP32_model_SHA256":source_sha,
        "final_FP32_model_SHA256":after_sha,
        "initial_old_TRAIN_correct_answer_ids":sorted(old_baseline),
        "all_initially_correct_old_TRAIN_probes_preserved":True,
        "TRAIN_results_before":before,
        "TRAIN_results_after":after,
        "all_block_decisions_and_TRAIN_losses":trace,
        "no_claim_of_generalization_or_preserving_all_historical_skills":True,
        "must_compare_heldout_only_on_sealed_final_checkpoints":True,
    }
    save(a.out/"ACTUAL_S31_GUARDED_FP32_36M_TRAIN_ONLY.json",report)
    print(json.dumps({
        "stage":report["stage"],"attempts":attempted,
        "accepted_blocks":accepted_blocks,
        "reverted_blocks":reverted_blocks,
        "before":before["per_kind"],"after":after["per_kind"],
        "source_changed":source_sha!=after_sha,
        "old_original_exact_answers_preserved":True,
        "S29_QA_unopened":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
