"""CALIBRI2 S1: test catastrophic forgetting vs frozen-pair rehearsal.

Start BOTH arms from exact same S0 128-step physically paid 22M checkpoint,
which already learned an opposite Yes/No TRAIN pair with correct EOS.
Teach a new opposite red/blue TRAIN binding pair for exactly 128 updates:
  no_rehearsal: only the new binding pair.
  balanced_rehearsal: alternate new pair and old original Yes/No pair.
Compare all four strict full-answer-plus-EOS outputs every 32 steps and after
reloading physically exported <=0.700 bpp CB59. TRAIN-only, not generalization.
No pretrained original FP32 weights, teacher, QA/DEV/VALID/TEST, or LIVE.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from transformers import AutoTokenizer

from direct_codebook_v59 import CB59
from binary_geometry_v64 import MODEL, MODEL_SHA
from v174.run import save
from v176.operators import clear
from v185.optim import propose_codes, apply_codes
from v185.codec import replace_discrete
from v186.pairs import build_pairs
from v186.fit import paired_loss
from calibri2.s0.bootstrap import CANDIDATES, build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s0.one_pair_learnability import snapshot

SEED=2021
STEPS=128
CHECKPOINTS=(0,32,64,96,128)
PAIR_IDS=(0,18)
LR=.008
MAX_BPP=.700
EXPECTED_PARAMS=22368320
EXPECTED_BYTES=1932195

def probe(student,sources,pairs,step,arm):
    by_pair={}
    for pair in pairs:
        by_pair[str(pair["pair_id"])]=snapshot(student,sources,pair)
    sides=[row["sides"][side]
        for row in by_pair.values() for side in ("left","right")]
    entry={"step":step,
       "strict_full_answer_plus_EOS_correct":sum(
            s["full_tokens_with_EOS_correct"] for s in sides),
       "first_word_correct":sum(s["first_token_correct"] for s in sides),
       "distinct_first_tokens":len(set(s["greedy_first_token"] for s in sides)),
       "both_previous_YesNo_answers_retained":by_pair["0"]["both_full_answers_and_EOS_correct"],
       "both_new_redblue_answers_correct":by_pair["18"]["both_full_answers_and_EOS_correct"],
       "per_training_pair":by_pair}
    print(json.dumps({"stage":"CALIBRI2_S1_REHEARSAL_CHECK",
       "arm":arm, "step":step,
       "full_EOS":entry["strict_full_answer_plus_EOS_correct"],
       "first_word_correct":entry["first_word_correct"],
       "unique_first_tokens":entry["distinct_first_tokens"],
       "old_pair_retained":entry["both_previous_YesNo_answers_retained"],
       "new_pair_complete":entry["both_new_redblue_answers_correct"]}),
       flush=True)
    return entry

def train_arm(arm,initial,cfg,pairs,out):
    torch.manual_seed(SEED)
    cb=CB59(initial)
    model,sources=create_small_paid_model(cb,cfg)
    optimizer=torch.optim.Adam([s.centroids for s in sources.values()],lr=LR)
    source_keys=sorted(sources)
    history=[];checks=[]
    for step in range(STEPS+1):
        if step in CHECKPOINTS:
            checks.append(probe(model,sources,pairs,step,arm))
        if step==STEPS:
            break
        old,new=pairs
        pair=new if (arm=="no_rehearsal" or step%2==0) else old
        target=sources[source_keys[step%len(source_keys)]]
        captured=[]
        def hook(_m,_inputs,weight):
            if not any(weight is w for w in captured):
                weight.retain_grad()
                captured.append(weight)
        cb_hook=target.register_forward_hook(hook)
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        try:
            loss,fields=paired_loss(model,target,pair)
            loss_before=float(loss.detach())
            loss.backward()
        finally:
            cb_hook.remove()
        if len(captured)!=1 or captured[0].grad is None:
            raise AssertionError("Cannot collect actual paid full-model gradient")
        proposed_gradient=captured[0].grad.detach().cpu().numpy().copy()
        if any(w.centroids.grad is None for w in sources.values()):
            raise AssertionError("All 43 stored neural operator dictionaries must train")
        grad_norm=float(torch.nn.utils.clip_grad_norm_(
            [s.centroids for s in sources.values()],1.,
            error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for src in sources.values():
                src.centroids.clamp_(-16,16)
        clear(sources)
        proposed=accepted=0
        if (step+1)%4==0:
            with torch.inference_mode():
                baseline,_=paired_loss(model,target,pair)
                loss_baseline=float(baseline)
                before=target().detach().cpu().numpy().copy()
            candidate=propose_codes(target,proposed_gradient,before)
            proposed=candidate["proposed_groups"]
            if proposed>512:
                raise AssertionError("Exceeded original paid index update budget")
            if proposed:
                with torch.no_grad():
                    apply_codes(target,candidate,True)
                clear(sources)
                with torch.inference_mode():
                    candidate_loss,_=paired_loss(model,target,pair)
                if np.isfinite(float(candidate_loss)) and float(candidate_loss)+1e-7<loss_baseline:
                    accepted=proposed
                else:
                    with torch.no_grad():
                        apply_codes(target,candidate,False)
        clear(sources)
        history.append({"step":step+1,"pair_id":pair["pair_id"],
            "domain":pair["domain"],"loss_before":loss_before,
            "source_gradient_norm":grad_norm,
            "target_first_token_margins":fields["first_token_logit_gaps"],
            "proposed_5bit_group_changes":proposed,
            "accepted_5bit_group_changes":accepted})
    packed,info=replace_discrete(cb,sources,strict=False)
    filename=f"CALIBRI2_S1_{arm}_128paid_steps.cb59"
    (out/filename).write_bytes(packed)
    sha=hashlib.sha256(packed).hexdigest()
    if len(packed)!=EXPECTED_BYTES or info["complete_bpp"]>MAX_BPP:
        raise AssertionError("Full physically paid CB59 size/rate changed")
    if sha!=info["sha256"]:
        raise AssertionError("Serialized physical source digest mismatch")

    # Critical: a result from float surrogate is not a valid claim unless
    # the SAME generated tokens survive physical archive serialization/reload.
    reloaded,reloaded_sources=create_small_paid_model(CB59(packed),cfg)
    reloaded_probe=probe(reloaded,reloaded_sources,pairs,STEPS,arm+"_reloaded")
    def observed(row):
        return [(str(k),side,v["generated"],v["expected"])
            for k,p in row["per_training_pair"].items()
            for side,v in p["sides"].items()]
    if observed(checks[-1])!=observed(reloaded_probe):
        raise AssertionError("PAID checkpoint reload altered actual TRAIN greedy tokens")
    report={"arm":arm,"starting_model_sha256":hashlib.sha256(initial).hexdigest(),
         "physical_final_SHA256":sha,"physical_original_parameter_count":EXPECTED_PARAMS,
         "physically_stored_bytes":len(packed),"physical_bpp":info["complete_bpp"],
         "source_uses_only_paid_5bit_groups_and_FP16_codebooks":True,
         "training_domains":["yesno_numeric","binding"],
         "frozen_24_pair_TRAIN_generator_selected_pair_ids":list(PAIR_IDS),
         "steps_precommitted":STEPS,
         "same_random_pretraining_source_as_s0":True,
         "both_arms_from_identical_S0_paid_128_checkpoint":True,
         "one_difference_rehearsal_strategy":True,
         "no_heldout_validation_or_original_FP32_teacher":True,
         "no_premature_claim_of_generalization":True,
         "strict_reloaded_equals_training_answers":True,
         "source_change":info,
         "TRAIN_only_snapshots":checks,
         "reloaded_final_TRAIN_only_answers":reloaded_probe,
         "update_curve":history}
    save(out/f"ACTUAL_CALIBRI2_S1_{arm}.json",report)
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("trained_s0_paid_model",type=Path)
    ap.add_argument("output_dir",type=Path)
    args=ap.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=args.trained_s0_paid_model.read_bytes()
    model_hash=hashlib.sha256(raw).hexdigest()
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=EXPECTED_PARAMS:
        raise AssertionError("Source does not have 22M unique paid neural parameters")
    if len(raw)!=EXPECTED_BYTES or 8*len(raw)/EXPECTED_PARAMS>MAX_BPP:
        raise AssertionError("S0 source not in physical paid budget")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    train=build_pairs(tok,"TRAIN")
    pairs=[train[i] for i in PAIR_IDS]
    if [(p["split"],p["domain"]) for p in pairs]!=[
        ("TRAIN","yesno_numeric"),("TRAIN","binding")]:
        raise AssertionError("Unexpected TRAIN pair contamination")
    config=build_config(next(c for c in CANDIDATES if c[0]=="small_320x6"))
    arms=[train_arm(strategy,raw,config,pairs,out) for strategy in
        ("no_rehearsal","balanced_rehearsal")]
    summary={"experiment":"CALIBRI2 S1 first continual learning after S0 paid checkpoint",
             "initial_128step_source_sha256":model_hash,
             "paired_comparison_from_identical_source":True,
             "QA_VALID_TEST_and_teacher_used":False,
             "no_weight_growth_or_model_size_increase":True,
             "frozen_exact_train_pairs":list(PAIR_IDS),
             "two_arms":[{"name":r["arm"],
                "old_pair_retained":r["TRAIN_only_snapshots"][-1]["both_previous_YesNo_answers_retained"],
                "new_pair_full_answers_correct":r["TRAIN_only_snapshots"][-1]["both_new_redblue_answers_correct"],
                "strict_four_training_answers_and_EOS":r["TRAIN_only_snapshots"][-1]["strict_full_answer_plus_EOS_correct"],
                "last_initial_train_logits_diversity":r["TRAIN_only_snapshots"][-1]["distinct_first_tokens"],
                "physical_bpp":r["physical_bpp"],
                "reloaded_model_verified":r["strict_reloaded_equals_training_answers"]}
                 for r in arms]}
    save(out/"ACTUAL_CALIBRI2_S1_REHEARSAL_VS_FORGETTING.json",summary)
    print(json.dumps({"stage":"CALIBRI2_S1_TRAIN_ONLY_AB_COMPLETED",**summary},
                     ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
