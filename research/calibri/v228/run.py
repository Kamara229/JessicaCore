"""CALIBRI v228: full-answer TRAIN supervision with preservation protection.

v227 successfully preserved 31/48 exact full answers at steps 1,2,4,
but made no gains; by step 8 it lost case (15,left), while its original
anti-loop goal had no external-QA meaning (four of five repeats were
legitimate gold). This follow-up drops anti-loop optimization entirely.
It balances a real gold entire-answer loss for one of 17 incorrect
TRAIN cases against preservation on two previously-correct TRAIN
answers (one rotating and one fragile (15,left)), and opposing-pair
contrast. All grads apply to the original 211 paid codebooks. Strict
greedy full-answer 48-case TRAIN selection only; no DEV, VALID, QA,
TEST, extra inference weights, decoder edits, or LIVE modifications.
"""
import argparse
import json
from pathlib import Path

import torch
from v174.run import save
from v176.operators import clear
from v185.codec import replace_discrete
from v186.fit import paired_loss
from v196 import protocol as P
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v200.rollout import probe_greedy_train, summarize_sequences,sequence_no_regression_gate
from v221.run import V220_SOURCE_SHA256
from v226.run import authenticate,build_frozen_pairs
from v227.run import correct_train_samples,already_correct_full_answer_loss

STEPS=24
LR=1.5e-5
CLIP=1.0
ANCHOR=0.05
CHECKPOINT_STEPS=(2,4,8,12,16,20,24)
PAIR_WEIGHT=0.25
NEW_ANSWER_WEIGHT=0.40
PREVIOUSLY_CORRECT_WEIGHT=0.35

def training_sets(rollout,pair_map):
    correct=correct_train_samples(rollout,pair_map)
    exact={(c["pair_id"],c["side"]) for c in correct}
    incorrect=[]
    for p in sorted(pair_map.values(),key=lambda x:x["pair_id"]):
        if p["split"]!="TRAIN":
            raise AssertionError("Training examples outside TRAIN")
        for side in ("left","right"):
            key=(p["pair_id"],side)
            if key in exact:
                continue
            sample=p[side]
            incorrect.append({
                "pair_id":p["pair_id"],"side":side,
                "prompt":sample["prompt_ids"].detach().long().clone().cpu(),
                "target":sample["targets"].detach().long().clone().cpu()
            })
    if len(correct)!=31 or len(incorrect)!=17 or (
            {(x["pair_id"],x["side"]) for x in correct} &
            {(x["pair_id"],x["side"]) for x in incorrect}):
        raise AssertionError("Original 31 correct / 17 wrong TRAIN split changed")
    fragile=next((x for x in correct if x["pair_id"]==15 and x["side"]=="left"),None)
    if fragile is None:
        raise AssertionError("Original v227 regressed TRAIN pair 15 left not found")
    return correct,incorrect,fragile

def run(v220_path,v222_path,v224_path,v225_path,out):
    torch.manual_seed(228)
    torch.set_num_threads(2)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    archive,rollout,audit,cause=authenticate(v220_path,v222_path,v224_path,v225_path)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    selected,pairs,valid,origin=training_inputs(tok)
    if len(selected)!=20 or len(pairs)!=24 or len(valid)!=P.VALID_PAIRS:
        raise AssertionError("Frozen original public TRAIN protocol drift")
    pair_map={p["pair_id"]:p for p in pairs}
    if len(pair_map)!=24 or any(p["split"]!="TRAIN" for p in pairs):
        raise AssertionError("Only original 24 opposing TRAIN pairs permitted")
    correct,incorrect,fragile=training_sets(rollout,pair_map)
    frozen=build_frozen_pairs(rollout)
    base=summarize_sequences(frozen)
    assert len(base["exact"])==31 and base["repetition_excess"]==5
    save(out/"PRECOMMITTED_v228_protected_full_TRAIN_answers.json",{
        "start_source_sha":V220_SOURCE_SHA256,
        "frozen_correct_TRAIN_full_answers":len(correct),
        "frozen_incorrect_TRAIN_full_answers":len(incorrect),
        "fragile_previously_regressed_TRAIN_case":[15,"left"],
        "all_211_paid_codebooks_train_every_step":True,
        "steps":STEPS,"LR":LR,"checkpoints":CHECKPOINT_STEPS,
        "weights":{"new_full_answer":NEW_ANSWER_WEIGHT,
                   "prior_correct_full_answers":PREVIOUSLY_CORRECT_WEIGHT,
                   "opposing_pairs":PAIR_WEIGHT,
                   "source_centroid_anchor":ANCHOR},
        "no_direct_repeat_penalty":True,
        "selection": "only 48 whole greedy TRAIN answers with no regression",
        "expected_physical_archive_bpp_max":0.7,
        "original_5bit_codes_and_runtime_inference_unchanged":True,
        "no_DEV_VALID_TEST_QA_selection_or_training":True,
        "no_Jessica_LIVE_main_Render_Supabase_APK_modifications":True
    })
    student,paid=build_trainable_from_paid_source(archive)
    student.cpu().eval()
    if len(paid)!=211:
        raise AssertionError("Missing original paid layer dictionary")
    original_codes={n:s.codes.detach().clone() for n,s in paid.items()}
    original_centroids={n:s.centroids.detach().clone() for n,s in paid.items()}
    opt=torch.optim.Adam([x.centroids for x in paid.values()],lr=LR)
    history=[];checkpoints=[];best=None;best_score=(len(base["exact"]),
                  base["prefix_tokens"],-base["repetition_excess"])
    for step in range(1,STEPS+1):
        clear(paid)
        opt.zero_grad(set_to_none=True)
        new_case=incorrect[(step-1)%len(incorrect)]
        safe_case=correct[(step-1)%len(correct)]
        pair=pairs[(step-1)%len(pairs)]
        problem=already_correct_full_answer_loss(student,new_case)
        preserve_a=already_correct_full_answer_loss(student,fragile)
        preserve_b=already_correct_full_answer_loss(student,safe_case)
        paired,paired_info=paired_loss(student,None,pair)
        anchor=torch.stack([(s.centroids-original_centroids[n]).square().mean()
                           for n,s in paid.items()]).mean()
        loss=(NEW_ANSWER_WEIGHT*problem+
              PREVIOUSLY_CORRECT_WEIGHT*(preserve_a+preserve_b)/2+
              PAIR_WEIGHT*paired+ANCHOR*anchor)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite full answer TRAIN restoration")
        loss.backward()
        active=sum(x.centroids.grad is not None for x in paid.values())
        if active!=211:
            raise AssertionError("Not all physically paid 211 codebooks received gradients")
        grad_norm=float(torch.nn.utils.clip_grad_norm_(
            [x.centroids for x in paid.values()],CLIP,error_if_nonfinite=True))
        opt.step()
        with torch.no_grad():
            for s in paid.values():
                s.centroids.clamp_(-16,16)
        clear(paid)
        history.append({
            "step":step,"all_paid_211_codebooks_gradients":active,
            "incorrect_TRAIN_case":[new_case["pair_id"],new_case["side"]],
            "preserved_TRAIN_case":[safe_case["pair_id"],safe_case["side"]],
            "fragile_TRAIN_case_always_preserved":[15,"left"],
            "opposing_TRAIN_pair_id":pair["pair_id"],
            "training_objective_diagnostic_only":float(loss.detach()),
            "grad_norm":grad_norm,
        })
        print(json.dumps({"stage":"v228_paid_full_answer_training",
          "step":step,"all_211_paid":active,"incorrect_TRAIN":
          [new_case["pair_id"],new_case["side"]]}),flush=True)
        if step not in CHECKPOINT_STEPS:
            continue
        generated=probe_greedy_train(student,paid,pairs,max_new_tokens=12)
        summ=summarize_sequences(generated)
        gate=sequence_no_regression_gate(frozen,generated)
        clear(paid)
        packed,seal=replace_discrete(archive,paid,strict=False)
        if (len(packed)!=P.SOURCE_BYTES or
                seal["complete_bpp"]>0.7 or
                seal["actual_changed_5bit_group_indices"]!=0 or
                seal["all_2d_original_matrices"]!=211 or
                any(not torch.equal(s.codes,original_codes[n])
                    for n,s in paid.items())):
            raise AssertionError("Physical paid source geometry or size drift")
        score=(len(summ["exact"]),summ["prefix_tokens"],-summ["repetition_excess"])
        accept=(bool(gate["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"]) and
                score>best_score and
                seal["actually_changed_FP16_codebook_cells"]>0)
        status={
            "checkpoint_step":step,
            "full_TRAIN_correct":len(summ["exact"]),
            "full_TRAIN_correct_prefix_tokens":summ["prefix_tokens"],
            "full_TRAIN_adjacent_repeats":summ["repetition_excess"],
            "lost_original_31_correct_TRAIN":gate[
                "lost_original_complete_TRAIN_answers"],
            "no_full_TRAIN_regression":gate["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"],
            "physical_archive_bpp":seal["complete_bpp"],
            "original_5bit_codes_unchanged":True,
            "FP16_centroid_cells_actually_changed":
                seal["actually_changed_FP16_codebook_cells"],
            "strictly_improves_source":score>(
                len(base["exact"]),base["prefix_tokens"],-base["repetition_excess"]),
            "accepted":accept,
            "candidate_sha256":seal["sha256"],
        }
        checkpoints.append(status)
        print(json.dumps({"stage":"v228_real_full_TRAIN_gate",**status}),flush=True)
        if accept:
            best_score=score
            best=(packed,seal,step)
    result={
        "completed":True,
        "source_TRAIN_exact":len(base["exact"]),
        "source_TRAIN_prefix_tokens":base["prefix_tokens"],
        "source_TRAIN_adjacent_repeats":base["repetition_excess"],
        "all_211_original_paid_dictionaries_updated":True,
        "checkpoint_outcomes":checkpoints,
        "new_source_accepted":best is not None,
        "QA_DEV_VALID_TEST_used_for_selection":False,
        "no_inference_sidecar_or_new_decoder":True,
        "native_CPU_benchmark_not_yet_performed":True,
        "not_FP32_generalization_or_QA_parity_evidence":True,
        "Jessica_tests_ran":False,
    }
    if best is not None:
        packed,seal,step=best
        (out/"CALIBRI_v228_TRAIN_ACCEPTED_fullanswer_paid_211.cb59").write_bytes(packed)
        result["accepted_checkpoint"]=step
        result["accepted_sha256"]=seal["sha256"]
        result["accepted_whole_source_bpp"]=seal["complete_bpp"]
    else:
        result["rejected_reason"]="No checkpoint improved complete TRAIN answers with exact-answer preservation"
    save(out/"ACTUAL_v228_48_full_TRAIN_answers_7_checkpoints.json",result)
    save(out/"ACTUAL_v228_paid_211_codebook_updates.json",{
        "step_logs":history,"codebook_count":211,
        "only_original_TRAIN":True})
    print(json.dumps({"stage":"v228_FINAL_TRAIN_full_answer_decision",
       "accepted":result["new_source_accepted"],
       "checkpoints":[{"step":x["checkpoint_step"],
                       "exact":x["full_TRAIN_correct"],
                       "lost":x["lost_original_31_correct_TRAIN"],
                       "accepted":x["accepted"]} for x in checkpoints],
       "source_exact":len(base["exact"]),
       "FP32_QA_parity_not_yet_tested":True}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("v220",type=Path)
    p.add_argument("v222",type=Path)
    p.add_argument("v224",type=Path)
    p.add_argument("v225",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.v220,a.v222,a.v224,a.v225,a.output)
