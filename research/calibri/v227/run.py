"""CALIBRI v227: train real paid weights while protecting correct TRAIN answers.

Precommitted follow-up to v226: removing all five adjacent repetitions
regressed complete opposing TRAIN answers (31 -> 29), and four of the
five repeats were legitimate gold token repetitions. Here we train all
original 211 paid 32x8 FP16 dictionaries with a direct full-answer
preservation signal, a balanced opposing answer-pair signal, and the
ONE verified non-gold repeat. Acceptance is guarded by actual greedy
TRAIN complete answers, not by proxy losses. No DEV/VALID/QA/TEST use,
no runtime postprocessing and no LIVE changes.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from v174.run import save
from v176.operators import clear
from v185.codec import replace_discrete
from v186.fit import paired_loss
from v196 import protocol as P
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v200.rollout import (probe_greedy_train, summarize_sequences,
                          sequence_no_regression_gate)
from v226.run import (authenticate, build_frozen_pairs,
                       five_repeat_cases, MARGIN, GRAD_CLIP)
from v221.run import V220_SOURCE_SHA256

SEED=227
LR=2e-5
STEPS=8
CHECKPOINT_STEPS=(1,2,4,8)
LOSS_WEIGHTS={"opposing_pair":0.35, "non_gold_repeat":0.30,
              "previously_correct_full_answer":0.35}
ANCHOR=0.02


def correct_train_samples(rollout,pairs):
    rows=rollout["all_48_opposing_TRAIN_rows"]
    exact=[(r["pair_id"],r["side"]) for r in rows
           if r["v220_exact_answer"]]
    if len(exact)!=31 or len(set(exact))!=31:
        raise AssertionError("Original frozen 31 v220 correct TRAIN answers drift")
    selected=[]
    for pid,side in sorted(exact):
        sample=pairs[pid][side]
        if pairs[pid]["split"]!="TRAIN":
            raise AssertionError("Correct answer source was not TRAIN")
        selected.append({"pair_id":pid,"side":side,
                         "prompt":sample["prompt_ids"].detach().long().cpu().clone(),
                         "target":sample["targets"].detach().long().cpu().clone()})
    return selected


def already_correct_full_answer_loss(student,case):
    prompt,gold=case["prompt"],case["target"]
    if len(gold)<1:
        raise AssertionError("Missing full TRAIN answer")
    context=torch.cat((prompt,gold[:-1]),dim=0)[None]
    start=len(prompt)-1
    logits=student(input_ids=context,use_cache=False
                   ).logits[0,start:start+len(gold)]
    if len(logits)!=len(gold):
        raise AssertionError("Whole TRAIN answer preservation truncated")
    return F.cross_entropy(logits.float(),gold)


def candidate_score(summary):
    return (len(summary["exact"]),
            -summary["repetition_excess"],
            summary["prefix_tokens"])


def run(v220_path,v222_path,v224_path,v225_path,out):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    archive,rollout,audit,cause=authenticate(
        v220_path,v222_path,v224_path,v225_path)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    selected,pairs,valid,origin=training_inputs(tok)
    if len(selected)!=20 or len(pairs)!=24 or len(valid)!=P.VALID_PAIRS:
        raise AssertionError("Original TRAIN pair source drift")
    if any(pair["split"]!="TRAIN" for pair in pairs):
        raise AssertionError("Only 24 TRAIN opposing pairs permitted")
    pair_map={pair["pair_id"]:pair for pair in pairs}
    baseline=build_frozen_pairs(rollout)
    old=summarize_sequences(baseline)
    if len(old["exact"])!=31 or old["repetition_excess"]!=5:
        raise AssertionError("Original v220 frozen full answer baseline drift")
    bad=five_repeat_cases(audit,cause,pair_map)
    if len(bad)!=1:
        raise AssertionError("Exactly one pathological non-gold repeat expected")
    correct=correct_train_samples(rollout,pair_map)
    student,sources=build_trainable_from_paid_source(archive)
    student.cpu().eval()
    if len(sources)!=211:
        raise AssertionError("Missing paid codebooks")
    original_centroids={name:x.centroids.detach().clone()
                        for name,x in sources.items()}
    original_codes={name:x.codes.detach().clone()
                    for name,x in sources.items()}
    save(out/"PRECOMMITTED_v227_TRAIN_only_correct_answer_guard.json",{
        "source_v220_sha":V220_SOURCE_SHA256,
        "baseline_TRAIN_exact":len(old["exact"]),
        "baseline_TRAIN_repetitions":old["repetition_excess"],
        "TRAIN_correct_full_sequences_preserved":len(correct),
        "TRAIN_non_gold_repeats_to_fix":len(bad),
        "gold_token_equal_repeat_not_penalized":4,
        "total_TRAIN_opposing_pairs":len(pairs),
        "all_original_paid_codebooks":len(sources),
        "training_steps":STEPS,
        "checkpoint_steps":CHECKPOINT_STEPS,
        "learning_rate":LR,
        "loss_weights":LOSS_WEIGHTS,
        "centroid_anchor_coefficient":ANCHOR,
        "original_5bit_indices_unchanged":True,
        "physical_bpp_cap":0.7,
        "no_DEV_VALID_TEST_or_QA_for_training_stopping_or_gate":True,
        "real_complete_TRAIN_answers_as_only_acceptance_gate":True,
        "original_Jessica_LIVE_main_Render_Supabase_APK_UNCHANGED":True,
    })
    optimizer=torch.optim.Adam(
        [x.centroids for x in sources.values()],lr=LR)
    progress=[]
    inspections=[]
    best=None
    best_score=candidate_score(old)
    bad_case=bad[0]
    for step in range(1,STEPS+1):
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        pair=pairs[(step-1)%len(pairs)]
        keep=correct[(step-1)%len(correct)]
        paired,paired_stats=paired_loss(student,None,pair)
        logits=student(input_ids=bad_case["input_ids"],use_cache=False
                       ).logits[0,-1].float()
        gold=bad_case["gold_next_token_id"]
        repeated=bad_case["wrong_repeated_token_id"]
        anti=F.cross_entropy(
            logits[None,:],logits.new_tensor([gold],dtype=torch.long))
        anti=anti+0.5*F.relu(MARGIN+logits[repeated]-logits[gold])
        kept=already_correct_full_answer_loss(student,keep)
        anchor=torch.stack([
            (x.centroids-original_centroids[name]).square().mean()
            for name,x in sources.items()
        ]).mean()
        total= (LOSS_WEIGHTS["opposing_pair"]*paired+
                LOSS_WEIGHTS["non_gold_repeat"]*anti+
                LOSS_WEIGHTS["previously_correct_full_answer"]*kept+
                ANCHOR*anchor)
        if not torch.isfinite(total):
            raise AssertionError("Nonfinite v227 TRAIN restoration")
        total.backward()
        active=sum(x.centroids.grad is not None for x in sources.values())
        if active!=211:
            raise AssertionError("v227 not updating all 211 codebooks")
        grad_norm=float(torch.nn.utils.clip_grad_norm_(
            [x.centroids for x in sources.values()],GRAD_CLIP,
            error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for x in sources.values():
                x.centroids.clamp_(-16,16)
        clear(sources)
        progress.append({
            "step":step,"all_211_codebooks_receive_gradient":active,
            "opposing_TRAIN_pair":pair["pair_id"],
            "preserved_correct_TRAIN_id":[keep["pair_id"],keep["side"]],
            "non_gold_TRAIN_repeat_id":bad_case["TRAIN_pair_id"],
            "loss_only_diagnostic":float(total.detach()),
            "gradient_norm":grad_norm,
        })
        print(json.dumps({"stage":"v227_paid_full_answer_guard_step",
            "step":step,"all_211_codebooks":active,
            "preserved_TRAIN":[keep["pair_id"],keep["side"]]}),flush=True)
        if step not in CHECKPOINT_STEPS:
            continue
        # A checkpoint is acceptable ONLY if actual 12-token greedy
        # sequences keep all previously correct answers and prefixes.
        generated=probe_greedy_train(
            student,sources,pairs,max_new_tokens=12)
        measured=summarize_sequences(generated)
        gate=sequence_no_regression_gate(baseline,generated)
        clear(sources)
        packed,seal=replace_discrete(archive,sources,strict=False)
        if (len(packed)!=P.SOURCE_BYTES or seal["complete_bpp"]>0.7
                or seal["actual_changed_5bit_group_indices"]!=0
                or seal["all_2d_original_matrices"]!=211):
            raise AssertionError("Physical codec or 211-source budget drift")
        if any(not torch.equal(sources[n].codes,x)
               for n,x in original_codes.items()):
            raise AssertionError("Original 5-bit indices mutated")
        score=candidate_score(measured)
        acceptable=bool(
            gate["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"] and
            score>candidate_score(old) and
            seal["actually_changed_FP16_codebook_cells"]>0)
        row={
            "checkpoint":step,
            "complete_TRAIN_exact":len(measured["exact"]),
            "complete_TRAIN_prefix_tokens":measured["prefix_tokens"],
            "TRAIN_adjacent_repeats":measured["repetition_excess"],
            "lost_original_correct_TRAIN_answers":
                gate["lost_original_complete_TRAIN_answers"],
            "TRAIN_gate_no_regression":
                gate["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"],
            "strictly_improves_original_TRAIN_baseline":score>candidate_score(old),
            "physically_paid_bpp":seal["complete_bpp"],
            "FP16_codebook_cells_changed":
                seal["actually_changed_FP16_codebook_cells"],
            "acceptable":acceptable,
            "source_sha256":seal["sha256"],
        }
        inspections.append(row)
        print(json.dumps({"stage":"v227_real_48_TRAIN_answer_guard",**row}),
              flush=True)
        if acceptable and score>best_score:
            best_score=score
            best=(packed,seal,step)
    result={
        "completed":True,
        "original_v220_TRAIN_exact":len(old["exact"]),
        "original_v220_TRAIN_repeats":old["repetition_excess"],
        "original_v220_TRAIN_prefix_tokens":old["prefix_tokens"],
        "checkpoints":inspections,
        "trained_all_211_original_paid_dictionaries":True,
        "all_original_5bit_indices_fixed":True,
        "only_precommitted_TRAIN_used_for_acceptance":True,
        "no_public_DEV_VALID_TEST_or_QA_read":True,
        "no_native_CPU_benchmark_or_FP32_parity_claim":True,
        "Jessica_tests_ran":False,
        "accepted_source":best is not None,
    }
    if best is not None:
        packed,seal,step=best
        target=out/"CALIBRI_v227_ACCEPTED_TRAIN_full_answer_guarded.cb59"
        target.write_bytes(packed)
        result["accepted_step"]=step
        result["accepted_sha256"]=seal["sha256"]
        result["accepted_whole_source_bpp"]=seal["complete_bpp"]
    else:
        result["reject_reason"]="no full TRAIN answer no-regression and improvement"
    save(out/"ACTUAL_v227_TRAIN_real_full_answers_4_checkpoints.json",result)
    save(out/"ACTUAL_v227_211_codebook_TRAIN_steps.json",{
        "steps":progress,"trained_all_211_paid_centroids":True})
    print(json.dumps({"stage":"v227_final_accepted_source_or_rejection",
        "accepted":result["accepted_source"],
        "accepted_step":result.get("accepted_step"),
        "original_TRAIN_correct":len(old["exact"]),
        "checkpoints":[{"step":x["checkpoint"],"correct":x["complete_TRAIN_exact"],
                        "repeats":x["TRAIN_adjacent_repeats"],
                        "accepted":x["acceptable"]} for x in inspections],
        "no_QA_or_DEV_selection":True}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("sealed_v220",type=Path)
    ap.add_argument("original_v222_TRAIN_rollouts",type=Path)
    ap.add_argument("original_v224_TRAIN_repeat_audit",type=Path)
    ap.add_argument("original_v225_TRAIN_logits",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    run(args.sealed_v220,args.original_v222_TRAIN_rollouts,
        args.original_v224_TRAIN_repeat_audit,
        args.original_v225_TRAIN_logits,args.output)
