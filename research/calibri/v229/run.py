"""CALIBRI v229: sealed paid-archive fidelity and first incorrect TRAIN token.

Research-only, no training or checkpoint selection. Reload the physically sealed
v228 step-12 CB59 and the immutable original v220 CB59 independently, generate
actual 48 opposing TRAIN full-answer sequences, and measure each erroneous
answer's FIRST incorrect next-token logit margin on its own greedy prefix.
This separates actual new full answers from incidental repetition differences.
No DEV/VALID/QA/TEST prompts, no Jessica tests, no LIVE modifications.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v174.run import save
from v176.operators import clear
from v196 import protocol as P
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v200.rollout import (probe_greedy_train, summarize_sequences,
                          sequence_no_regression_gate)
from v221.run import V220_SOURCE_SHA256
from v226.run import authenticate, build_frozen_pairs

ACCEPTED_V228_STEP12_SHA256="0db01518534fb61e57a903f1c5155233993664ca34c4e7f9010fd9a1c8901483"
TRAIN_ANSWER_LIMIT=12

def sha_checked(path, expected):
    raw=Path(path).read_bytes()
    got=hashlib.sha256(raw).hexdigest()
    if len(raw)!=P.SOURCE_BYTES or got!=expected:
        raise AssertionError("Incorrect immutable paid archive bytes / source SHA")
    return CB59(raw)

@torch.no_grad()
def first_incorrect_token_probe(student, paid, pairs, generated):
    reports=[]
    try:
        for pair in pairs:
            if pair["split"]!="TRAIN":
                raise AssertionError("Non-TRAIN sample in read-only diagnostic")
            pid=pair["pair_id"]
            for side in ("left","right"):
                sample=pair[side]
                prompt=sample["prompt_ids"].detach().long().cpu()
                gold=[int(v) for v in sample["targets"]]
                seq=generated[pid]["sides"][side]["generated_tokens"]
                if generated[pid]["sides"][side]["exact_with_EOS"]:
                    continue
                k=0
                while k<min(len(seq),len(gold)) and seq[k]==gold[k]:
                    k+=1
                if k>=len(seq) or k>=len(gold):
                    reports.append({"pair_id":pid,"side":side,"first_bad_1based":None,
                      "reason":"bounded_rollout_or_length_mismatch",
                      "gold_len":len(gold),"generated_len":len(seq)})
                    continue
                context=torch.cat((prompt,torch.tensor(seq[:k],dtype=torch.long)),0)
                clear(paid)
                logits=student(input_ids=context[None],use_cache=False).logits[0,-1].float()
                if not torch.isfinite(logits).all():
                    raise AssertionError("Nonfinite first wrong TRAIN logits")
                best=int(logits.argmax())
                actual=int(seq[k])
                target=int(gold[k])
                if best!=actual or best==target:
                    raise AssertionError("Greedy first-error token and next-token logits disagree")
                target_logit=float(logits[target])
                winning_logit=float(logits[best])
                rank=1+int((logits>logits[target]).sum())
                reports.append({
                  "pair_id":pid,"side":side,"first_bad_1based":k+1,
                  "first_bad_after_correct_prefix":k,
                  "expected_gold_id":target,"generated_wrong_id":actual,
                  "gold_logit_minus_wrong_top1":target_logit-winning_logit,
                  "gold_next_token_rank":rank,
                  "gold_len":len(gold),"generated_len":len(seq),
                  "evaluated_on_real_self_generated_greedy_prefix":True,
                })
    finally:
        clear(paid)
    if len(reports)!=48-len(summarize_sequences(generated)["exact"]):
        raise AssertionError("Missing incorrect TRAIN full answer first-token diagnostics")
    return reports

def evaluate(archive,pairs):
    student,paid=build_trainable_from_paid_source(archive)
    student.cpu().eval()
    if len(paid)!=211:
        raise AssertionError("Expected original 211 physically paid codebooks")
    try:
        generated=probe_greedy_train(student,paid,pairs,max_new_tokens=TRAIN_ANSWER_LIMIT)
        reports=first_incorrect_token_probe(student,paid,pairs,generated)
        summary=summarize_sequences(generated)
    finally:
        del student,paid
        gc.collect()
    return generated,reports,summary

def run(v220_path,v222_path,v224_path,v225_path,v228_step12_path,out):
    torch.manual_seed(229)
    torch.set_num_threads(2)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    v220,rollout,audit,cause=authenticate(v220_path,v222_path,v224_path,v225_path)
    v228=sha_checked(v228_step12_path,ACCEPTED_V228_STEP12_SHA256)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    chosen,pairs,valid,origin=training_inputs(tok)
    if len(chosen)!=20 or len(pairs)!=24 or len(valid)!=P.VALID_PAIRS:
        raise AssertionError("Frozen 24 opposing TRAIN pairs changed")
    if any(p["split"]!="TRAIN" for p in pairs):
        raise AssertionError("Non-TRAIN pairs included in diagnostics")
    frozen=build_frozen_pairs(rollout)
    if len(summarize_sequences(frozen)["exact"])!=31:
        raise AssertionError("v220 frozen baseline drifted")
    save(out/"PRECOMMITTED_v229_read_only_sealed_archive_first_error.json",{
        "source_original_v220":V220_SOURCE_SHA256,
        "source_sealed_v228_step12":ACCEPTED_V228_STEP12_SHA256,
        "original_48_TRAIN_prompts_only":True,
        "original_211_physical_paid_dictionaries_only":True,
        "reloaded_packed_archives_probed_not_unpacked_training_checkpoint":True,
        "full_answer_greedy_limit_tokens":TRAIN_ANSWER_LIMIT,
        "first_error_measured_only_on_actually_generated_context":True,
        "no_optimizer_no_training_no_new_model_export":True,
        "no_DEV_VALID_QA_TEST_scoring_or_training":True,
        "no_LIVE_Jessica_main_Render_Supabase_APK_change":True,
    })
    g0,errors0,s0=evaluate(v220,pairs)
    # Validate the baseline's physical roundtrip rather than trusting a cached score.
    for pid,row in frozen.items():
        for side in ("left","right"):
            if g0[pid]["sides"][side]["generated_tokens"] != row["sides"][side]["generated_tokens"]:
                raise AssertionError("Reloaded sealed v220 differs from original frozen TRAIN sequences")
    print(json.dumps({"stage":"v229_sealed_v220_reproduced",
        "exact":len(s0["exact"]),"correct_prefix":s0["prefix_tokens"],
        "repeats":s0["repetition_excess"],"wrong":len(errors0)}),flush=True)
    del v220
    gc.collect()
    g1,errors1,s1=evaluate(v228,pairs)
    gate=sequence_no_regression_gate(g0,g1)
    known_correct={(pid,side) for pid,side in s0["exact"]}
    corrected=sorted(s1["exact"]-known_correct)
    lost=sorted(known_correct-s1["exact"])
    e0={(r["pair_id"],r["side"]):r for r in errors0}
    e1={(r["pair_id"],r["side"]):r for r in errors1}
    changes=[]
    for key in sorted(e0):
        a=e0[key]
        b=e1.get(key)
        changes.append({
            "pair_id":key[0],"side":key[1],
            "original_v220_first_error_1based":a.get("first_bad_1based"),
            "sealed_v228_first_error_1based":None if b is None else b.get("first_bad_1based"),
            "v220_gold_minus_top1":a.get("gold_logit_minus_wrong_top1"),
            "sealed_v228_gold_minus_top1":None if b is None else b.get("gold_logit_minus_wrong_top1"),
            "now_fully_correct":b is None
        })
    severity=sorted(
      (dict(row,source="sealed_v228_step12") for row in errors1
        if row.get("gold_logit_minus_wrong_top1") is not None),
      key=lambda x:abs(x["gold_logit_minus_wrong_top1"]))
    decision={
        "completed":True,"read_only":True,
        "actual_reloaded_original_v220_TRAIN_full_correct":len(s0["exact"]),
        "actual_reloaded_sealed_v228_TRAIN_full_correct":len(s1["exact"]),
        "v220_correct_prefix":s0["prefix_tokens"],
        "v228_correct_prefix":s1["prefix_tokens"],
        "v220_repetitions":s0["repetition_excess"],
        "v228_repetitions":s1["repetition_excess"],
        "newly_correct_complete_TRAIN_answers":corrected,
        "lost_original_correct_complete_TRAIN_answers":lost,
        "no_regression_gate":gate,
        "v228_first_wrong_gold_margin_nearest_top1":severity,
        "original_17_wrong_answer_first_divergence_comparison":changes,
        "actual_sealed_checkpoint_full_answer_improvement":
           len(s1["exact"])>len(s0["exact"]) and not lost,
        "physical_archive_bpp":0.6678394131307638,
        "no_public_DEV_VALID_TEST_QA_used":True,
        "no_Jessica_test":True,
        "not_evidence_of_FP32_or_external_QA_parity":True,
    }
    save(out/"ACTUAL_v229_sealed_v220_v228_48_full_TRAIN_first_error_margins.json",decision)
    print(json.dumps({"stage":"v229_FINAL_sealed_archive_correct_answers",
       "v220":len(s0["exact"]),"v228":len(s1["exact"]),
       "corrected":corrected,"lost":lost,
       "v228_first_error_nearest_gold_margins":[
         {"case":[r["pair_id"],r["side"]],
          "first_bad":r["first_bad_1based"],
          "gold_minus_top1":round(r["gold_logit_minus_wrong_top1"],4)}
         for r in severity[:8]],
       "source_accepted_only_if_correct_answer_gain":False}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("v220",type=Path)
    ap.add_argument("v222",type=Path)
    ap.add_argument("v224",type=Path)
    ap.add_argument("v225",type=Path)
    ap.add_argument("v228_sealed",type=Path)
    ap.add_argument("out",type=Path)
    a=ap.parse_args()
    run(a.v220,a.v222,a.v224,a.v225,a.v228_sealed,a.out)
