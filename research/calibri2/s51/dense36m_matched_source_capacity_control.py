"""CALIBRI S51: distinguish 36M TRAIN capacity from compressed recovery limits.

This is a controlled, isolated *learning-capacity* study, not FP32 parity
or a generalization test. Use the EXACT physically reloaded S45 CB59 source
that produced the accepted S47 controller, not an unrelated pretrained
teacher and not the older S26 checkpoint used by S30/S34.

At fixed weights, test logits parity for the decompressed dense 36M FP32
network and byte-identical greedy 64 new+8 historical TRAIN answers.
Then allow all 36,380,480 FP32 weights to learn the exact same 64 S30
TRAIN color+opposite-noun examples; measure strict full+EOS and BOTH
sides of 32 pairs, preserving a fixed historical TRAIN replay schedule.
Compare these outcomes to the immutable physically paid S47 controller.
This can disprove an absolute 17/64 or first-token 34/64 *architecture*
ceiling if FP32 exceeds it. It cannot prove a structural model ceiling
if FP32 also fails (finite optimizer, data, seed and compute confound).

Do NOT open S29 sealed independent QA or S26 previously inspected QA.
Do NOT modify Jessica chat, LIVE, Render, Supabase, APK, or main.
No original FP32 teacher, no future high-parameter model training.
FP32 dense checkpoints are *not* 0.7 bpp: all 36M dense trainable
weights in RAM cost ~145.5MB and are never promoted as compressed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from v176.operators import clear
from v186.pairs import build_pairs
from v174.run import save

SOURCE_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
CONTROLLER_SHA="02227f62d961b73ed1e7cf89e370d5a4852bf53839ce8fa8f608c44e286cdb43"
SOURCE_S47_CI=42505575247
STEPS=192
PROBE_STEPS=(48,96,144,192)
LR=.00015
SEED=5151
SCHEDULE=("new","new","old","new")
assert STEPS%len(SCHEDULE)==0

def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pure_paid_probe(model,sources,tok,new,old):
    a=s47.probe(model,sources,None,None,tok,new)
    b=s47.probe(model,sources,None,None,tok,old[:4])
    return a,b

def fp32_probe(model,tok,new,old):
    """Independent free-greedy actual tokens; score the FULL response + EOS."""
    out={}
    for kind,pairs in (("new",new),("old",old[:4])):
        rows={}
        for pair in pairs:
            for side in ("left","right"):
                row=pair[side]
                key=f'{pair["pair_id"]}_{side}'
                actual=s30.greedy_actual(model,tok,row["question"],max_new=10)
                expected=tok.encode(row["answer"],add_special_tokens=False)+[tok.eos_token_id]
                rows[key]={"actual":actual,"expected":expected,
                           "first_gold_correct":bool(actual) and actual[0]==expected[0],
                           "full_correct":actual==expected,
                           "text":tok.decode(actual,skip_special_tokens=True)}
        out[kind]=rows
    return out

def correct(rows):
    return {k for k,r in rows.items() if r["full_correct"]}

def summarize(probe,new):
    result={}
    for kind,rows in probe.items():
        result[kind]={"full":len(correct(rows)),
           "correct_first":sum(v["first_gold_correct"] for v in rows.values()),
           "total":len(rows)}
    newrows=probe["new"]
    pairs={pair["pair_id"]:{"left":newrows[f'{pair["pair_id"]}_left'],
                            "right":newrows[f'{pair["pair_id"]}_right']} for pair in new}
    result["paired_noun"]={
       "total":32,
       "both_full_correct":sum(x["left"]["full_correct"] and x["right"]["full_correct"] for x in pairs.values()),
       "one_full_correct":sum(x["left"]["full_correct"]!=x["right"]["full_correct"] for x in pairs.values()),
       "identical_free_greedy_token_sequences":sum(x["left"]["actual"]==x["right"]["actual"] for x in pairs.values()),
       "both_first_colors_correct":sum(x["left"]["first_gold_correct"] and x["right"]["first_gold_correct"] for x in pairs.values())
    }
    result["response_diversity"]={
       "new_unique_actual_texts":len({v["text"] for v in newrows.values()}),
       "most_common":Counter(v["text"] for v in newrows.values()).most_common(5)
    }
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("s45_source",type=Path)
    p.add_argument("s47_controller",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw=a.s45_source.read_bytes()
    if len(raw)!=SOURCE_BYTES or hash_file(a.s45_source)!=SOURCE_SHA:
        raise AssertionError("S51 actual S45 paid source does not match immutable checkpoint")
    if hash_file(a.s47_controller)!=CONTROLLER_SHA or a.s47_controller.stat().st_size!=22068:
        raise AssertionError("S51 actual S47 Green Tunnel does not match paid checkpoint")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    if len(new)!=32 or len(old)!=24:
        raise AssertionError("S51 TRAIN pair composition drift")
    cb=CB59(raw)
    if cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S51 source architecture differs")
    paid,sources=create_small_paid_model(cb,build_config(MODEL_SPEC))
    paid.requires_grad_(False)
    dense=s30.dense_from_paid(cb)
    parity_max=s30.double_check_original_logits(cb,dense,tok)
    baseline_paid_new,baseline_paid_old=pure_paid_probe(paid,sources,tok,new,old)
    fp32_before=fp32_probe(dense,tok,new,old)
    initial_mismatch={
        "new":sorted(k for k,r in baseline_paid_new.items() if r["actual"]!=fp32_before["new"][k]["actual"]),
        "old":sorted(k for k,r in baseline_paid_old.items() if r["actual"]!=fp32_before["old"][k]["actual"])
    }
    if any(initial_mismatch.values()):
        raise AssertionError("Full dense FP32 and native paid differ before any training; incomparable")
    controller=s47.ActiveGreenTunnel(320,8)
    controller.load_state_dict(load_file(str(a.s47_controller)))
    controller.eval()
    with_controller={
        "new":s47.probe(paid,sources,controller,paid.lm_head.source().detach(),tok,new),
        "old":s47.probe(paid,sources,controller,paid.lm_head.source().detach(),tok,old[:4])
    }
    s47_summary=summarize(with_controller,new)
    if (s47_summary["new"]["full"]!=17 or
        s47_summary["new"]["correct_first"]!=34 or
        s47_summary["old"]["full"]!=7 or
        s47_summary["paired_noun"]["both_full_correct"]!=0):
        raise AssertionError("S47 immutable physical controller output differs")
    raw_summary=summarize(fp32_before,new)
    print(json.dumps({"event":"S51_FP32_SOURCE_PARITY", "max_abs_test_logits":parity_max,
       "all64_new_and8_old_full_source_matches":True,
       "S45_before_training":raw_summary,
       "S47_physical_green_tunnel":s47_summary},ensure_ascii=False),flush=True)
    del paid,sources,cb,controller
    optimizer=torch.optim.AdamW(dense.parameters(),lr=LR,weight_decay=0.0)
    tasks=Counter()
    snapshots=[]
    best_no_old_loss={"new_full":-1,"both_pairs":-1,"step":0}
    best_any={"new_full":-1,"both_pairs":-1,"step":0}
    old7=correct(fp32_before["old"])
    if len(old7)!=7:
        raise AssertionError("physical source old historical correct count drift")
    gradients=0
    for step in range(1,STEPS+1):
        kind=SCHEDULE[(step-1)%len(SCHEDULE)]
        ix=tasks[kind]
        tasks[kind]+=1
        pair=(new if kind=="new" else old)[ix%(len(new) if kind=="new" else len(old))]
        optimizer.zero_grad(set_to_none=True)
        loss=s30.train_loss(dense,pair,kind=="new")
        if not torch.isfinite(loss):
            raise AssertionError("FP32 TRAIN loss is not finite")
        loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(dense.parameters(),1.0,error_if_nonfinite=True)
        if float(gn)>0:gradients+=1
        optimizer.step()
        if step not in PROBE_STEPS:
            continue
        dense.eval()
        now=fp32_probe(dense,tok,new,old)
        counts=summarize(now,new)
        old_lost=sorted(old7-correct(now["old"]))
        # Compare to S47 WITHOUT equating optimization budgets, training
        # distributions or untested generalization to model-size ceilings.
        s={"step":step,"new_full":counts["new"]["full"],
           "new_first_color":counts["new"]["correct_first"],
           "new_pair_both_full":counts["paired_noun"]["both_full_correct"],
           "new_pair_identical_text":counts["paired_noun"]["identical_free_greedy_token_sequences"],
           "old_full":counts["old"]["full"],
           "old_protected_original_7_lost":old_lost,
           "original_S45_exact_full_better_than_S47":counts["new"]["full"]>17,
           "all_original_old7_preserved":not old_lost,
           "full_response_diversity":counts["response_diversity"],
           "TRAIN_only_not_generalization":True}
        snapshots.append(s)
        if (s["new_full"],s["new_pair_both_full"])>(best_any["new_full"],best_any["both_pairs"]):
            best_any={"new_full":s["new_full"],"both_pairs":s["new_pair_both_full"],"step":step}
        if not old_lost and (s["new_full"],s["new_pair_both_full"])>(best_no_old_loss["new_full"],best_no_old_loss["both_pairs"]):
            best_no_old_loss={"new_full":s["new_full"],"both_pairs":s["new_pair_both_full"],"step":step}
        print(json.dumps({"event":"S51_DENSE_FP32_36M_TRAIN_CAPACITY_PROBE",**s},ensure_ascii=False),flush=True)
    if gradients!=STEPS:
        raise AssertionError("Actual independent all-weight FP32 gradients were missing")
    final={
       "stage":"S51_SAME_S45_SOURCE_36M_DENSE_FP32_TRAIN_CAPACITY_CONTROL",
       "same_architecture":True,
       "S45_source_SHA256":SOURCE_SHA,
       "source_S47_Green_Tunnel_SHA256":CONTROLLER_SHA,
       "original_unique_parameter_count":SOURCE_PARAMS,
       "source_CB59_all_inclusive_bytes":SOURCE_BYTES,
       "source_CB59_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
       "dense_FP32_parameters_bytes":4*SOURCE_PARAMS,
       "dense_FP32_effective_bpp":32,
       "max_abs_single_prompt_initial_logits_difference":parity_max,
       "full64_and_old8_initial_fp32_vs_compressed_identical":True,
       "S47_source_controller_full_answers_compared_on_exact_same_64_TRAIN":True,
       "S45_decompressed_FP32_before_training":raw_summary,
       "S47_immutable_paid_Green_Tunnel_before_S51":s47_summary,
       "S51_36M_all_FP32_weights_trainable":True,
       "matched_new_64_TRAIN_and_old_historical_TRAIN":True,
       "training_steps":STEPS,
       "training_LR":LR,
       "TRAIN_task_schedule":SCHEDULE,
       "TRAIN_count_by_kind":dict(tasks),
       "all_weights_nonzero_gradient_steps":gradients,
       "probe_steps":PROBE_STEPS,
       "each_dense_FP32_checkpoint_metrics":snapshots,
       "best_any_TRAIN_before_forgetting_gate":best_any,
       "best_TRAIN_retaining_7_original_old_answers":best_no_old_loss,
       "scientific_interpretation":(
           "If 36M full-precision can fit additional TRAIN full answers without losing old7, then fixed architecture 36M alone is not the hard 17/64 ceiling. "
           "If FP32 does not do so in this limited training, this cannot prove capacity saturation. "
           "Actual generalization and comparison against a larger architecture remain untested."
       ),
       "S29_sealed_QA_never_opened":True,
       "S26_inspected_QA_never_opened":True,
       "no_original_pretrained_FP32_reference":True,
       "FP32_checkpoint_not_shipped_as_compressed":True,
       "Jessica_ordinary_chat_tests_NEVER_RUN":True,
       "LIVE_Render_Supabase_APK_unchanged":True
    }
    save(a.output/"S51_36M_FP32_CAPACITY_CONTROL_REAL_TRAIN_ONLY.json",final)
    print(json.dumps({"event":"S51_FP32_CAPACITY_CONTROL_FINAL",
       "source_same_full64_old8":True,
       "dense_training_real":gradients==STEPS,
       "baseline_paid_S47_full":17,
       "best_dense_FP32_new_full":best_any["new_full"],
       "best_dense_FP32_new_both_pairs":best_any["both_pairs"],
       "best_preserving_old7":best_no_old_loss,
       "S29_QA_not_touched":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
