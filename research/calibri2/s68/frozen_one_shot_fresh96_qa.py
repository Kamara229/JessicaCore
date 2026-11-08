"""CALIBRI S68: one-time FRESH independent QA (96) of locked S67 vs S56.

This file is an evaluation-only sealed instrument created *after* the
S67 physically paid rank128 candidate was frozen and S67 CPU TRAIN gate
passed. It may NEVER be used for gradients, hyperparameter selection,
additional Green Tunnel training, or checkpoint choice. Run once; after
the result, label it inspected and freeze all its prompts and scores.

96 questions = 4 colors x 2 pools (S30 TRAIN item combinations vs S30
WITHHELD color/noun combinations) x 6 genuinely NEW contexts x 2
opposing assignments. These exact new ID, names and format strings
are not included in S30/S54 TRAIN. Unlike TRAIN improvements, these
are evidence about a specific fresh task distribution, not general
language capability. Compare S56 to physically paid S67 only.
No FP32-trained baseline available in this experiment; do not claim
FP32 quality parity. No GPU, no Jessica chat, no LIVE edits.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics,time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s61.cpu_native_greedy_parity import build
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from v186.pairs import tokenize_one

# Freeze challenge before execution; there is no downstream training.
COLORS=("red","green","blue","yellow")
NOUN_UNIVERSE=("bowl","drum","ring","rope")
OWNERS=(
    ("Kavon","Rellis"),("Mordel","Zivra"),("Javin","Helor"),
    ("Novar","Fellis"),("Yemin","Othin"),("Parek","Velsa")
)
FRAMES=(
 "EVAL-ONLY ledger {i}: Mark {a} has code {left}; mark {b} has code {right}. Copy the exact two-word code for {a}.",
 "EVAL-ONLY index {i}: {b} maps to {right}; {a} maps to {left}. Give the two words mapped to {a}.",
 "EVAL-ONLY desk {i}: At {a}, the assigned label reads {left}. At {b}, it reads {right}. State the label at {a}.",
 "EVAL-ONLY card {i}: Entry {b} holds {right}. Entry {a} holds {left}. Return {a}'s color and object.",
 "EVAL-ONLY parcel {i}: {a} is stamped {left}, whereas {b} is stamped {right}. Transcribe {a}'s stamp.",
 "EVAL-ONLY receipt {i}: The receipt has {b}={right} and {a}={left}. Output the two-word value of {a}.",
)
assert len(OWNERS)==len(FRAMES)==6

def sha(raw):return hashlib.sha256(raw).hexdigest()

def challenge_pairs(tok):
    """Create exact independent 48 contrasting pairs / 96 fresh questions."""
    all_pairs=[]
    serial=68000
    if set(s30.TRAIN_ELIGIBLE)!=set(COLORS):
        raise AssertionError("TRAIN color provenance changed")
    for color in COLORS:
        seen=tuple(s30.TRAIN_ELIGIBLE[color])
        withheld=tuple(n for n in NOUN_UNIVERSE if n not in seen)
        if len(seen)!=2 or len(withheld)!=2:raise AssertionError("must preserve exact S30 TRAIN/withheld split")
        for cohort,nouns in (("seen_item_unseen_frame",seen),
                             ("withheld_color_noun_unseen_frame",withheld)):
            x,y=[f"{color} {noun}" for noun in nouns]
            for j,frame in enumerate(FRAMES):
                owner,other=OWNERS[j]
                q0=frame.format(i=serial,a=owner,b=other,left=x,right=y)
                q1=frame.format(i=serial,a=owner,b=other,left=y,right=x)
                if ("TRAIN" in q0 or "TRAIN" in q1 or
                    q0==q1 or x==y or x.split()[0]!=y.split()[0]):
                    raise AssertionError("new EVAL is not a unique opposed pair")
                left=tokenize_one(tok,q0,x)
                right=tokenize_one(tok,q1,y)
                if left["first_expected_id"]!=right["first_expected_id"]:
                    raise AssertionError("opposition must share exact first color token")
                all_pairs.append({"pair_id":serial,"color":color,"cohort":cohort,
                                  "left":left,"right":right})
                serial+=1
    if len(all_pairs)!=48:
        raise AssertionError("fresh EVAL must contain exactly 48 pairs")
    allq=[p[side]["question"] for p in all_pairs for side in ("left","right")]
    if len(set(allq))!=96:
        raise AssertionError("independent EVAL duplicate prompt")
    return all_pairs

def run(state,tok,question):
    model,sources,first,later,head=state
    t=time.perf_counter()
    actual=s56.greedy_first_active(model,sources,first,later,head,tok,question)
    return actual, time.perf_counter()-t

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("S45_original_CB59",type=Path)
    parser.add_argument("S56_immutable_rank12",type=Path)
    parser.add_argument("S67_frozen_rank128",type=Path)
    parser.add_argument("S67_all_inclusive_manifest",type=Path)
    parser.add_argument("S61_unchanged_CB61_source",type=Path)
    parser.add_argument("output_dir",type=Path)
    a=parser.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(680068)
    original=a.S45_original_CB59.read_bytes()
    baseline=a.S56_immutable_rank12.read_bytes()
    candidate=a.S67_frozen_rank128.read_bytes()
    envelope=a.S61_unchanged_CB61_source.read_bytes()
    manifest=json.loads(a.S67_all_inclusive_manifest.read_text())
    if (sha(original)!=s65.S45_SHA or
        sha(baseline)!=s65.S56_SHA or
        sha(candidate)!=manifest["candidate_S67_rank128_controller_sha256"] or
        sha(envelope)!=manifest["physical_CB61_source_sha256"] or
        manifest["all_inclusive_bytes"]!=len(envelope)+len(candidate)+a.S67_all_inclusive_manifest.stat().st_size or
        float(manifest["all_inclusive_bpp"])>.7):
        raise ValueError("S68 disallowed unfrozen or unbudgeted S67 checkpoint")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    pairs=challenge_pairs(tok)
    if len(pairs)!=48:
        raise AssertionError("bad challenge length")
    challenge_data=[{"pair_id":p["pair_id"],"cohort":p["cohort"],"color":p["color"],
      "left":{"prompt":p["left"]["question"],"answer":p["left"]["answer"]},
      "right":{"prompt":p["right"]["question"],"answer":p["right"]["answer"]}}
      for p in pairs]
    challenge_json=json.dumps(challenge_data,sort_keys=True,separators=(",",":")).encode()
    challenge_sha=sha(challenge_json)
    (a.output_dir/"S68_PERMANENTLY_EMBARGOED_EVAL_CHALLENGE.json").write_bytes(challenge_json)
    print(json.dumps({"event":"S68_FROZEN_CHALLENGE_BEFORE_FIRST_INFERENCE",
        "challenge_sha256":challenge_sha,
        "question_count":96,
        "pair_count":48,
        "frozen_candidate_controller_sha256":sha(candidate)}),flush=True)

    s56_controller=load_file(str(a.S56_immutable_rank12))
    s67_controller=load_file(str(a.S67_frozen_rank128))
    states={
        "S56_immutable_reference":build(original,s56_controller,tok),
        "S67_pretrained_only_candidate":build(original,s67_controller,tok),
    }
    results={name:{} for name in states}
    times={name:[] for name in states}
    for index,pair in enumerate(pairs):
        for side in ("left","right"):
            row=pair[side]
            key=f'{pair["pair_id"]}_{side}'
            expected=tok.encode(row["answer"],add_special_tokens=False)+[tok.eos_token_id]
            order=list(states) if (index+(side=="right"))%2==0 else list(reversed(states))
            for name in order:
                actual,elapsed=run(states[name],tok,row["question"])
                results[name][key]={
                    "actual_token_ids":actual,
                    "decoded":tok.decode(actual,skip_special_tokens=True),
                    "expected":row["answer"],
                    "full_answer_plus_EOS_correct":actual==expected,
                    "first_color_correct":bool(actual) and actual[0]==expected[0],
                    "cohort":pair["cohort"],
                    "color":pair["color"],
                }
                times[name].append(elapsed)
    scores={}
    for name in states:
        rows=results[name]
        scores[name]={"n_questions":96,"n_pairs":48}
        for cohort in ("seen_item_unseen_frame",
                       "withheld_color_noun_unseen_frame","all"):
            selection=[p for p in pairs if cohort=="all" or p["cohort"]==cohort]
            keys=[f'{p["pair_id"]}_{side}' for p in selection for side in ("left","right")]
            full=sum(rows[k]["full_answer_plus_EOS_correct"] for k in keys)
            first=sum(rows[k]["first_color_correct"] for k in keys)
            paired=sum(rows[f'{p["pair_id"]}_left']["full_answer_plus_EOS_correct"] and
                       rows[f'{p["pair_id"]}_right']["full_answer_plus_EOS_correct"] for p in selection)
            scores[name][cohort]={"full_correct":full,"out_of":len(keys),
                                  "first_correct":first,
                                  "both_opposing_correct":paired,
                                  "out_of_pairs":len(selection)}
    base=results["S56_immutable_reference"]
    new=results["S67_pretrained_only_candidate"]
    wins=[k for k in base if not base[k]["full_answer_plus_EOS_correct"] and new[k]["full_answer_plus_EOS_correct"]]
    regress=[k for k in base if base[k]["full_answer_plus_EOS_correct"] and not new[k]["full_answer_plus_EOS_correct"]]
    ratio=statistics.median(times["S56_immutable_reference"])/statistics.median(times["S67_pretrained_only_candidate"])
    evaluation={
        "stage":"CALIBRI_S68_FIRST_AND_ONLY_FRESH_96_RESEARCH_QA_AFTER_S67_FREEZE",
        "exact_challenge_SHA256":challenge_sha,
        "locked_S67_rank128_controller_sha256":sha(candidate),
        "S67_full_physical_bpp":manifest["all_inclusive_bpp"],
        "scores":scores,
        "S67_new_correct_vs_S56":wins,
        "S67_lost_correct_vs_S56":regress,
        "CPU_median_time_ratio_S67_vs_S56":ratio,
        "CPU_90percent_of_S56_gate_pass":ratio>=0.9,
        "real_model_generated_full_token_sequences":results,
        "blind_QA_inspected_after_this_one_run":True,
        "this_QA_permanently_embargoed_from_any_future_training":True,
        "fresh_96_is_task_specific_not_general_language_quality":True,
        "this_is_not_trained_FP32_quality_parity":True,
        "GPU_used":False,
        "Jessica_chat_test_run":False,
        "LIVE_modified":False,
    }
    (a.output_dir/"S68_FRESH_96_FINAL_NO_TRAIN_RESULTS.json").write_text(json.dumps(evaluation,indent=2,ensure_ascii=False))
    print(json.dumps({"event":"S68_NEW_QA_ONE_SHOT_RESULTS",
        "candidate_SHA":sha(candidate),"scores":scores,
        "new_correct":len(wins),"lost_correct":len(regress),
        "cpu_speed_ratio":ratio}),flush=True)

if __name__=="__main__":main()
