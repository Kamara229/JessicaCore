"""CALIBRI S40: diagnose all 64 new TRAIN noun decisions in physical S38 vs S39.

The S39 index experiment changed 328 paid 5-bit groups and accepted 11/24
proposals but yielded 0/16 exact new TRAIN full answers. This predeclared
diagnostic traces BOTH prompt answer positions: first-color and second-noun
(teacher-forced after the GOLD first color), then compares actual free-greedy
full answers+EOS. Includes every S30 new TRAIN pair, both orientations.
No train, no model selection, no S29 held-out or S26 inspected QA. Both
input models are literal saved 0.690812875 bpp CB59 checkpoints; no FP32
weight checkpoint, no model mutation or sidecar. No ordinary Jessica test,
no LIVE/Render/Supabase/APK changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v186.pairs import build_pairs
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS, SOURCE_BYTES, MODEL_SPEC, PAID_OPERATORS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s37 import paid_cb59_margin_protected_recovery as s37
from v174.run import save

S38_SHA="6b74f67b4d2324e6e0a1d99bdf525b8fdda81f03f008069ef10aedbcd26c8b34"
S39_SHA="92d87d6b2fce6871c98a99257ae752e4a777604a10e1be60bfca058cd61e5bba"
CANDIDATE_NOUN_IDS=(8727,17559,1863,12505,15754,8233,18606)
TOP_K=6


def physical_model(path,sha):
    raw=path.read_bytes()
    if len(raw)!=SOURCE_BYTES or hashlib.sha256(raw).hexdigest()!=sha:
        raise AssertionError("S40 physical source SHA or size drift")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S40 36M architecture changed")
    if 8*len(raw)/SOURCE_PARAMS>.7:
        raise AssertionError("S40 unphysical budget")
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    if len(sources)!=PAID_OPERATORS:
        raise AssertionError("S40 missing previously paid 8D codebooks")
    return model,sources


@torch.inference_mode()
def inspect_row(model,sources,tok,row):
    clear(sources)
    if len(row["targets"])!=3 or len(row["positions"])!=3:
        raise AssertionError("TRAIN pair no longer color+noun+EOS")
    logits=model(input_ids=row["ids"][None],use_cache=False).logits[0][row["positions"]]
    result=[]
    for pos,role in ((0,"first_color"),(1,"second_noun_after_gold_color"),(2,"EOS_after_gold_color_noun")):
        probs=logits[pos]
        expected=int(row["targets"][pos])
        winning=int(probs.argmax())
        best=probs.topk(TOP_K)
        rank=1+int((probs>probs[expected]).sum())
        arr=[{"id":int(t),"text":tok.decode([int(t)],skip_special_tokens=False),
              "logit":float(v)} for v,t in zip(best.values,best.indices)]
        result.append({
            "role":role,"gold_id":expected,
            "gold":tok.decode([expected],skip_special_tokens=False),
            "chosen_id":winning,
            "chosen":tok.decode([winning],skip_special_tokens=False),
            "rank_gold":rank,
            "gold_logit":float(probs[expected]),
            "winner_logit":float(probs[winning]),
            "gold_minus_winner_logit":float(probs[expected]-probs[winning]),
            "gold_wins":winning==expected,
            "top6":arr,
            "candidate_nouns":{str(i):float(probs[i]) for i in CANDIDATE_NOUN_IDS}
                     if pos==1 else None,
        })
    return result


def all_train_rows(model,sources,tok,new,old,stage):
    rows=[]
    for pair in new:
        for side in ("left","right"):
            row=pair[side]
            positions=inspect_row(model,sources,tok,row)
            actual=s37.score_free(model,sources,tok,row)
            rows.append({"id":f'{pair["pair_id"]}_{side}',
                         "TRAIN_pair_id":pair["pair_id"],
                         "side":side,
                         "gold":row["answer"],
                         "free_actual":actual,
                         "teacher_forced_positions":positions})
    first=sum(x["teacher_forced_positions"][0]["gold_wins"] for x in rows)
    noun=sum(x["teacher_forced_positions"][1]["gold_wins"] for x in rows)
    noun_given_color = sum(x["teacher_forced_positions"][1]["rank_gold"]<=3 for x in rows)
    eos=sum(x["teacher_forced_positions"][2]["gold_wins"] for x in rows)
    fully=sum(x["free_actual"]["full_correct"] for x in rows)
    measured=s37.small_train_probe(model,sources,tok,new,old)
    historic=measured["historical_TRAIN"]["fully_correct"]
    fresh=measured["new_TRAIN"]["fully_correct"]
    if historic!=7 or fresh!=0:
        raise AssertionError("Unexpected changed historical/new TRAIN baseline")
    return {
        "stage":stage,"all_64_new_TRAIN_full_pairs":rows,
        "first_color_teacher_forced_top1":first,
        "gold_second_noun_teacher_forced_top1":noun,
        "gold_second_noun_teacher_forced_top3":noun_given_color,
        "EOS_teacher_forced_top1":eos,
        "new_full_greedy_TRAIN_all_64":fully,
        "old_historical_full_TRAIN_first_8":historic,
        "new_probe_16_full_TRAIN":fresh,
        "full_probe":measured,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("s38",type=Path)
    ap.add_argument("s39",type=Path)
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    assert len(new)==32 and len(old)==24
    source38,src38=physical_model(args.s38,S38_SHA)
    source39,src39=physical_model(args.s39,S39_SHA)
    compared={}
    for label,m,sources in (("S38",source38,src38),("S39",source39,src39)):
        compared[label]=all_train_rows(m,sources,tok,new,old,label)
        q=compared[label]
        print(json.dumps({
          "event":"S40_PHYSICAL_TEACHER_FORCED_NOUN_TRAIN",
          "model":label,"color_top1":q["first_color_teacher_forced_top1"],
          "noun_top1":q["gold_second_noun_teacher_forced_top1"],
          "noun_top3":q["gold_second_noun_teacher_forced_top3"],
          "eos_top1":q["EOS_teacher_forced_top1"],
          "free_full_64":q["new_full_greedy_TRAIN_all_64"],
          "protected_old_8":q["old_historical_full_TRAIN_first_8"],
          "physical_bpp":8*SOURCE_BYTES/SOURCE_PARAMS
        },ensure_ascii=False),flush=True)
    a=compared["S38"]["all_64_new_TRAIN_full_pairs"]
    b=compared["S39"]["all_64_new_TRAIN_full_pairs"]
    if [x["id"] for x in a]!=[x["id"] for x in b]:
        raise AssertionError("S40 paired TRAIN case alignment broken")
    deltas=[]
    for first,second in zip(a,b):
        fr=first["teacher_forced_positions"][1]
        to=second["teacher_forced_positions"][1]
        deltas.append({
            "id":first["id"],
            "noun_gold_rank_S38":fr["rank_gold"],
            "noun_gold_rank_S39":to["rank_gold"],
            "noun_gold_logit_minus_winner_delta":
               to["gold_minus_winner_logit"]-fr["gold_minus_winner_logit"],
            "noun_gold_rank_improved":to["rank_gold"]<fr["rank_gold"],
            "noun_gold_rank_worsened":to["rank_gold"]>fr["rank_gold"],
            "new_free_full_answer_gained":
               (not first["free_actual"]["full_correct"]) and second["free_actual"]["full_correct"],
        })
    report={
      "stage":"S40_S39_NOUN_COLLAPSE_DIAGNOSTIC_REAL_PAID_36M_CB59",
      "physical_source_SHA256":{"S38":S38_SHA,"S39":S39_SHA},
      "physical_bytes":SOURCE_BYTES,
      "physical_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
      "real_36M_unique_parameters":SOURCE_PARAMS,
      "S29_QA_and_S26_inspected_QA_NOT_ACCESSED":True,
      "ONLY_frozen_TRAIN_pairs_inspected":True,
      "no_optimization_no_model_selection":True,
      "full_greedy_and_gold_teacher_forced_both_measured":True,
      "unseen_quality_and_FP32_parity_NOT_established":True,
      "only_prepaid_CB59_weights_loaded":True,
      "Jessica_chat_test_NOT_RUN":True,
      "LIVE_untouched":True,
      "paired_results":compared,
      "per_pair_noun_rank_changes":deltas,
      "summary":{
        "gold_noun_rank_improved":sum(x["noun_gold_rank_improved"] for x in deltas),
        "gold_noun_rank_worsened":sum(x["noun_gold_rank_worsened"] for x in deltas),
        "S38_teacher_forced_noun_top1":compared["S38"]["gold_second_noun_teacher_forced_top1"],
        "S39_teacher_forced_noun_top1":compared["S39"]["gold_second_noun_teacher_forced_top1"],
        "S38_free_full_answers":compared["S38"]["new_full_greedy_TRAIN_all_64"],
        "S39_free_full_answers":compared["S39"]["new_full_greedy_TRAIN_all_64"],
      },
    }
    save(args.out/"S40_REAL_CB59_S38_VS_S39_NOUN_LOGITS_64_TRAIN_ONLY.json",report)
    print(json.dumps({"event":"S40_FINAL_PAIRED_NOUN_RANKS",**report["summary"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
