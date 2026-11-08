"""S28: post-hoc causal diagnosis of the *completed* S26 zero/96 unseen QA.

Read-only: compare ONE sealed S24 and ONE final physically paid S26 archive.
For each previously frozen independent S26 color+noun question, condition on
the EXACT gold color (teacher forcing) and inspect gold second/noun token rank
among the entire output vocabulary. Compare to actual greedy generation from
the independently archived S24/S26 QA files.

This measures whether noun token information is present but fails decoding,
or is not learned at all. Gold-prefix forcing is *not* a correct real answer.
Every question already belonged to S26 QA. NEVER use this dataset as TRAIN,
optimizer feedback, checkpoint selection, decoder heuristics or S28 validation.
A subsequent learning experiment needs another independently precommitted,
fresh QA suite. No modifying LIVE or Green Tunnel weights.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC
from calibri2.s26.frozen_new_qa import cases,manifest
from v186.pairs import tokenize_one
from v176.operators import clear
from v174.run import save

S24_SHA="b21619b12ff6014b63129916857a525455960852c8f5a530de233acf10e8f900"
S26_SHA="eaa71093d89cd658a1e9d7a7e5297586f2d4549778d1b9bfae3cf54c26ce502c"
QA_SHA="e8e3e75c5fedf845aef512a104723a4d04fab3980ecec7f4deb3575ae020ac3f"
NOVEL="reserved_novel_color_noun"


def evidence(path,expected_sha):
    obj=json.loads(path.read_text(encoding="utf-8"))
    if (obj.get("physical_archive_SHA256")!=expected_sha or
        obj.get("frozen_suite_sha256")!=QA_SHA or
        obj.get("unique_params")!=SOURCE_PARAMS or
        obj.get("complete_bytes")!=SOURCE_BYTES):
        raise AssertionError("QA/SHA/physical metadata not sealed")
    rows=obj["every_144_actual_answer_and_gold"]
    if len(rows)!=144 or len({r["id"] for r in rows})!=144:
        raise AssertionError("Incomplete sealed 144 QA evidence")
    return {r["id"]:r for r in rows}


def score_archive(path,sha,label,rows,old,after,tok):
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=sha or len(blob)!=SOURCE_BYTES:
        raise AssertionError("Model archive not immutable physically paid 36M")
    cb=CB59(blob)
    if cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Architecture differs")
    model,sources=create_small_paid_model(cb,build_config(MODEL_SPEC))
    model.eval()
    data=[]
    with torch.inference_mode():
        for idx,c in enumerate(rows):
            gold=c["gold"].split()
            if len(gold)!=2:raise AssertionError("Expected frozen color+noun")
            color_ids=tok.encode(gold[0],add_special_tokens=False)
            answer_ids=tok.encode(c["gold"],add_special_tokens=False)
            if len(color_ids)!=1 or len(answer_ids)!=2 or color_ids[0]!=answer_ids[0]:
                raise AssertionError("S26 second lexical token not isolated")
            item=tokenize_one(tok,c["question"],c["gold"])
            if len(item["positions"])!=len(item["targets"]) or len(item["targets"])!=3:
                raise AssertionError("Expected exactly color+noun+EOS")
            if list(map(int,item["targets"]))!=answer_ids+[tok.eos_token_id]:
                raise AssertionError("Answer tokenizer differs from frozen QA")
            # This is teacher forcing, not the model's actual free generation.
            logits=model(input_ids=item["ids"][None],use_cache=False).logits[0][item["positions"]]
            gold_id=answer_ids[1]
            second_logits=logits[1]
            rank=int((second_logits>second_logits[gold_id]).sum())+1
            second_top1=int(second_logits.argmax())
            first_top1=int(logits[0].argmax())
            after_gold_eos_top1=int(logits[-1].argmax())==tok.eos_token_id
            reference=old if label=="S24" else after
            real=reference[c["id"]]
            if (c["question"]!=real["question"] or c["gold"]!=real["gold"] or
                real["expected_token_ids_full_EOS"]!=answer_ids+[tok.eos_token_id]):
                raise AssertionError("S28 diagnostic vs sealed S26 QA mismatched")
            data.append({
                "id":c["id"],"noun":gold[1],"color":gold[0],
                "gold_noun_token_id":gold_id,
                "gold_noun_rank_after_teacher_forced_correct_color":rank,
                "gold_noun_top1_after_forced_color":rank==1,
                "gold_noun_top5_after_forced_color":rank<=5,
                "gold_noun_top20_after_forced_color":rank<=20,
                "gold_noun_top100_after_forced_color":rank<=100,
                "second_top1_token_id":second_top1,
                "second_top1_token":repr(tok.decode([second_top1])),
                "premature_eos_after_forced_color":second_top1==tok.eos_token_id,
                "first_color_top1_without_forcing":first_top1==answer_ids[0],
                "EOS_top1_after_teacher_forced_entire_gold":after_gold_eos_top1,
                "actual_free_reply":real["actual_text"],
                "actual_free_strict_full_EOS":real["strict_full_EOS"],
                "actual_free_first_token_correct":real["correct_first_token"],
            })
            if (idx+1)%24==0:
                print(json.dumps({"event":"S28_READONLY_CAUSAL_NOUN_PROGRESS",
                    "checkpoint":label,"done":idx+1}),flush=True)
    clear(sources)
    del model,sources
    return data


def summarize(rows):
    count=len(rows)
    if count!=96:raise AssertionError("Must retain exactly 96 independent nouns")
    ranks=sorted(r["gold_noun_rank_after_teacher_forced_correct_color"] for r in rows)
    k=Counter(r["second_top1_token"] for r in rows)
    per_noun={}
    for noun in sorted(set(r["noun"] for r in rows)):
        rr=[r for r in rows if r["noun"]==noun]
        per_noun[noun]={
            "n":len(rr),
            "forced_gold_noun_top1":sum(r["gold_noun_top1_after_forced_color"] for r in rr),
            "forced_gold_noun_top5":sum(r["gold_noun_top5_after_forced_color"] for r in rr),
            "forced_gold_noun_top20":sum(r["gold_noun_top20_after_forced_color"] for r in rr),
        }
    if sorted(v["n"] for v in per_noun.values())!=[24]*4:
        raise AssertionError("Wrong independent noun distribution")
    return {
        "n":count,
        "strict_greedy_full_EOS":sum(r["actual_free_strict_full_EOS"] for r in rows),
        "actual_first_color_correct":sum(r["actual_free_first_token_correct"] for r in rows),
        "forced_gold_noun_top1":sum(r["gold_noun_top1_after_forced_color"] for r in rows),
        "forced_gold_noun_top5":sum(r["gold_noun_top5_after_forced_color"] for r in rows),
        "forced_gold_noun_top20":sum(r["gold_noun_top20_after_forced_color"] for r in rows),
        "forced_gold_noun_top100":sum(r["gold_noun_top100_after_forced_color"] for r in rows),
        "forced_gold_noun_median_rank":ranks[count//2],
        "forced_gold_noun_rank_range":[ranks[0],ranks[-1]],
        "premature_EOS_top1_after_forced_color":sum(r["premature_eos_after_forced_color"] for r in rows),
        "forced_full_gold_EOS_top1":sum(r["EOS_top1_after_teacher_forced_entire_gold"] for r in rows),
        "different_second_word_top1_token_count":len(k),
        "top1_second_word_token_modes":k.most_common(12),
        "per_noun":per_noun,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s24-archive",type=Path,required=True)
    parser.add_argument("--s26-archive",type=Path,required=True)
    parser.add_argument("--s24-qa",type=Path,required=True)
    parser.add_argument("--s26-qa",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    torch.set_num_threads(1)
    old=evidence(a.s24_qa,S24_SHA)
    after=evidence(a.s26_qa,S26_SHA)
    if set(old)!=set(after):raise AssertionError("Original QA identity changed")
    if manifest()["qa_suite_sha256"]!=QA_SHA:
        raise AssertionError("Independent QA suite changed after S26")
    rows=[r for r in cases() if r["domain"]==NOVEL]
    if len(rows)!=96 or len({r["id"] for r in rows})!=96:
        raise AssertionError("Unexpected existing sealed S26 novel QA")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s24=score_archive(a.s24_archive,S24_SHA,"S24",rows,old,after,tok)
    s26=score_archive(a.s26_archive,S26_SHA,"S26",rows,old,after,tok)
    by_id={r["id"]:r for r in s24}
    pairs=[]
    for r in s26:
        prev=by_id[r["id"]]
        pairs.append({
            "id":r["id"],"noun":r["noun"],"color":r["color"],
            "S24_gold_noun_forced_rank":prev["gold_noun_rank_after_teacher_forced_correct_color"],
            "S26_gold_noun_forced_rank":r["gold_noun_rank_after_teacher_forced_correct_color"],
            "S26_rank_improved":r["gold_noun_rank_after_teacher_forced_correct_color"]<
                                 prev["gold_noun_rank_after_teacher_forced_correct_color"],
            "S26_rank_worsened":r["gold_noun_rank_after_teacher_forced_correct_color"]>
                                 prev["gold_noun_rank_after_teacher_forced_correct_color"],
        })
    report={
        "stage":"S28_READONLY_CAUSAL_NOUN_S24_VS_S26_POSTHOC",
        "frozen_QA_SHA256":QA_SHA,
        "S24_archive_SHA256":S24_SHA,
        "S26_archive_SHA256":S26_SHA,
        "physically_paid_complete_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
        "same_architecture_and_parameter_count":True,
        "S24":summarize(s24),
        "S26":summarize(s26),
        "S26_forced_gold_noun_rank_improved":sum(p["S26_rank_improved"] for p in pairs),
        "S26_forced_gold_noun_rank_worsened":sum(p["S26_rank_worsened"] for p in pairs),
        "per_question_paired_rank_deltas":pairs,
        "all_actual_forced_context_predictions":{
            "S24":s24,"S26":s26},
        "already_seen_frozen_S26_QA":True,
        "forced_correct_first_token_is_NOT_actual_correct_answer":True,
        "no_QA_for_training_stopping_hyperparameter_or_checkpoint_selection":True,
        "next_training_requires_new_precommitted_QA":True,
        "no_weights_modified":True,
        "no_runtime_inference_adapter":True,
    }
    if report["physically_paid_complete_bpp"]>.7:
        raise AssertionError("Physical rate contract exceeded")
    a.out.mkdir(parents=True,exist_ok=True)
    save(a.out/"ACTUAL_S28_S24_S26_CAUSAL_NOUN_96_POSTHOC.json",report)
    print(json.dumps({
        "stage":report["stage"],
        "S24":report["S24"],
        "S26":report["S26"],
        "S26_rank_improved":report["S26_forced_gold_noun_rank_improved"],
        "S26_rank_worsened":report["S26_forced_gold_noun_rank_worsened"],
        "no_training":True
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
