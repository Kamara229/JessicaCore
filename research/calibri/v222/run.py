"""CALIBRI v222: sealed v206 vs v220 TRAIN-only 12-token rollout divergence.

v221 proved both sources select identical first tokens on all 48 opposing
TRAIN prompts, while the previously sealed PUBLIC DEV16 full answers
regressed 3/16 to 2/16. Check where TRAIN-generated later tokens diverge,
WITHOUT consulting QA/VALID/DEV/TEST or fitting a model.

Read-only SHA-authenticated original 135M packed archives; never modify
the LIVE Jessica/backend/Render/Supabase/APK or model sources.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v174.run import save
from v196 import protocol as P
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v200.rollout import probe_greedy_train, summarize_sequences
from v212.run import SOURCE_V206_SHA256
from v221.run import V220_SOURCE_SHA256

MAX_TOKENS=12


def first_different_token(a,b):
    """1-indexed first actual token where generation differs; 0 means same."""
    for i,(x,y) in enumerate(zip(a,b),1):
        if x!=y:return i
    if len(a)!=len(b):
        return min(len(a),len(b))+1
    return 0


def compare_rows(reference,candidate):
    if set(reference)!=set(candidate) or len(reference)!=24:
        raise AssertionError("Opposing TRAIN pair ids missing or changed")
    rows=[]
    for pair_id in sorted(reference):
        old=reference[pair_id]
        new=candidate[pair_id]
        if old["domain"]!=new["domain"]:
            raise AssertionError("TRAIN pair identity changed")
        for side in ("left","right"):
            before=old["sides"][side];after=new["sides"][side]
            if before["gold_tokens_with_EOS"]!=after["gold_tokens_with_EOS"]:
                raise AssertionError("Frozen TRAIN answer changed across sources")
            at=first_different_token(before["generated_tokens"],
                                      after["generated_tokens"])
            row={"split":"TRAIN","pair_id":pair_id,"side":side,
                 "domain":old["domain"],
                 "first_diff_position_1based":at,
                 "v206_generated_token_ids":before["generated_tokens"],
                 "v220_generated_token_ids":after["generated_tokens"],
                 "gold_token_ids_including_EOS":before["gold_tokens_with_EOS"],
                 "v206_gold_prefix":before["correct_prefix_tokens"],
                 "v220_gold_prefix":after["correct_prefix_tokens"],
                 "v206_exact_answer":before["exact_with_EOS"],
                 "v220_exact_answer":after["exact_with_EOS"],
                 "v206_repeat_excess":before["adjacent_repetition_excess"],
                 "v220_repeat_excess":after["adjacent_repetition_excess"]}
            rows.append(row)
    if len(rows)!=48:
        raise AssertionError("Not all 48 opposing TRAIN generated")
    return rows


def run(source_v206,source_v220,out):
    torch.manual_seed(222);torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    selected,train_pairs,valid_pairs,_=training_inputs(tokenizer)
    if len(selected)!=20 or len(train_pairs)!=24 or len(valid_pairs)!=P.VALID_PAIRS:
        raise AssertionError("Frozen TRAIN protocol altered")
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    save(output/"PRECOMMITTED_v222_TRAIN_12_TOKEN_DIAGNOSTIC.json",{
        "frozen_v206_SHA256":SOURCE_V206_SHA256,
        "frozen_v220_SHA256":V220_SOURCE_SHA256,
        "max_new_tokens":MAX_TOKENS,"opposing_TRAIN_prompts":48,
        "cannot_train_or_select_source":True,
        "no_DEV_VALID_TEST_or_previous_QA_used":True,
        "no_Jessica_tests_no_LIVE_main_changes":True})
    results={}
    for name,path,expected in (("v206",source_v206,SOURCE_V206_SHA256),
                               ("v220",source_v220,V220_SOURCE_SHA256)):
        raw=Path(path).read_bytes()
        if len(raw)!=P.SOURCE_BYTES or sha256(raw).hexdigest()!=expected:
            raise AssertionError("Input not the original sealed "+name+" source")
        model,sources=build_trainable_from_paid_source(CB59(raw))
        model.eval()
        if len(sources)!=211:
            raise AssertionError("Missing native 135M original paid matrices")
        results[name]=probe_greedy_train(
            model,sources,train_pairs,max_new_tokens=MAX_TOKENS)
        print(json.dumps({"stage":"v222_TRAIN_12token_probe",
                          "source":name,"pairs":len(results[name])}),flush=True)
        del model,sources
    rows=compare_rows(results["v206"],results["v220"])
    old=summarize_sequences(results["v206"])
    new=summarize_sequences(results["v220"])
    changed=[r for r in rows if r["first_diff_position_1based"]>0]
    first_bad=[r for r in rows if r["first_diff_position_1based"]==1]
    first_change_by_position=dict(sorted(
        Counter(r["first_diff_position_1based"] for r in changed).items()))
    lost_exact=[(r["pair_id"],r["side"]) for r in rows
                if r["v206_exact_answer"] and not r["v220_exact_answer"]]
    gained_exact=[(r["pair_id"],r["side"]) for r in rows
                  if r["v220_exact_answer"] and not r["v206_exact_answer"]]
    summary={
        "v206_native_source_TRAIN_exact":len(old["exact"]),
        "v220_native_source_TRAIN_exact":len(new["exact"]),
        "v206_aggregate_gold_prefix_tokens":old["prefix_tokens"],
        "v220_aggregate_gold_prefix_tokens":new["prefix_tokens"],
        "v206_adjacent_repetition_excess":old["repetition_excess"],
        "v220_adjacent_repetition_excess":new["repetition_excess"],
        "different_generated_sequences":len(changed),
        "first_token_changes":len(first_bad),
        "earliest_divergence_position_histogram":first_change_by_position,
        "v206_exact_TRAIN_answers_lost":lost_exact,
        "v220_exact_TRAIN_answers_gained":gained_exact,
        "same_first_token_for_all_48_as_v221_reported":len(first_bad)==0,
        "diagnostic_ONLY_not_fp32_parity_or_public_DEV_quality":True}
    result={"completed":True,"max_new_tokens":MAX_TOKENS,
            "source_hashes":{"v206":SOURCE_V206_SHA256,
                             "v220":V220_SOURCE_SHA256},
            "summary":summary,"all_48_opposing_TRAIN_rows":rows,
            "never_train_or_update_source":True,
            "public_DEV16_not_reused":True,
            "Jessica_tests_performed":False}
    save(output/"v222_ACTUAL_TRAIN_first_12token_rollout_divergence.json",result)
    print(json.dumps({"stage":"v222_ACTUAL_TRAIN_12token_compare",**summary}),
          flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("source_v206",type=Path)
    a.add_argument("source_v220",type=Path)
    a.add_argument("output",type=Path)
    x=a.parse_args()
    run(x.source_v206,x.source_v220,x.output)
