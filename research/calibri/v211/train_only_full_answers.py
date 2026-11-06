"""v211: READ-ONLY native original 135M all-24-pair TRAIN full-answer analysis.

No gradient steps, optimizer, source mutation, VALID/DEV scoring or model
selection. Compare frozen v206 against physically sealed v208 on all 48
contradictory TRAIN answers, including their EOS and repetition behavior.
The purpose is to diagnose Green Tunnel's actual TRAIN generalization,
NOT to claim public/blind QA improvement or choose DEV checkpoints.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v196 import protocol as p
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v200.rollout import probe_greedy_train, summarize_sequences
from v198.guard import probe_first_tokens, summarize_guard

SEALED_v208_SHA256 = "5a6d5f07f9e49fa6063dd772d7e2aee53beeb3527d10d5a429715593b68939de"
TRAIN_ONLY_PAIRS = 24
TRAIN_ONLY_SIDES = 48
TRAIN_MAX_TOKENS = 6


def check_original_source(path, expected_sha):
    data=Path(path).read_bytes()
    if len(data)!=p.SOURCE_BYTES or hashlib.sha256(data).hexdigest()!=expected_sha:
        raise AssertionError("The signed paid original 135M source has drifted")
    return CB59(data)


def compare(before, after, first_before, first_after):
    if set(before)!=set(after) or len(before)!=TRAIN_ONLY_PAIRS:
        raise AssertionError("Exactly 24 immutable TRAIN pair IDs must match")
    if set(first_before)!=set(first_after) or set(first_before)!=set(before):
        raise AssertionError("TRAIN first-token probes missing or inconsistent")
    base=summarize_sequences(before)
    current=summarize_sequences(after)
    frozen0=summarize_guard(first_before)
    frozen1=summarize_guard(first_after)
    transitions=Counter()
    domains=defaultdict(lambda:Counter())
    per_answer=[]
    gained=[];lost=[]
    for pid in sorted(before):
        b,a=before[pid],after[pid]
        if b["domain"]!=a["domain"]:
            raise AssertionError("Frozen TRAIN domain changed")
        if set(b["sides"])!=set(a["sides"]) or set(b["sides"])!={"left","right"}:
            raise AssertionError("Missing opposite frozen TRAIN answer")
        for side in ("left","right"):
            x,y=b["sides"][side],a["sides"][side]
            if x["gold_tokens_with_EOS"]!=y["gold_tokens_with_EOS"]:
                raise AssertionError("Frozen complete TRAIN answer/EOS changed")
            # These input probes are TRAIN-only; never substitute held-out answers.
            u,v=first_before[pid]["sides"][side],first_after[pid]["sides"][side]
            if u["expected"]!=v["expected"] or u["expected"]!=x["gold_tokens_with_EOS"][0]:
                raise AssertionError("Frozen TRAIN first answer target changed")
            old_ok=bool(x["exact_with_EOS"]);new_ok=bool(y["exact_with_EOS"])
            bucket=("exact" if old_ok else "nonexact")+" -> "+("exact" if new_ok else "nonexact")
            transitions[bucket]+=1
            domains[b["domain"]][bucket]+=1
            key=(pid,side)
            if old_ok and not new_ok:lost.append(key)
            if not old_ok and new_ok:gained.append(key)
            per_answer.append({"TRAIN_pair_id":pid,"side":side,"TRAIN_domain":b["domain"],
                "gold_with_EOS":x["gold_tokens_with_EOS"],
                "v206_generated":x["generated_tokens"],
                "v208_generated":y["generated_tokens"],
                "v206_exact":old_ok,"v208_exact":new_ok,
                "v206_correct_prefix":x["correct_prefix_tokens"],
                "v208_correct_prefix":y["correct_prefix_tokens"],
                "v206_repetition_excess":x["adjacent_repetition_excess"],
                "v208_repetition_excess":y["adjacent_repetition_excess"]})
    if len(per_answer)!=TRAIN_ONLY_SIDES or sum(transitions.values())!=TRAIN_ONLY_SIDES:
        raise AssertionError("Incomplete all-24 pair TRAIN answer audit")
    if current["prefix_tokens"] < 0 or base["prefix_tokens"]<0:
        raise AssertionError("Invalid token prefix totals")
    return {"experiment":"v211_read_only_frozen_all24_FULL_TRAIN_after_sealed_v208",
            "NO_DEV_VALID_TEST_or_Jessica_scored_or_used":True,
            "NO_model_fitting_no_checkpoint_selection":True,
            "original_135M_24_opposing_TRAIN_pair_count":TRAIN_ONLY_PAIRS,
            "frozen_FULL_TRAIN_answer_count":TRAIN_ONLY_SIDES,
            "v206_full_TRAIN_exact":len(base["exact"]),
            "v208_full_TRAIN_exact":len(current["exact"]),
            "full_answer_exact_transitions":dict(sorted(transitions.items())),
            "net_exact_answers_gained":len(gained)-len(lost),
            "gained_exact_TRAIN_answers":gained,"lost_exact_TRAIN_answers":lost,
            "v206_full_TRAIN_correct_prefix_tokens":base["prefix_tokens"],
            "v208_full_TRAIN_correct_prefix_tokens":current["prefix_tokens"],
            "v206_adjacent_generated_repetitions":base["repetition_excess"],
            "v208_adjacent_generated_repetitions":current["repetition_excess"],
            "v206_all24_first_token_correct_count":len(frozen0["correct_prompt_ids"]),
            "v208_all24_first_token_correct_count":len(frozen1["correct_prompt_ids"]),
            "v206_first_token_unique_count":len(frozen0["unique_first_token_ids"]),
            "v208_first_token_unique_count":len(frozen1["unique_first_token_ids"]),
            "by_domain":{k:dict(sorted(v.items())) for k,v in sorted(domains.items())},
            "actual_unmodified_TRAIN_greedy_tokens_and_gold":per_answer,
            "NOT_a_certification_of_FP32_parity":True,
            "NO_LIVE_main_Render_Supabase_APK":True}


def probe_archive(archive, pairs):
    net,sources=build_trainable_from_paid_source(archive)
    net.eval()
    # This is read-only even though the surrogate supports gradients;
    # no optimizer or backward call and all inference uses no_grad.
    with torch.no_grad():
        first=probe_first_tokens(net,sources,pairs)
        full=probe_greedy_train(net,sources,pairs,max_new_tokens=TRAIN_MAX_TOKENS)
    return first,full


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("v206_source")
    parser.add_argument("v208_source")
    parser.add_argument("output")
    args=parser.parse_args()
    torch.manual_seed(200)
    torch.set_num_threads(2)
    start=check_original_source(args.v206_source,SOURCE_V206_SHA256)
    sealed=check_original_source(args.v208_source,SEALED_v208_SHA256)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    unused_train,train_pairs,unused_valid,origin=training_inputs(tokenizer)
    if len(train_pairs)!=TRAIN_ONLY_PAIRS or any(row["split"]!="TRAIN" for row in train_pairs):
        raise AssertionError("Missing original 24 opposite TRAIN pairs")
    before_first,before_full=probe_archive(start,train_pairs)
    del start
    after_first,after_full=probe_archive(sealed,train_pairs)
    del sealed
    result=compare(before_full,after_full,before_first,after_first)
    result["starting_v206_sha256"]=SOURCE_V206_SHA256
    result["sealed_v208_sha256"]=SEALED_v208_SHA256
    result["TRAIN_origin_manifest"]=origin
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    (out/"ALL24_READ_ONLY_frozen_TRAIN_full_answers_v206_vs_v208.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"stage":"v211_all24_frozen_FULL_TRAIN_completed",
        "v206_exact":result["v206_full_TRAIN_exact"],
        "v208_exact":result["v208_full_TRAIN_exact"],
        "v206_correct_prefix":result["v206_full_TRAIN_correct_prefix_tokens"],
        "v208_correct_prefix":result["v208_full_TRAIN_correct_prefix_tokens"],
        "v206_repeat":result["v206_adjacent_generated_repetitions"],
        "v208_repeat":result["v208_adjacent_generated_repetitions"],
        "transitions":result["full_answer_exact_transitions"],
        "neither_valid_nor_DEV_used":True},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
