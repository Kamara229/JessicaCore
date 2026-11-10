"""S110: true S101 frozen 456-TRAIN first-divergence & route attribution.

No new model, no training, NO sealed QA/TEST, and no LIVE changes. Build
the immutable physical S101 and run exact cached S95 routing/S98 copier.
Distinguish wrong first token vs wrong later/EOS; evaluate owner copy
eligibility, first-content-token issues, paired counterfactual both-right
coverage, S93-route addressability, and whether expected wrong token is
even available among literal prompt tokens.

This experiment does not construct oracle target inference nor change the
S95 prompt router. Any numeric TRAIN feature is used only for diagnostics.
All 456 full + EOS greedy responses MUST match frozen 353/387 quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import CHANGED, freeze_report, sha
from calibri2.s95.train_and_audit_router import CHOICES, prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    apply_profile, load_frozen_numeric_weights,
)
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s104.gain_only_mps_transaction import load_cpu_original
from calibri2.s102.v_legacy_quality_diagnostics import sequence_diagnostics
from calibri2.s82.real_mps_generic_token_context_copy import candidate_positions


def first_divergence(gold, actual):
    for i in range(max(len(gold),len(actual))):
        if i >= len(gold) or i >= len(actual) or gold[i] != actual[i]:
            return i
    return -1


def classify(gold, actual, eos):
    """All labels describe TRUE autoregressive token sequence, with EOS."""
    d=first_divergence(gold,actual)
    if d == -1:
        return "full_EOS_correct"
    if d == 0:
        return "first_token_wrong"
    if actual[d] == eos if d < len(actual) else False:
        return "early_EOS_at_position_"+str(d)
    if d >= len(actual):
        return "truncated_before_position_"+str(d)
    if d == len(gold)-1 and gold[-1] == eos:
        return "final_EOS_wrong"
    return "later_token_wrong_position_"+str(d)


def main():
    ap=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S101_full456_report",
        "S87_report","S91_report","S93_report",
    ):
        ap.add_argument(name,type=Path)
    ap.add_argument("out",type=Path)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    p99=json.loads(a.S99_full456_report.read_text())
    p101=json.loads(a.S101_full456_report.read_text())
    if (p101.get("cached_full_EOS_correct")!=353
            or p101.get("new_first_token_correct")!=387
            or p101.get("full_EOS_token_mismatch_count")!=0
            or p101.get("candidate_accepted") is not False
            or p99.get("S99_new_correct_full_EOS")!=353):
        raise AssertionError("Only exact frozen S101 353/387 source allowed")
    for path,digest in (
        (a.S93_CB61,p99["immutable_S95_source_sha256"]),
        (a.S94_bank,p99["S94_weightbank_sha256"]),
        (a.S95_router,p99["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net,p99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes())!=digest:
            raise AssertionError("Frozen physical S101 source SHA mismatch")

    _,_,_,bank,router,copy_net=load_frozen_numeric_weights(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    reports={name:freeze_report(getattr(a,name+"_report"))
             for name in ("S87","S91","S93")}
    questions={}
    pair_groups=defaultdict(dict)
    for family,pairs in s87.families(tok).items():
        for pair in pairs:
            for side in ("left","right"):
                key=f'{family}/{pair["pair_id"]}/{side}'
                question=pair[side]["question"]
                if sha(question.encode())!=reports["S87"][key]["question_sha256"]:
                    raise AssertionError("Immutable TRAIN question changed")
                questions[key]=question
                pair_groups[(family,str(pair["pair_id"]))][side]=key
    ids=sorted(questions)
    if len(ids)!=456 or set(ids)!=set(reports["S87"]):
        raise AssertionError("Original 456 TRAIN identities not recovered")
    if len(pair_groups)!=228 or any(set(v)!={"left","right"} for v in pair_groups.values()):
        raise AssertionError("Unmatched paired TRAIN counterfactual identities")

    features=torch.tensor(np.stack([
        prompt_only_vector(tok,questions[key]) for key in ids
    ]),dtype=torch.float32)
    with torch.inference_mode():
        picked=router(features).argmax(dim=1).tolist()
    routed={key:CHOICES[int(choice)] for key,choice in zip(ids,picked)}
    side=load_file(str(a.S93_sidecar))
    cpu=load_cpu_original(s87.open_envelope(a.S93_CB61.read_bytes()),side)
    model,sources,first,later,pointer,head,tunnel,original_core,original_gt=cpu
    outputs={}
    notes={}
    error_bucket=Counter()
    by_route=defaultdict(Counter)
    by_family=defaultdict(Counter)
    by_route_first_div=defaultdict(Counter)
    eligible_copy=Counter()
    failure_second_reachable=Counter()
    first_correct=0
    full_correct=0
    short_gold=Counter()
    short_response=Counter()
    old_first_gold_whitespace=Counter()
    try:
        selected=None
        for key in ids:
            profile=routed[key]
            if profile!=selected:
                apply_profile(sources,tunnel,profile,bank,original_core,original_gt)
                selected=profile
            question=questions[key]
            gold=reports["S87"][key]["target_ids"]
            tokens,note=cached_full_cpu_with_neural_copy(
                model,sources,first,later,head,tok,question,pointer,copy_net,
            )
            outputs[key]=tokens
            notes[key]=note
            d=first_divergence(gold,tokens)
            result=classify(gold,tokens,tok.eos_token_id)
            family=key.split("/",1)[0]
            correct_full=tokens==gold
            correct_first=bool(tokens) and tokens[0]==gold[0]
            full_correct+=int(correct_full)
            first_correct+=int(correct_first)
            short_gold[len(gold)]+=1
            short_response[len(tokens)]+=1
            error_bucket[result]+=1
            by_route[profile][result]+=1
            by_family[family][result]+=1
            by_route_first_div[profile][str(d)]+=1
            eligible_copy[("eligible",bool(note.get("candidate_present")))]+=1
            eligible_copy[("applied",bool(note.get("neural_override_applied")))]+=1

            # V-series warning: a correct leading whitespace token is NOT
            # necessarily a correctly chosen content word.
            first_expected=tok.decode(gold[:1],skip_special_tokens=False)
            old_first_gold_whitespace[bool(first_expected and not first_expected.strip())]+=1
            if not correct_full and correct_first and d>=1:
                chat=tok.apply_chat_template(
                    [{"role":"user","content":question}],
                    tokenize=False,add_generation_prompt=True,
                )
                prompt_ids=tok.encode(chat,add_special_tokens=False)
                direct_candidates=set(prompt_ids[i] for i in candidate_positions(tok,prompt_ids))
                expected_at_div=gold[d] if d<len(gold) else None
                if expected_at_div is not None:
                    failure_second_reachable[(
                        "wrong_second_gold_available_as_literal_prompt_source"
                        if d==1 else "wrong_later_gold_available_as_literal_prompt_source",
                        bool(expected_at_div in direct_candidates),
                    )]+=1
    finally:
        tunnel.uninstall()
    if full_correct!=353 or first_correct!=387:
        raise AssertionError("S110 actual reconstructed S101 answer/EOS regression")

    pair_stats=Counter()
    for _,pair in pair_groups.items():
        left=pair["left"];right=pair["right"]
        l=outputs[left]==reports["S87"][left]["target_ids"]
        r=outputs[right]==reports["S87"][right]["target_ids"]
        if l and r:pair_stats["both_right"]+=1
        elif l or r:pair_stats["one_right"]+=1
        else:pair_stats["both_wrong"]+=1
    if sum(pair_stats.values())!=228:
        raise AssertionError("S110 pair counts do not cover all 456")
    diag=sequence_diagnostics(
        {k:reports["S87"][k]["target_ids"] for k in ids},
        outputs,outputs,routed,tok=tok,copy_notes={k:{"old":notes[k],"new":notes[k]} for k in ids},
    )
    all_out={
        "stage":"S110_S101_REAL_CACHED_CPU_S95_S98_FIRST_DIVERGENCE_456_TRAIN",
        "source_full_correct":353,
        "source_first_correct":387,
        "incorrect_full":103,
        "first_wrong":456-387,
        "first_correct_but_full_wrong":387-353,
        "error_positions":dict(sorted(error_bucket.items())),
        "routed_error_positions":{k:dict(v) for k,v in by_route.items()},
        "family_error_positions":{k:dict(v) for k,v in by_family.items()},
        "routed_first_divergence":{k:dict(v) for k,v in by_route_first_div.items()},
        "copy_selector_observations":{f"{k[0]}/{k[1]}":v for k,v in eligible_copy.items()},
        "expected_failure_token_in_literal_prompt_candidates":{
            f"{k[0]}/{k[1]}":v for k,v in failure_second_reachable.items()
        },
        "first_gold_token_is_whitespace":dict(old_first_gold_whitespace),
        "expected_full_token_length":dict(short_gold),
        "actual_generated_token_length":dict(short_response),
        "paired_owner_counterfactual_counts":dict(pair_stats),
        "legacy_sequence_diagnostic":diag,
        "source_S101_BPP":float(p101["S99_all_inclusive_physical_bpp"]),
        "all_456_original_TRAIN_sha256":sha("\n".join(ids).encode()),
        "no_further_training_no_new_weights":True,
        "frozen_router_copy_policy":True,
        "old_QA_and_test_not_opened":True,
        "candidate_accepted":False,
        "LIVE_main_Render_Supabase_APK_untouched":True,
    }
    (a.out/"S110_S101_FROZEN_POLICY_BOTTLENECK_456_TRAIN.json").write_text(
        json.dumps(all_out,indent=2,sort_keys=True,ensure_ascii=False)
    )
    print(json.dumps({
        "event":"S110_FROZEN_S101_FIRST_DIVERGENCE_CAUSAL_DIAG",
        "full_correct":full_correct,"first_correct":first_correct,
        "error_buckets":dict(error_bucket),
        "profile_full_correct":{
            k:counts.get("full_EOS_correct",0) for k,counts in by_route.items()
        },
        "paired_both_correct":pair_stats["both_right"],
        "first_gold_whitespace":old_first_gold_whitespace[True],
        "no_model_mutation":True,
    }),flush=True)


if __name__=="__main__":
    main()
