"""S101: CPU KV-cache acceleration of frozen S99 with zero output drift.

ONE physical compressed 36M S93 CB59/CB61 core + paid S94 Green Tunnel
profiles, immutable S95 prompt-only router and S98 small MPS-trained
second-token copier; no new weights, no new training or QA. Freeze S100
once-inspected novel prompts: no S100 questions are read or regenerated.

On the original authorised 456 TRAIN questions, independently generate
both original S99 full-prefix/no-cache replies and new KV-cache replies,
compare token IDs including EOS, and check exact historical 353/456, all
previously protected 287/282 full answers, first tokens and CPU speed.
Original S82 is reconstructed separately for same-run paired speed.
FAIL CLOSED on even one full/EOS discrepancy; keep evidence on failure.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s94.physical_profile_bank import CHANGED, sha, freeze_report
from calibri2.s95.train_and_audit_router import CHOICES, prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    CONFIDENCE_THRESHOLD, full_cpu_with_optional_neural_copy,
    load_frozen_numeric_weights, apply_profile)
from v176.operators import clear

SPEED_SAMPLE=48
MAX_BPP=0.7
SPEED_GATE=0.90


def cached_full_cpu_with_neural_copy(
        model,sources,first,later,head,tok,question,pointer,copy_net):
    """Incremental forward with exact immutable S99 token-selection policy.

    Retain the prompt's hidden states for the S82 generic source pointer,
    which must never see future output. DynamicCache carries only KV
    activations, not another copy of source weights. The identical
    S98 0.95-confidence numeric copier ranks only literal prompt spans.
    """
    prompt=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    context=tok.encode(prompt,add_special_tokens=False)
    first_ids=torch.tensor([context],dtype=torch.long)
    kept=s82.candidate_positions(tok,context) if pointer is not None else []
    source_token_ids=torch.tensor([context[i] for i in kept],dtype=torch.long)
    output=[]
    cache=None
    source_keys=None
    eligible=False
    applied=False
    with torch.inference_mode():
        for step in range(s82.s56.MAX_NEW):
            # A fresh decode step reconstructs only its own 1-token
            # activations; unchanged prompt KV tensors are reused.
            inputs=first_ids if step==0 else torch.tensor(
                [[output[-1]]],dtype=torch.long)
            clear(sources)
            result=model(input_ids=inputs,past_key_values=cache,
                         use_cache=True,output_hidden_states=True)
            cache=result.past_key_values
            if cache is None:
                raise AssertionError("S101 KV cache not produced by actual paid model")
            hidden=result.hidden_states[-1][0]
            if step==0 and pointer is not None and kept:
                source_keys=hidden[torch.tensor(kept,dtype=torch.long)].detach().clone()
            h=hidden[-1]
            logits=result.logits[0,-1]+F.linear(
                first(h) if step==0 else later(h),head)
            if step==1 and pointer is not None and kept:
                if int(logits.argmax())!=tok.eos_token_id:
                    bonus=pointer.full_vocab_bonus(h,source_keys,source_token_ids)
                    logits=logits+torch.zeros_like(logits).index_add(
                        0,source_token_ids,bonus)
            chosen=int(logits.argmax())
            if step==1 and copy_net is not None and output:
                from calibri2.s98.train_owner_anchored_copy import candidates
                pair=candidates(tok,question,output[0])
                if len(pair)==2 and pair[0][0]!=pair[1][0]:
                    eligible=True
                    vectors=torch.as_tensor(
                        np.stack([x[1] for x in pair]),dtype=torch.float32)
                    score=copy_net(vectors)
                    probabilities=torch.softmax(score,dim=0)
                    winner=int(score.argmax())
                    if float(probabilities[winner])>=CONFIDENCE_THRESHOLD:
                        desired=int(pair[winner][0])
                        if desired!=chosen and desired!=tok.eos_token_id:
                            chosen=desired
                            applied=True
            output.append(chosen)
            del result
            clear(sources)
            if chosen==tok.eos_token_id:
                break
    del cache,source_keys
    return output,{"candidate_present":eligible,
                   "neural_override_applied":applied}


def main():
    parser=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S87_report","S91_report","S93_report",
        "S82_original_CB59","S82_original_sidecar"):
        parser.add_argument(name,type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    prior=json.loads(args.S99_full456_report.read_text())
    if (prior.get("stage")!="S99_PHYSICAL_S95_PLUS_S98_COPY_FULL456_TRAIN_GUARD" or
        prior.get("S99_new_correct_full_EOS")!=353 or
        prior.get("S99_S95_previously_correct_287_lost")!=0 or
        prior.get("S99_legacy_union_282_lost_full_count")!=0 or
        prior.get("S99_legacy_union_lost_first_count")!=0 or
        prior.get("candidate_accepted") is not False or
        not prior.get("below_0_7_bpp")):
        raise AssertionError("S101 needs immutable passed S99 physical TRAIN guard")
    p94,p95,p98,bank,router,copy_net=load_frozen_numeric_weights(args)
    for path,expected in (
        (args.S93_CB61,prior["immutable_S95_source_sha256"]),
        (args.S94_bank,prior["S94_weightbank_sha256"]),
        (args.S95_router,prior["S95_neural_profile_selector_sha256"]),
        (args.S98_copy_net,prior["S98_neural_copy_selector_sha256"])):
        if sha(path.read_bytes())!=expected:
            raise AssertionError("S101 cannot change previously measured weights")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    frozen={"S87":freeze_report(args.S87_report),
            "S91":freeze_report(args.S91_report),
            "S93":freeze_report(args.S93_report)}
    questions={}
    for family,pairs in s87.families(tok).items():
        for pair in pairs:
            for side in ("left","right"):
                ident=f'{family}/{pair["pair_id"]}/{side}'
                questions[ident]=pair[side]["question"]
    ids=sorted(questions)
    if len(ids)!=456 or set(ids)!=set(frozen["S87"]):
        raise AssertionError("S101 full TRAIN cohort changed")
    for ident in ids:
        if sha(questions[ident].encode())!=frozen["S87"][ident]["question_sha256"]:
            raise AssertionError("Unapproved frozen TRAIN input altered")
    feature=torch.tensor(np.stack([
        prompt_only_vector(tok,questions[k]) for k in ids]),dtype=torch.float32)
    with torch.inference_mode():
        indices=router(feature).argmax(dim=1).tolist()
    chosen={k:CHOICES[int(i)] for k,i in zip(ids,indices)}
    side=load_file(str(args.S93_sidecar))
    model,sources,first,later,pointer,head,inner=s87.build_cpu(
        s87.open_envelope(args.S93_CB61.read_bytes()),side,True)
    if install_verified_fastpath(inner)!=("3","9"):
        raise AssertionError("S101 frozen zero-gate fastpath missing")
    original_centroids={k:sources[k].centroids.detach().clone() for k in CHANGED}
    original_tunnel={k.removeprefix("internal_tunnel."):v
                     for k,v in side.items() if k.startswith("internal_tunnel.")}
    sample={"S99_no_cache":[],"S101_cache":[],"S82_original":[]}
    mismatches=[]
    lost_legacy_full=[]
    lost_legacy_first=[]
    old_correct=0
    new_correct=0
    first_correct=0
    copied_old=0
    copied_new=0
    selected=None
    eval_rows=0
    # For timing, run a same-run ORIGINAL S82 model independently on the
    # same first 48 frozen TRAIN prompts. The source is never trained.
    baseline_side=load_file(str(args.S82_original_sidecar))
    if any(k not in side or not torch.equal(v,side[k])
           for k,v in baseline_side.items()):
        raise AssertionError("S82 frozen controller provenance mismatch")
    old_model,old_sources,ofirst,olater,opointer,ohead,_=s87.build_cpu(
        args.S82_original_CB59.read_bytes(),baseline_side,False)
    try:
        for index,ident in enumerate(ids):
            profile=chosen[ident]
            if profile!=selected:
                apply_profile(sources,inner,profile,bank,original_centroids,
                              original_tunnel)
                selected=profile
            question=questions[ident]
            goal=frozen["S87"][ident]["target_ids"]
            if index<SPEED_SAMPLE:
                start=time.perf_counter()
                s82.full_cpu_generate(old_model,old_sources,ofirst,olater,
                                      ohead,tok,question,opointer)
                sample["S82_original"].append(time.perf_counter()-start)
            start=time.perf_counter()
            baseline,basenote=full_cpu_with_optional_neural_copy(
                model,sources,first,later,head,tok,question,pointer,copy_net)
            spent_normal=time.perf_counter()-start
            start=time.perf_counter()
            cached,newnote=cached_full_cpu_with_neural_copy(
                model,sources,first,later,head,tok,question,pointer,copy_net)
            spent_cached=time.perf_counter()-start
            if index<SPEED_SAMPLE:
                sample["S99_no_cache"].append(spent_normal)
                sample["S101_cache"].append(spent_cached)
            eval_rows+=1
            copied_old+=int(basenote["neural_override_applied"])
            copied_new+=int(newnote["neural_override_applied"])
            old_correct+=int(baseline==goal)
            new_correct+=int(cached==goal)
            first_correct+=int(bool(cached) and cached[0]==goal[0])
            if cached!=baseline:
                mismatches.append({"TRAIN_identity":ident,
                    "baseline_length":len(baseline),
                    "cache_length":len(cached),
                    "first_token_equal":bool(cached) and bool(baseline) and
                                        cached[0]==baseline[0]})
            if (any(frozen[p][ident]["S87_ids"]==goal
                    for p in ("S87","S91")) and cached!=goal):
                lost_legacy_full.append(ident)
            if (any(frozen[p][ident]["S87_ids"]
                    and frozen[p][ident]["S87_ids"][0]==goal[0]
                    for p in ("S87","S91")) and
                (not cached or cached[0]!=goal[0])):
                lost_legacy_first.append(ident)
    finally:
        inner.uninstall()
        del model,old_model
    if eval_rows!=456 or any(len(sample[k])!=SPEED_SAMPLE for k in sample):
        raise AssertionError("S101 physical 456 CPU evaluation incomplete")
    ratio_vs_s82=statistics.median(sample["S82_original"])/statistics.median(
        sample["S101_cache"])
    ratio_vs_s99=statistics.median(sample["S99_no_cache"])/statistics.median(
        sample["S101_cache"])
    physical_bpp=prior["conservative_entire_physical_bpp"]
    # The S101 path merely changes cache activations, no numeric trained
    # weights, hence the same S99 physical weights and bank SHA are used.
    summary={
        "stage":"S101_FROZEN_S99_CPU_KV_CACHE_EXACT_456_TRAIN_ONLY",
        "physical_compressed_full_TRAIN_questions_generated":456,
        "same_frozen_S99_numeric_weights_used":True,
        "S96_and_S100_previous_novel_examples_never_loaded":True,
        "previous_sealed_QA_TEST_never_read":True,
        "no_new_training_or_new_trainable_weights":True,
        "cached_full_EOS_correct":new_correct,
        "original_no_cache_full_EOS_correct":old_correct,
        "new_first_token_correct":first_correct,
        "full_EOS_token_mismatch_count":len(mismatches),
        "full_EOS_token_mismatch_details":mismatches,
        "legacy_old_union_282_full_lost":len(lost_legacy_full),
        "legacy_old_union_first_lost":len(lost_legacy_first),
        "legacy_old_full_lost_ids":lost_legacy_full,
        "legacy_first_lost_ids":lost_legacy_first,
        "baseline_neural_copy_overrides":copied_old,
        "KV_neural_copy_overrides":copied_new,
        "paired_first_48_median_seconds":{k:statistics.median(v)
                                           for k,v in sample.items()},
        "cached_CPU_speed_vs_original_S82":ratio_vs_s82,
        "cached_CPU_speed_vs_no_cache_S99":ratio_vs_s99,
        "original_S82_CPU_speed_at_least_90pct":ratio_vs_s82>=SPEED_GATE,
        "S99_all_inclusive_physical_bpp":physical_bpp,
        "storage_meets_0_7_bpp":physical_bpp<=MAX_BPP,
        "candidate_accepted":False,
        "QA_sealed_pending":True,
        "native_C_MPS_inference_parity_pending":True,
        "LIVE_untouched":True,
    }
    target=args.out/"S101_KV_CACHE_EXACT_S99_456_TRAIN_CPU_GUARD.json"
    target.write_text(json.dumps(summary,indent=2,sort_keys=True))
    print(json.dumps({"event":"S101_KV_CACHE_ACTUAL_CPU_FULL456",
        "strict_full_plus_EOS":new_correct,
        "old_S99_full":old_correct,
        "token_parity_failures":len(mismatches),
        "legacy_lost_full":len(lost_legacy_full),
        "legacy_lost_first":len(lost_legacy_first),
        "S82_cpu_speed_ratio":ratio_vs_s82,
        "no_cache_s99_cpu_speed_ratio":ratio_vs_s99,
        "bpp":physical_bpp,"candidate_accepted":False}),flush=True)
    if (old_correct!=353 or new_correct!=353 or mismatches or
        lost_legacy_full or lost_legacy_first or
        copied_old!=copied_new or
        not summary["original_S82_CPU_speed_at_least_90pct"] or
        not summary["storage_meets_0_7_bpp"]):
        raise SystemExit("REJECTED S101: KV cache changed S99 full/EOS "
                         "or fails historical safety/0.90 CPU/bpp guard")
    # Passing TRAIN cache parity does NOT constitute untouched sealed QA.


if __name__=="__main__":
    main()
