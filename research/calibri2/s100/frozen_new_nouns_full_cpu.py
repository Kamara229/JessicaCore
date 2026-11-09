"""S100: irrevocably frozen one-shot evaluation of S99 real physical decoding.

No new training or model selection is permitted. S99 was trained only on
old authorised S82 TRAIN. S96's 64 already inspected new examples are NOT
imported, read, reconstructed or evaluated. All S100 nouns are disjoint
from the WHOLE s82.NOUNS list, hence also disjoint from S96's noun subset.

Evaluate 64 newly written, predeclared two-word copying prompts through
EOS, comparing original S82 vs original S95 vs integrated S99 compressed
CPU, in exactly the same physical model. Save strict negative results on
failure. This is new developmental evaluation, not a sealed benchmark.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s94.physical_profile_bank import CHANGED,sha
from calibri2.s95.train_and_audit_router import CHOICES,prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    full_cpu_with_optional_neural_copy,apply_profile,
    load_frozen_numeric_weights)
from v176.operators import clear

# Authored only in S100, following the completed S99 frozen TRAIN audit;
# never touched by earlier model training or the already inspected S96.
CANDIDATE_NOUNS=(
    "badge","basket","beacon","bottle","button","camera","carpet","castle",
    "cereal","chisel","coin","collar","compass","cookie","crayon","crown",
    "curtain","cushion","engine","fabric","feather","flag","flute","fork",
    "fountain","frame","guitar","helmet","jacket","kettle","ladder","lantern",
    "magnet","marble","medal","needle","notebook","orange","parcel","pebble",
    "pillow","planet","plum","pocket","poster","pumpkin","ribbon","rocket",
    "salad","satchel","shelter","shovel","sponge","stamp","sweater","sword",
    "tablet","tiger","tomato","towel","trophy","tunnel","violin","wallet",
    "whistle","yogurt","zipper","anchor","banana","bicycle","blanket",
    "cabinet","calendar","camera","canoe","cherry","comb","drone",
)
COLORS=("red","green","blue","yellow")
FROZEN_S100_FRAMES=(
    "S100 inventory transfer: A labelled drawer controlled by {a} contains {left}. "
    "The separate drawer controlled by {b} contains {right}. "
    "For {a}, return the color and object in exactly two words.",
    "S100 dispatch note: Technician {b} checked the item {right}; technician {a} "
    "checked the different item {left}. Enter only the exact two-word item "
    "recorded for technician {a}.",
)
OWNER_PAIRS=(("Varolin","Pevrad"),("Nolaria","Kestora"))


def freeze_developmental_questions(tok):
    previously_seen=set(s82.NOUNS)
    available=[]
    for noun in dict.fromkeys(CANDIDATE_NOUNS):
        if noun in previously_seen:
            continue
        if all(len(tok.encode(f"{color} {noun}",add_special_tokens=False))==2
               for color in COLORS):
            available.append(noun)
    if len(available)<8:
        raise AssertionError("S100 needs 8 disjoint physically copyable noun types")
    nouns=available[:8]
    examples=[]
    for color in COLORS:
        for pair in range(4):
            left=f"{color} {nouns[pair]}"
            right=f"{color} {nouns[pair+4]}"
            for frame,(owner,other) in zip(FROZEN_S100_FRAMES,OWNER_PAIRS):
                for chosen,contrast in ((left,right),(right,left)):
                    prompt=frame.format(a=owner,b=other,
                                        left=chosen,right=contrast)
                    gold=tok.encode(chosen,add_special_tokens=False)+[
                         tok.eos_token_id]
                    if len(gold)!=3:
                        raise AssertionError("S100 two-word and EOS target changed")
                    examples.append((prompt,gold))
    if len(examples)!=64 or len({p for p,_ in examples})!=64:
        raise AssertionError("S100 corpus must contain 64 unique new prompts")
    if set(nouns)&set(s82.NOUNS):
        raise AssertionError("S100 would reuse historical S82/S96 nouns")
    # Prior 456 TRAIN questions may be compared to screen for overlap, but
    # never import or inspect previously sealed QA or S96 cases.
    prior=set()
    for _,pairs in s87.families(tok).items():
        for entry in pairs:
            for side in ("left","right"):
                prior.add(entry[side]["question"])
    if any(q in prior for q,_ in examples):
        raise AssertionError("S100 accidentally reused an old TRAIN prompt")
    return examples,nouns


def score(actual,gold):
    result={
        "cases":len(actual),
        "strict_full_plus_EOS":sum(a==y for a,y in zip(actual,gold)),
        "correct_first":sum(bool(a) and a[0]==y[0]
                            for a,y in zip(actual,gold)),
        "correct_first_two":sum(len(a)>1 and a[:2]==y[:2]
                                for a,y in zip(actual,gold)),
        "terminated_in_EOS":sum(bool(a) and a[-1]==y[-1]
                                for a,y in zip(actual,gold)),
    }
    result["strict_full_rate"]=result["strict_full_plus_EOS"]/len(actual)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_previous_full456_report","S82_CB59","S82_sidecar"):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    p99=json.loads(a.S99_previous_full456_report.read_text())
    if (p99.get("stage")!="S99_PHYSICAL_S95_PLUS_S98_COPY_FULL456_TRAIN_GUARD" or
        p99.get("S99_new_correct_full_EOS")!=353 or
        p99.get("S99_S95_previously_correct_287_lost")!=0 or
        not p99.get("below_0_7_bpp") or
        p99.get("candidate_accepted") is not False):
        raise AssertionError("S100 requires completed true S99 353/456 TRAIN audit")
    p94,p95,p98,bank,router,copy_net=load_frozen_numeric_weights(a)
    if (sha(a.S98_copy_net.read_bytes())!=p99["S98_neural_copy_selector_sha256"]
        or sha(a.S95_router.read_bytes())!=p99["S95_neural_profile_selector_sha256"]
        or sha(a.S93_CB61.read_bytes())!=p99["immutable_S95_source_sha256"]):
        raise AssertionError("S99 trained weights changed AFTER frozen S99 evaluation")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    examples,nouns=freeze_developmental_questions(tok)
    prompts=[p for p,_ in examples]
    gold=[g for _,g in examples]
    with torch.inference_mode():
        routing=router(torch.tensor(np.stack([
            prompt_only_vector(tok,q) for q in prompts]),
            dtype=torch.float32)).argmax(dim=1).tolist()
    labels=[CHOICES[i] for i in routing]
    side=load_file(str(a.S93_sidecar))
    model,sources,first,later,pointer,head,inner=s87.build_cpu(
        s87.open_envelope(a.S93_CB61.read_bytes()),side,True)
    if install_verified_fastpath(inner)!=("3","9"):
        raise AssertionError("Certified S95 actual CPU fastpath missing")
    original={k:sources[k].centroids.detach().clone() for k in CHANGED}
    tunnel={k.removeprefix("internal_tunnel."):v for k,v in side.items()
             if k.startswith("internal_tunnel.")}
    baseline=[]
    new=[]
    baseline_lat=[]
    new_lat=[]
    eligible=0
    applied=0
    previous=None
    try:
        for profile,question in zip(labels,prompts):
            if profile!=previous:
                apply_profile(sources,inner,profile,bank,original,tunnel)
                previous=profile
            start=time.perf_counter()
            old=s82.full_cpu_generate(model,sources,first,later,
                                      head,tok,question,pointer)
            baseline_lat.append(time.perf_counter()-start)
            start=time.perf_counter()
            changed,info=full_cpu_with_optional_neural_copy(
               model,sources,first,later,head,tok,question,pointer,copy_net)
            new_lat.append(time.perf_counter()-start)
            eligible+=int(info["candidate_present"])
            applied+=int(info["neural_override_applied"])
            baseline.append(old)
            new.append(changed)
    finally:
        inner.uninstall()
        del model
    # A separate S82 baseline carries NO S95/S98 neural modifications.
    s82side=load_file(str(a.S82_sidecar))
    original_side=load_file(str(a.S93_sidecar))
    if any(k not in original_side or not torch.equal(v,original_side[k])
           for k,v in s82side.items()):
        raise AssertionError("S82 controller provenance mismatch")
    model,sources,first,later,pointer,head,_=s87.build_cpu(
       a.S82_CB59.read_bytes(),s82side,False)
    older=[]
    older_lat=[]
    for question in prompts:
        start=time.perf_counter()
        older.append(s82.full_cpu_generate(model,sources,first,later,
                      head,tok,question,pointer))
        older_lat.append(time.perf_counter()-start)
    del model
    scores={name:score(data,gold) for name,data in [
        ("S82",older),("S95",baseline),("S99",new)]}
    effects={
       "S95_to_S99_gained_full":sum(o!=g and n==g for o,n,g in
                               zip(baseline,new,gold)),
       "S95_to_S99_lost_full":sum(o==g and n!=g for o,n,g in
                               zip(baseline,new,gold)),
       "S82_to_S99_gained_full":sum(o!=g and n==g for o,n,g in
                               zip(older,new,gold)),
       "S82_to_S99_lost_full":sum(o==g and n!=g for o,n,g in
                               zip(older,new,gold))}
    ratio_s95=statistics.median(baseline_lat)/statistics.median(new_lat)
    ratio_s82=statistics.median(older_lat)/statistics.median(new_lat)
    report={
       "stage":"S100_ONE_TIME_NEW_NOUN_AND_TEMPLATE_PHYSICAL_S99_CPU",
       "status":"completed","developmental_independent_not_sealed_QA":True,
       "no_training_or_posthoc_router_tuning":True,
       "previously_used_S96_nouns_or_questions_never_read":True,
       "whole_original_S82_noun_pool_excluded":True,
       "new_nouns_count":len(nouns),
       "new_prompts_count":len(examples),
       "new_templates_count":len(FROZEN_S100_FRAMES),
       "actual_physical_model_CPU":True,
       "all_full_EOS_decoded":True,
       "S99_PRIOR_TRAIN_456_full":p99["S99_new_correct_full_EOS"],
       "frozen_S99_prior_all_inclusive_bpp":p99["conservative_entire_physical_bpp"],
       "once_only_new_results":scores,
       "strict_cross_model_changes":effects,
       "new_neural_copy_candidate_opportunities":eligible,
       "new_neural_copy_overrides_applied":applied,
       "selected_profiles":dict(Counter(labels)),
       "paired_CPU_S95_over_S99_ratio":ratio_s95,
       "paired_CPU_S82_over_S99_ratio":ratio_s82,
       "S99_CPU_ge_90pct_S95":ratio_s95>=0.9,
       "S99_CPU_ge_90pct_S82":ratio_s82>=0.9,
       "candidate_accepted":False,
       "external_sealed_QA_unverified":True,
       "native_C_and_MPS_parity_unverified":True,
       "LIVE_untouched":True,
    }
    dest=a.out/"S100_UNSEEN_NONLEAKING_64_FULL_EOS_PHYSICAL_CPU.json"
    dest.write_text(json.dumps(report,sort_keys=True,indent=2))
    print(json.dumps({"event":"S100_FROZEN_UNSEEN_64_EXACT_CPU_FULL_EOS",
      "S82":scores["S82"],"S95":scores["S95"],"S99":scores["S99"],
      "net_delta_vs_S95":effects,
      "S95_over_S99_cpu_speed":ratio_s95,
      "S82_over_S99_cpu_speed":ratio_s82,
      "candidate_accepted":False}),flush=True)
    # A negative result is scientifically useful and MUST remain visible.
    # Never accept from this new developmental set without independent
    # truly sealed QA, CPU/GPU and C-runtime certification.
    if ratio_s82<0.90 or effects["S95_to_S99_lost_full"]:
        raise SystemExit("S100 novel generalization/speed guard REJECTED; "
                         "do not retrain on inspected prompts")


if __name__=="__main__":
    main()
