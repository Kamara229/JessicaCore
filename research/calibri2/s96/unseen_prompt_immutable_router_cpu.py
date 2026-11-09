"""S96: one-time post-freeze genuinely new noun + prompt evaluation of S95.

No model or router training occurs. The S95 FP16 prompt-only classifier,
S94 exact Green Tunnel profile bank, and S93 compressed core are immutable.
S96 test cases are newly authored here from 8 previously excluded S82 nouns
and two never-before-trained prompt templates. Old, inspected QA/TEST
corpora are explicitly forbidden. A pass does not imply general intelligence.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s94.physical_profile_bank import sha,CHANGED,PROFILES
from calibri2.s95.train_and_audit_router import (
    PromptOnlyRouter,prompt_only_vector,read_verified_bank,CHOICES)
from v176.operators import clear

# Created AFTER S95 router training and freeze, never imported by S95.
# Both templates use novel wording and owner names relative to S82 TRAIN.
FROZEN_S96_UNSEEN_TEMPLATES=(
    "UNSEEN S96 ledger: In drawer {a}, the tagged object is {left}. "
    "A different drawer {b} has {right}. Give only {a}'s color and object as two words.",
    "UNSEEN S96 delivery sheet: The dispatch code {b} refers to {right}, "
    "while dispatch code {a} instead refers to {left}. "
    "Write the exact two-word phrase for dispatch code {a}.",
)
FROZEN_OWNER_PAIRS=(("Selara","Fenori"),("Kumari","Torevin"))
COLORS=("red","green","blue","yellow")


def independent_new_questions(tok):
    accepted=[n for n in s82.NOUNS if all(len(tok.encode(f"{color} {n}",
                   add_special_tokens=False))==2 for color in COLORS)]
    trained=set(s82.newly_authored_train(tok)[1])
    if len(trained)!=24 or trained!=set(accepted[:24]):
        raise AssertionError("S95 historic TRAIN target corpus no longer frozen")
    novel=[n for n in accepted[24:] if n not in trained]
    if len(novel)<8:
        raise AssertionError("S96 requires >=8 truly unused tokenizer-compatible nouns")
    novel=novel[:8]
    # Exactly 64 labelled prompts, independently preregistered before
    # looking at any S95 results on these prompts.
    examples=[]
    for color in COLORS:
        for k in range(4):
            a=f"{color} {novel[k]}"
            b=f"{color} {novel[k+4]}"
            for frame,(owner,other) in zip(FROZEN_S96_UNSEEN_TEMPLATES,FROZEN_OWNER_PAIRS):
                for left,right in ((a,b),(b,a)):
                    prompt=frame.format(a=owner,b=other,left=left,right=right)
                    gold=tok.encode(left,add_special_tokens=False)+[tok.eos_token_id]
                    if len(gold)!=3:
                        raise AssertionError("Expected exactly two tokenizer words plus EOS")
                    examples.append({"prompt":prompt,"gold":gold,"color":color,
                                     "noun":left.split()[1]})
    if len(examples)!=64 or len({x["prompt"] for x in examples})!=64:
        raise AssertionError("S96 did not produce 64 unique frozen prompts")
    old=set()
    for family,pairs in s87.families(tok).items():
        for pair in pairs:
            for side in ("left","right"):
                old.add(pair[side]["question"])
    if any(x["prompt"] in old for x in examples):
        raise AssertionError("S96 unseen prompts overlap prior TRAIN")
    return examples,novel


def swap(core_sources,inner,profile,original_centroids,original_tunnel,bank):
    with torch.no_grad():
        for name in CHANGED:
            value=original_centroids[name] if profile=="S93" else bank[profile][name]
            core_sources[name].centroids.copy_(value)
        state=original_tunnel if profile=="S93" else {
            k.removeprefix("internal_tunnel."):v for k,v in bank[profile].items()
            if k.startswith("internal_tunnel.")}
        inner.load_state_dict(state,strict=True)
    clear(core_sources)


def score(preds,gold):
    full=sum(a==b for a,b in zip(preds,gold))
    first=sum(bool(a) and a[0]==b[0] for a,b in zip(preds,gold))
    eos=sum(bool(a) and a[-1]==b[-1] for a,b in zip(preds,gold))
    return {"full_with_EOS":full,"first_token":first,"ends_in_EOS":eos,
            "count":len(gold),"strict_full_rate":full/len(gold)}


def main():
    p=argparse.ArgumentParser()
    for name in ("S95_router","S95_report","S94_bank","S94_report",
                 "S93_CB61","S93_sidecar","S82_old_CB59","S82_controller"):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    last=json.loads(a.S95_report.read_text())
    bank_meta=json.loads(a.S94_report.read_text())
    if (last.get("stage")!="S95_PROMPT_ONLY_NEURAL_THREE_PROFILE_ROUTER_TRAIN_RESEARCH" or
        not last.get("all_456_EOS_output_exact_routed") or
        not last.get("strict_train_eligibility_only") or
        last.get("frozen_three_profile_union_oracle_only")!=287 or
        last.get("candidate_accepted") is not False):
        raise AssertionError("S96 needs frozen, measured S95 TRAIN-only physical model")
    if (last["router_sha256"]!=sha(a.S95_router.read_bytes()) or
        last["green_tunnel_physical_bank_sha256"]!=sha(a.S94_bank.read_bytes()) or
        bank_meta["source_SHA256"]["S93"]!=sha(a.S93_CB61.read_bytes()) or
        bank_meta["sidecar_SHA256"]["S93"]!=sha(a.S93_sidecar.read_bytes())):
        raise AssertionError("S96 frozen S94/S95 physical artifact SHAs changed")
    router=PromptOnlyRouter().eval()
    router.load_state_dict(load_file(str(a.S95_router)),strict=True)
    assert all(not q.requires_grad or q.device.type=="cpu" for q in router.parameters())
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    examples,novel_nouns=independent_new_questions(tok)
    features=torch.tensor(np.stack([prompt_only_vector(tok,e["prompt"])
                      for e in examples]),dtype=torch.float32)
    with torch.no_grad():
        decision=router(features).argmax(dim=1).tolist()
    chosen=[CHOICES[i] for i in decision]
    bank=read_verified_bank(a.S94_bank,sha(a.S94_bank.read_bytes()))
    original_side=load_file(str(a.S93_sidecar))
    physical_source=s87.open_envelope(a.S93_CB61.read_bytes())
    model,sources,first,later,pointer,head,inner=s87.build_cpu(
            physical_source,original_side,True)
    if install_verified_fastpath(inner)!=("3","9"):
        raise AssertionError("S96 requires proven original zero-gate CPU fastpath")
    originals={k:sources[k].centroids.detach().clone() for k in CHANGED}
    gates={k.removeprefix("internal_tunnel."):v for k,v in original_side.items()
           if k.startswith("internal_tunnel.")}
    truths=[item["gold"] for item in examples]
    all_preds={}
    active=None
    times={}
    try:
        for profile in PROFILES:
            swap(sources,inner,profile,originals,gates,bank)
            active=profile
            generated=[]
            durations=[]
            for e in examples:
                start=time.perf_counter()
                ids=s82.full_cpu_generate(model,sources,first,later,head,tok,
                                          e["prompt"],pointer)
                durations.append(time.perf_counter()-start)
                generated.append(ids)
            times[profile]=durations
            all_preds[profile]=generated
            print(json.dumps({"event":"S96_FROZEN_UNSEEN_PROFILE",
                  "profile":profile,**score(generated,truths)}),flush=True)
        # Warm single-model end-to-end routed evaluation includes exact
        # core + internal Tunnel swaps; no answer table is ever loaded.
        routed=[]
        latencies=[]
        current=active
        switches=0
        for profile,e in zip(chosen,examples):
            start=time.perf_counter()
            if profile!=current:
                swap(sources,inner,profile,originals,gates,bank)
                current=profile
                switches+=1
            ids=s82.full_cpu_generate(model,sources,first,later,head,tok,
                                      e["prompt"],pointer)
            latencies.append(time.perf_counter()-start)
            routed.append(ids)
        if any(routed[i]!=all_preds[chosen[i]][i] for i in range(len(examples))):
            raise AssertionError("Physical routed answer differs from selected actual profile")
    finally:
        inner.uninstall()
        del model
    # Original S82 model and controller for an honest same-run reference.
    cb59=a.S82_old_CB59.read_bytes()
    if sha(cb59)!=last.get("source_original_sha256",sha(cb59)):
        raise AssertionError("Original CB59 changed")
    baseline_side=load_file(str(a.S82_controller))
    for k,v in baseline_side.items():
        if k not in original_side or not torch.equal(v,original_side[k]):
            raise AssertionError("Frozen S82 controller differs from physical profile")
    baseline,sources,one,two,ptr,head,_=s87.build_cpu(cb59,baseline_side,False)
    baseline_preds=[]
    baseline_times=[]
    for e in examples:
        start=time.perf_counter()
        baseline_preds.append(s82.full_cpu_generate(baseline,sources,
                             one,two,head,tok,e["prompt"],ptr))
        baseline_times.append(time.perf_counter()-start)
    del baseline
    router_scores=score(routed,truths)
    oracle=[truths[i] if any(all_preds[p][i]==truths[i] for p in PROFILES)
            else [] for i in range(len(examples))]
    oracle_full=sum(x==t for x,t in zip(oracle,truths))
    loss_vs_s82=sum(baseline_preds[i]==truths[i] and routed[i]!=truths[i]
                    for i in range(len(examples)))
    gain_vs_s82=sum(baseline_preds[i]!=truths[i] and routed[i]==truths[i]
                    for i in range(len(examples)))
    speed=statistics.median(baseline_times)/statistics.median(latencies)
    result={
        "stage":"S96_ONE_TIME_FROZEN_PREVIOUSLY_UNSEEN_PROMPTS_PHYSICAL_CPU_EOS",
        "sealed_QA_TEST_never_read":True,
        "S95_neural_router_and_all_profile_weights_frozen":True,
        "no_training_or_router_tuning_after_inspection":True,
        "newly_authored_unique_noun_count":len(novel_nouns),
        "all_target_nouns_disjoint_from_original_456_TRAIN":True,
        "unseen_prompts_count":len(examples),
        "new_prompt_templates_count":len(FROZEN_S96_UNSEEN_TEMPLATES),
        "TRAIN_measured_only_previous_stage":287,
        "unseen_routed_strict":router_scores,
        "unseen_S82_baseline_strict":score(baseline_preds,truths),
        "unseen_individual_profiles":{k:score(v,truths) for k,v in all_preds.items()},
        "unseen_three_profile_oracle_upper_bound_only":oracle_full,
        "unseen_regressions_vs_S82":loss_vs_s82,
        "unseen_gains_vs_S82":gain_vs_s82,
        "routed_physical_profile_selection":{p:chosen.count(p) for p in PROFILES},
        "actual_profile_switches":switches,
        "median_CPU_speed_ratio_S82_over_routed":speed,
        "routed_CPU_at_least_90pct_S82":speed>=0.9,
        "S95_all_inclusive_physical_bpp":last["physical_bpp_conservative"],
        "S95_meets_0_7_bpp":last["physical_bpp_at_most_0_7"],
        "MPS_and_native_C_parity_still_not_verified":True,
        "candidate_accepted":False,
        "LIVE_untouched":True,
    }
    (a.out/"S96_ONE_TIME_UNSEEN_CPU_GENERALIZATION_AUDIT.json").write_text(
        json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({"event":"S96_ONE_TIME_FROZEN_TRUE_NOVELTY_CPU",
        "unseen_count":len(truths),
        "S95_routed_full_plus_EOS":router_scores["full_with_EOS"],
        "S82_baseline_full_plus_EOS":result["unseen_S82_baseline_strict"]["full_with_EOS"],
        "three_profiles_oracle_upper_bound_only":oracle_full,
        "new_unseen_gains_vs_S82":gain_vs_s82,
        "lost_unseen_vs_S82":loss_vs_s82,
        "cpu_speed_ratio":speed,
        "candidate_accepted":False}),flush=True)
    if not result["routed_CPU_at_least_90pct_S82"]:
        raise SystemExit("REJECTED S96: CPU performance below 90% S82")
    if loss_vs_s82:
        raise SystemExit("REJECTED S96: forgotten original S82 skills on new prompts")
    if not gain_vs_s82:
        raise SystemExit("REJECTED S96: no new independent prompt gains")
    # Passing these tests is still not sealed external QA, so never promote.


if __name__=="__main__":
    main()
