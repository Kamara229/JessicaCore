#!/usr/bin/env python3
"""CALIBRI v53: frozen equal-family, leave-one-family-out TRAIN gate.

All candidate physically encoded INT6 Pythia source and CR47 residual patches
are unchanged from verified v52/v51. Only their TRAIN-only selection
criterion changes. Three genuine source families, two disjoint training
2048-token excerpts apiece. Six windows are NOT six independent families.

Eligibility: improve at least 2/3 family-mean NLLs; equal-family macro NLL
gain >= +0.0005 nat/token; worst family gain >= -0.008; worst leave-one-
family-out mean (two other families) >= +0.0001. Rank eligibles by worst
leave-one-family-out mean, tie-break smaller correction cap.
This is an exploratory robustness heuristic, not a statistical CI or
formal proof of generalization.
"""
from __future__ import annotations
import math

METHODS=("base_int6_pre","sse_cap_06pct","sse_cap_12pct","sse_cap_24pct")
FAMILIES={
    "WikiText":("WikiText_TRAIN_E_8192_10240","WikiText_TRAIN_F_10240_12288"),
    "AGNews":("AGNews_TRAIN_1024_1152","AGNews_TRAIN_1152_1280"),
    "SciQ":("SciQ_TRAIN_0_128","SciQ_TRAIN_128_256"),
}
MIN_POSITIVE_FAMILIES=2
MIN_MACRO_GAIN=0.0005
MAX_FAMILY_LOSS=-0.0080
MIN_LEAVE_ONE_OUT_GAIN=0.0001


def frozen_family_gate(trained):
    """Select source-CB46 / physical-CR47 method using ONLY TRAIN NLL."""
    domains=tuple(d for win in FAMILIES.values() for d in win)
    if set(trained)!=set(METHODS):
        raise AssertionError("missing source-only INT6 baseline or frozen corrections")
    for method in METHODS:
        if set(trained[method])!=set(domains):
            raise AssertionError("required independent TRAIN family not measured")
        if any(not math.isfinite(trained[method][name]["nll"])
               for name in domains):
            raise AssertionError("non-finite pretrained quantized full-model NLL")
    base=trained["base_int6_pre"]
    outcomes={}
    for method in METHODS[1:]:
        gains={family:math.fsum(
            base[window]["nll"]-trained[method][window]["nll"]
            for window in windows)/len(windows)
            for family,windows in FAMILIES.items()}
        macro=math.fsum(gains.values())/len(FAMILIES)
        loo={excluded:math.fsum(
            gain for family,gain in gains.items() if family!=excluded)/2
            for excluded in FAMILIES}
        worst_loo=min(loo.values())
        wins=sum(gain>0 for gain in gains.values())
        passes=(wins>=MIN_POSITIVE_FAMILIES and
                macro>=MIN_MACRO_GAIN and
                min(gains.values())>=MAX_FAMILY_LOSS and
                worst_loo>=MIN_LEAVE_ONE_OUT_GAIN)
        outcomes[method]={
            "family_mean_NLL_gain":gains,
            "number_positive_corpus_families":wins,
            "equal_family_macro_NLL_gain":macro,
            "leave_one_family_out_pair_means":loo,
            "worst_leave_one_family_out_gain":worst_loo,
            "passes_preregistered_training_only_gate":passes}
    viable=[m for m in METHODS[1:]
            if outcomes[m]["passes_preregistered_training_only_gate"]]
    selected=(max(viable,key=lambda m:(
        outcomes[m]["worst_leave_one_family_out_gain"],
        -METHODS.index(m))) if viable else "base_int6_pre")
    return {
        "chosen":selected,
        "frozen_eligibility":"at least two of three training families improve; macro gain >=0.0005; worst-family gain >=-0.008; worst leave-one-family-out pair mean >=0.0001",
        "frozen_selection":"maximum minimum pairwise family mean gain, tie-break lower per-tensor percentage, otherwise plain physical INT6 PRE",
        "method_order":list(METHODS),
        "family_window_mapping":{k:list(v) for k,v in FAMILIES.items()},
        "candidate_training_stability":outcomes,
        "all_training_NLL_and_PPL":trained,
        "not_a_calibrated_confidence_interval":True,
    }


def smoke():
    # All values are fictional TRAIN NLL values for testing gate math only.
    # Production evaluation always uses actual pretrained Pythia checkpoint.
    b={name:{"nll":5.} for names in FAMILIES.values() for name in names}
    alt={}
    for method in METHODS[1:]:
        alt[method]={name:{"nll":5.01} for name in b}
    alt["sse_cap_06pct"]={name:{"nll":4.995} for name in b}
    outcome=frozen_family_gate({"base_int6_pre":b,**alt})
    assert outcome["chosen"]=="sse_cap_06pct"
    allbad={"base_int6_pre":b,**{
        method:{name:{"nll":5.01} for name in b} for method in METHODS[1:]}}
    assert frozen_family_gate(allbad)["chosen"]=="base_int6_pre"
    # One bad corpus family with 2 good ones: leave-one-out must bound
    # fragile positive aggregate that hides domain harm.
    frag={name:{"nll":4.99 if not name.startswith("SciQ") else 5.007}
          for name in b}
    fragtest={"base_int6_pre":b,**allbad}
    fragtest["sse_cap_06pct"]=frag
    result=frozen_family_gate(fragtest)
    assert result["candidate_training_stability"]["sse_cap_06pct"][
        "number_positive_corpus_families"]==2
    assert result["candidate_training_stability"]["sse_cap_06pct"][
        "worst_leave_one_family_out_gain"]>0
    print("CALIBRI v53 3-family leave-one-out TRAIN-only heuristic smoke PASS")


if __name__=="__main__":
    smoke()
