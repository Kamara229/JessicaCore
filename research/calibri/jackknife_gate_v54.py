#!/usr/bin/env python3
"""CALIBRI v54 frozen 4-family + 8 jackknife TRAIN-window acceptance gate.

Avoid equating two windows of one corpus with two independent populations.
One fixed 4-corpus macro-average, independent family leave-out, and
window-wise jackknife eligibility consistency. All NLL inputs TRAIN-only.
Physical full-model INT6/CR47 patch construction is frozen from v53.
"""
from __future__ import annotations
import math

METHODS=("base_int6_pre","sse_cap_06pct","sse_cap_12pct","sse_cap_24pct")
FAMILIES={
    "WikiText":("WikiText_TRAIN_G_12288_14336","WikiText_TRAIN_H_14336_16384"),
    "AGNews":("AGNews_TRAIN_1280_1408","AGNews_TRAIN_1408_1536"),
    "SciQ":("SciQ_TRAIN_256_384","SciQ_TRAIN_384_512"),
    "BoolQ":("BoolQ_TRAIN_0_128","BoolQ_TRAIN_128_256"),
}
MIN_POSITIVE_FAMILIES=3
MIN_MACRO_NLL_GAIN=0.0005
MIN_FAMILY_NLL_GAIN=-0.0080
MIN_WORST_LOFO_NLL_GAIN=0.0005
MIN_SUCCESSFUL_JACKKNIFE_FOLDS=6
JACKKNIFE_FOLDS=8


def _summary(window_gains):
    family={name:math.fsum(window_gains[w] for w in windows)/len(windows)
            for name,windows in FAMILIES.items()}
    macro=math.fsum(family.values())/len(family)
    lofo={out:math.fsum(v for n,v in family.items() if n!=out)/(len(family)-1)
          for out in FAMILIES}
    positive=sum(v>0 for v in family.values())
    worst=min(family.values())
    floor=min(lofo.values())
    accept=(positive>=MIN_POSITIVE_FAMILIES and
            macro>=MIN_MACRO_NLL_GAIN and
            worst>=MIN_FAMILY_NLL_GAIN and
            floor>=MIN_WORST_LOFO_NLL_GAIN)
    return {"four_family_NLL_improvements":family,
            "equal_family_macro_NLL_gain":macro,
            "positive_family_count":positive,
            "worst_single_family_NLL_gain":worst,
            "leave_one_family_out_three_family_means":lofo,
            "worst_leave_one_family_out_NLL_gain":floor,
            "passes_four_family_acceptance":accept}


def jackknife_family_gate(trained):
    """Select from real pretrained model NLL; never access heldout data.

    Primary: full 4-family eligibility (3 positive, macro gain, per-family
    downside and minimum LOFO mean). Secondary: drop every one of 8
    training windows individually, recompute the same family-balanced
    eligibility using the remaining one in its source family. Require
    6/8 jackknife folds pass. Select eligible candidate by number of
    successful folds, then worst LOFO, then macro, then smaller cap.
    This is a heuristic stability stress-test, NOT formal confidence.
    """
    names=tuple(w for wins in FAMILIES.values() for w in wins)
    if set(trained)!=set(METHODS):
        raise AssertionError("missing physically verified model/correction")
    for key in METHODS:
        if set(trained[key])!=set(names):
            raise AssertionError("missing one preregistered TRAIN window")
        if any(not math.isfinite(trained[key][w]["nll"]) for w in names):
            raise AssertionError("nonfinite full trained model NLL")
    reference=trained["base_int6_pre"]
    evidence={}
    for method in METHODS[1:]:
        gains={w:reference[w]["nll"]-trained[method][w]["nll"] for w in names}
        full=_summary(gains)
        jackknife={}
        for dropped in names:
            reduced={w:val for w,val in gains.items() if w!=dropped}
            fold_families={
                family:math.fsum(reduced[w] for w in win if w!=dropped)/
                       sum(w!=dropped for w in win)
                for family,win in FAMILIES.items()
            }
            fammean=math.fsum(fold_families.values())/4
            loo={excluded:math.fsum(x for f,x in fold_families.items()
                   if f!=excluded)/3 for excluded in FAMILIES}
            passes=(sum(v>0 for v in fold_families.values())>=3
                    and fammean>=MIN_MACRO_NLL_GAIN
                    and min(fold_families.values())>=MIN_FAMILY_NLL_GAIN
                    and min(loo.values())>=MIN_WORST_LOFO_NLL_GAIN)
            jackknife[dropped]={
                "four_family_macro_NLL_gain":fammean,
                "family_gains_after_dropping_window":fold_families,
                "worst_three_family_LOFO_NLL_gain":min(loo.values()),
                "passes":passes}
        passed=sum(row["passes"] for row in jackknife.values())
        eligible=(full["passes_four_family_acceptance"] and
                  passed>=MIN_SUCCESSFUL_JACKKNIFE_FOLDS)
        evidence[method]={
            "per_window_paired_NLL_gain":gains,
            "full_four_family":full,
            "leave_one_training_window_out_results":jackknife,
            "jackknife_passed_folds":passed,
            "jackknife_total_folds":JACKKNIFE_FOLDS,
            "eligible":eligible}
    options=[m for m in METHODS[1:] if evidence[m]["eligible"]]
    chosen=(max(options,key=lambda m:(
        evidence[m]["jackknife_passed_folds"],
        evidence[m]["full_four_family"]["worst_leave_one_family_out_NLL_gain"],
        evidence[m]["full_four_family"]["equal_family_macro_NLL_gain"],
        -METHODS.index(m))) if options else "base_int6_pre")
    return {
        "chosen":chosen,
        "candidate_training_stability":evidence,
        "method_order":list(METHODS),
        "family_window_mapping":{name:list(win) for name,win in FAMILIES.items()},
        "selection_rule":"4 equal TRAIN families >=3 positive, macro >=0.0005, worst family >=-0.008, min family-leaveout >=0.0005; >=6 of 8 leave-one-window-out passes; maximize jackknife wins then LOFO robustness, otherwise plain PRE",
        "parameters":{"minimum_positive_families":MIN_POSITIVE_FAMILIES,
             "minimum_macro":MIN_MACRO_NLL_GAIN,
             "minimum_single_family":MIN_FAMILY_NLL_GAIN,
             "minimum_leave_one_family_out_mean":MIN_WORST_LOFO_NLL_GAIN,
             "minimum_jackknife_folds":MIN_SUCCESSFUL_JACKKNIFE_FOLDS},
        "all_training_NLL_and_PPL":trained,
        "statistical_confidence_claim":False,
    }


def smoke():
    base={w:{"nll":5.} for wins in FAMILIES.values() for w in wins}
    good={w:{"nll":4.99} for w in base}
    bad={w:{"nll":5.01} for w in base}
    t={"base_int6_pre":base,
       "sse_cap_06pct":good,"sse_cap_12pct":bad,"sse_cap_24pct":bad}
    r=jackknife_family_gate(t)
    assert r["chosen"]=="sse_cap_06pct"
    assert r["candidate_training_stability"]["sse_cap_06pct"]["jackknife_passed_folds"]==8
    assert jackknife_family_gate({k:(base if k=="base_int6_pre" else bad)
                                for k in METHODS})["chosen"]=="base_int6_pre"
    # Counterexample: 2 good families is NOT 3 positive families even
    # if one source family happens to contribute much higher gains.
    unstable={}
    for name,wins in FAMILIES.items():
        for w in wins:
            unstable[w]={"nll":4.95 if name in ("WikiText","SciQ") else 5.001}
    t["sse_cap_06pct"]=unstable
    r=jackknife_family_gate(t)
    assert r["chosen"]=="base_int6_pre"
    print("CALIBRI v54 4-family 8-fold jackknife stability gate smoke PASS")


if __name__=="__main__":
    smoke()
