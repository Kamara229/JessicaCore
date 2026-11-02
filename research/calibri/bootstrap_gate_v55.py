#!/usr/bin/env python3
"""CALIBRI v55 preregistered 4-family paired TRAIN bootstrap uncertainty gate.

The bootstrapped unit is the TRAIN CORPUS FAMILY, not individual language
model tokens. Draw four source families with replacement, and draw one of
two new disjoint TRAIN windows independently per selected family. Only
real NLL differences to the physically stored plain INT6 PRE baseline
enter selection; NO external evaluation labels/PPL are read before
frozen decision. Fixed seed and percentile make cross-CPU reproducible.

Bootstrap here is a *heuristic stability stress test* over FOUR corpus
families; a 10th percentile is NOT an inferential confidence bound on a
population of unseen applications.
"""
from __future__ import annotations
import math,random

METHODS=("base_int6_pre","sse_cap_06pct","sse_cap_12pct","sse_cap_24pct")
FAMILIES={
    "WikiText":("WikiText_TRAIN_I_16384_18432",
                "WikiText_TRAIN_J_18432_20480"),
    "AGNews":("AGNews_TRAIN_1536_1664","AGNews_TRAIN_1664_1792"),
    "SciQ":("SciQ_TRAIN_512_640","SciQ_TRAIN_640_768"),
    "BoolQ":("BoolQ_TRAIN_256_384","BoolQ_TRAIN_384_512"),
}
SEED=55
REPS=1024
LOWER_DECILE_INDEX=102  # floor((1024-1)*0.10)
FAMILY_WINS_MIN=3
WORST_FAMILY_FLOOR=-0.0080
BOOTSTRAP_LOWER_DECILE_MIN=0.0005


def stable_selection(training):
    """Only new TRAIN data; source correction byte geometry frozen."""
    names=tuple(w for wins in FAMILIES.values() for w in wins)
    if set(training)!=set(METHODS) or any(
            set(training[k])!=set(names) for k in METHODS):
        raise AssertionError("missing preregistered four-family TRAIN source")
    for method in METHODS:
        if any(not math.isfinite(training[method][w]["nll"])
               for w in names):
            raise AssertionError("nonfinite full pretrained quantized TRAIN NLL")
    source=training["base_int6_pre"]
    records={}
    for method in METHODS[1:]:
        gain={w:source[w]["nll"]-training[method][w]["nll"] for w in names}
        family_gain={f:math.fsum(gain[w] for w in windows)/2
                     for f,windows in FAMILIES.items()}
        rng=random.Random(SEED)
        families=tuple(FAMILIES)
        samples=[]
        for _ in range(REPS):
            sampled=[]
            for _ in families:
                family=families[rng.randrange(len(families))]
                window=FAMILIES[family][rng.randrange(2)]
                sampled.append(gain[window])
            samples.append(math.fsum(sampled)/len(families))
        samples.sort()
        lower=samples[LOWER_DECILE_INDEX]
        positive=sum(x>0 for x in family_gain.values())
        accepted=(positive>=FAMILY_WINS_MIN and
                  min(family_gain.values())>=WORST_FAMILY_FLOOR and
                  lower>=BOOTSTRAP_LOWER_DECILE_MIN)
        records[method]={
            "paired_training_gain_by_window":gain,
            "equal_family_mean_NLL_gains":family_gain,
            "positive_corpus_families":positive,
            "worst_family_NLL_gain":min(family_gain.values()),
            "macro_four_family_mean_NLL_gain":
                math.fsum(family_gain.values())/len(family_gain),
            "bootstrap_draws":REPS,"seed":SEED,
            "bootstrap_tenth_percentile_NLL_gain":lower,
            "bootstrap_median_NLL_gain":samples[(REPS-1)//2],
            "bootstrap_positive_share":sum(x>0 for x in samples)/REPS,
            "passes_frozen_heuristic_gate":accepted
        }
    eligible=[m for m in METHODS[1:]
              if records[m]["passes_frozen_heuristic_gate"]]
    chosen=max(eligible,key=lambda m:(
        records[m]["bootstrap_tenth_percentile_NLL_gain"],
        records[m]["macro_four_family_mean_NLL_gain"],
        -METHODS.index(m))) if eligible else "base_int6_pre"
    return {
        "chosen":chosen,
        "family_window_mapping":{k:list(v) for k,v in FAMILIES.items()},
        "method_order":list(METHODS),
        "bootstrap_seed":SEED,"bootstrap_repetitions":REPS,
        "lower_decile_index_zero_based":LOWER_DECILE_INDEX,
        "gate_rule":"at least 3/4 positive family average improvements; worst family>=-0.008; 10th percentile of 1024 paired four-family bootstrap NLL improvements>=0.0005; choose highest lower decile else PRE",
        "candidate_training_stability":records,
        "all_training_NLL_and_PPL":training,
        "not_a_statistical_confidence_bound":True,
    }


def smoke():
    names=tuple(w for ws in FAMILIES.values() for w in ws)
    baseline={w:{"nll":5.0} for w in names}
    good={w:{"nll":4.98} for w in names}
    bad={w:{"nll":5.02} for w in names}
    test={"base_int6_pre":baseline,"sse_cap_06pct":good,
          "sse_cap_12pct":bad,"sse_cap_24pct":bad}
    good_result=stable_selection(test)
    assert good_result["chosen"]=="sse_cap_06pct"
    assert good_result["candidate_training_stability"]["sse_cap_06pct"][
        "bootstrap_repetitions" if False else "bootstrap_draws"]==1024
    test["sse_cap_06pct"]=bad
    assert stable_selection(test)["chosen"]=="base_int6_pre"
    print("CALIBRI v55 TRAIN-only four-family paired bootstrap gate smoke PASS")


if __name__=="__main__":
    smoke()
