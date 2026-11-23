"""S119q2i8z6: source-paired, true S142 43M physical readout decomposition.

For each of THREE SHA-verified research-only fitted W/Z2 FP16 children
and 14 ALREADY SPENT original synthetic Z2 DEV, at two GOLD-PREVIOUS
second/third native BPE positions: execute ONE ORIGINAL S142 43M
full12GT Core/NRA3/NPA1 native49152 physical forward, capture
the *existing* NativeHeadAlignment raw additive delta PRE scaling,
decompose raw = learned context rank4 + learned output_bias exactly.

Score FOUR purely analytical, read-only same-hidden/full49152 variants:
full trained gain.25, base original no correction, bias ONLY,
context rank4 ONLY. 84 real forwards, 336 analytical full49k
comparisons, no trainable additional module, no inference change,
no protected holdout, no model promotion.
"""
from __future__ import annotations
from collections import Counter
import statistics

STAGE="CALIBRI_S119Q2I8Z6_ORIGINAL43M_ALL12GT_BPE2_BPE3_CONTEXT_RANK4_VS_GLOBAL_NATIVE_BIAS"
STATES=("W_PREVIOUS_EARLY23_X3_FROZEN",
        "Z2_FULLJOINT_EARLY23_X3_CE",
        "Z2_FULLJOINT_EARLY23_X3_CORE_TRAINONLY_L2")
POSITIONS=(1,2)
VARIANTS=("FITTED_FULL_GAIN025","ORIGINAL_NATIVE_43M_WITHOUT_DELTA",
          "LEARNED_BIAS_ONLY_GAIN025","LEARNED_CONTEXT_RANK4_ONLY_GAIN025")
VOCAB=49152
SOURCES=14
EXPECTED_PHYSICAL=3*14*2
EXPECTED_ANALYTIC=EXPECTED_PHYSICAL*len(VARIANTS)
GAIN=.25

def check(value,why):
    if not value:raise ValueError("S119q2i8z6 true physical readout fail-closed: "+why)

def summarize(records):
    check(len(records)==EXPECTED_ANALYTIC==336,
          "exact 336 full original49k gold rank scored from 84 physical S142 43M")
    pairs={}
    for row in records:
        check(row["state"] in STATES and
              row["position_native_zero_based"] in POSITIONS and
              row["variant"] in VARIANTS and
              row["synthetic_author_Z2_DEV_ALREADY_SPENT"] is True and
              row["human_GOLD_PREVIOUS_native_context_only_NOT_OWN"] is True and
              row["correct_native_gold_ID"] not in (None,2) and
              row["correct_native_rank_full49152"] in range(1,VOCAB+1) and
              row["greedy_native_ID"] in range(VOCAB),
              "actual same previously scored Z2 gold2/3, complete49k readout only")
        key=(row["state"],row["position_native_zero_based"],
             row["source_id"],row["variant"])
        check(key not in pairs,"cannot spoof duplicate physical same source variant")
        pairs[key]=row
    check(len(pairs)==EXPECTED_ANALYTIC,"all original source/position/mode unique")
    result={}
    for state in STATES:
        result[state]={}
        for position in POSITIONS:
            ids={k[2] for k in pairs if k[0]==state and k[1]==position}
            check(len(ids)==SOURCES,"exact same 14 previously scored author Z2 per fitted neural child")
            stats={}
            for variant in VARIANTS:
                rr=[pairs[(state,position,id,variant)] for id in sorted(ids)]
                check(len(rr)==SOURCES,"original fixed 43M same readout algebra states")
                vocab_winners=Counter(x["greedy_native_ID"] for x in rr)
                full=[pairs[(state,position,id,VARIANTS[0])] for id in sorted(ids)]
                stats[variant]={
                    "same14_spent_source_count":SOURCES,
                    "human_next_native_gold_top1":sum(x["correct_native_top1"] for x in rr),
                    "human_next_native_gold_top5":sum(x["correct_native_top5"] for x in rr),
                    "mean_full49152_gold_rank":round(statistics.mean(
                        x["correct_native_rank_full49152"] for x in rr),5),
                    "mean_full49152_gold_NLL_nats":round(statistics.mean(
                        x["correct_native_NLL_nats"] for x in rr),7),
                    "wrong_native_EOS_greedy":vocab_winners.get(2,0),
                    "distinct_competing_greedy_native_IDs":len(vocab_winners),
                    "greedy_native_TOP1_changed_vs_full_gain025":sum(
                        x["greedy_native_ID"]!=f["greedy_native_ID"]
                        for x,f in zip(rr,full)),
                    "paired_gold_rank_better_than_full":sum(
                        x["correct_native_rank_full49152"]<
                        f["correct_native_rank_full49152"]
                        for x,f in zip(rr,full)),
                    "paired_gold_rank_worse_than_full":sum(
                        x["correct_native_rank_full49152"]>
                        f["correct_native_rank_full49152"]
                        for x,f in zip(rr,full)),
                }
            comparisons=[]
            for id in sorted(ids):
                rr={variant:pairs[(state,position,id,variant)] for variant in VARIANTS}
                comparisons.append({
                    "source_id":id,
                    "bias_only_NLL_lower_than_rank4_only":rr[VARIANTS[2]][
                        "correct_native_NLL_nats"]<rr[VARIANTS[3]][
                        "correct_native_NLL_nats"],
                    "context_rank4_only_NLL_lower_than_bias_only":rr[VARIANTS[3]][
                        "correct_native_NLL_nats"]<rr[VARIANTS[2]][
                        "correct_native_NLL_nats"],
                    "bias_only_top1_agrees_with_full":rr[VARIANTS[2]][
                        "greedy_native_ID"]==rr[VARIANTS[0]]["greedy_native_ID"],
                    "context_rank4_only_top1_agrees_with_full":rr[VARIANTS[3]][
                        "greedy_native_ID"]==rr[VARIANTS[0]]["greedy_native_ID"],
                    "base_top1_agrees_with_full":rr[VARIANTS[1]][
                        "greedy_native_ID"]==rr[VARIANTS[0]]["greedy_native_ID"],
                })
            stats["same14_paired_interpretive_NLL_bias_vs_context"]={
                "rank4_context_only_better_NLL_cases":sum(
                    x["context_rank4_only_NLL_lower_than_bias_only"] for x in comparisons),
                "global_bias_only_better_NLL_cases":sum(
                    x["bias_only_NLL_lower_than_rank4_only"] for x in comparisons),
                "rank4_context_only_same_winner_as_full":sum(
                    x["context_rank4_only_top1_agrees_with_full"] for x in comparisons),
                "global_bias_only_same_winner_as_full":sum(
                    x["bias_only_top1_agrees_with_full"] for x in comparisons),
                "base_only_same_winner_as_full":sum(
                    x["base_top1_agrees_with_full"] for x in comparisons),
                "same14_author_spent_source_paired":comparisons,
            }
            result[state][f"read_only_gold_previous_native_position_{position+1}"]=stats
    return result
