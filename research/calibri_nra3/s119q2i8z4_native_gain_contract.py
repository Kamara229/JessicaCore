"""S119q2i8z4: PREDECLARED PHYSICAL true43M 12GT native49152 readout study.

Exactly THREE SHA-verified previously fitted physical FP16 neural children:
W prior, Z2 matched CE and Z2 same+TRAIN-only Core L2.
At gold-previous-native BPE positions two and three on 14 already
SCORED synthetic Z2 author DEV (NOT blind), probe reversible existing
rank4 native head gains {0,.25,1} with full original43M and GT12.
No optimizer, backward, training, source mutation or production.
"""
from __future__ import annotations
from collections import Counter
import statistics

STAGE="CALIBRI_S119Q2I8Z4_TRUE43M_12GT_CORE_NRA3_NPA1_NATIVE49K_READOUT_GAIN_GOLD_BPE2_BPE3"
STATES=("W_PREVIOUS_EARLY23_X3_FROZEN",
        "Z2_FULLJOINT_EARLY23_X3_CE",
        "Z2_FULLJOINT_EARLY23_X3_CORE_TRAINONLY_L2")
POSITIONS=(1,2)
GAINS=(0.0,0.25,1.0)
FRESH_DEV_ALREADY_SPENT=14
NATIVE_VOCAB=49152
EXPECTED_PHYSICAL_FORWARDS=len(STATES)*len(POSITIONS)*len(GAINS)*FRESH_DEV_ALREADY_SPENT

def check(ok,msg):
    if not ok:raise ValueError("CALIBRI S119q2i8z4 strict research gate: "+msg)

def classify_summary(records):
    check(len(records)==EXPECTED_PHYSICAL_FORWARDS==252,
          "exact 252 original S142 43M physical gold-BPE2/3 rank fullnative")
    output={}
    for state in STATES:
        per_state=[x for x in records if x["state"]==state]
        check(len(per_state)==84,
              "each original43M physical neural FP16 state must get 14x2x3")
        output[state]={}
        for position in POSITIONS:
            rows=[x for x in per_state if x["position_native_zero_based"]==position]
            check(len(rows)==42 and len({x["source_id"] for x in rows})==14,
                  "exact 14 already spent same-source synthetic Z2 examples per position")
            grouped={}
            for gain in GAINS:
                rr=[x for x in rows if x["native_head_gain"]==gain]
                check(len(rr)==14
                      and len({x["source_id"] for x in rr})==14
                      and all(x["source_already_spent_author_Z2_DEV"] is True
                              and x["given_GOLD_PREVIOUS_BPE_only_NOT_OWN"] is True
                              and x["correct_gold_native_id"]!=2
                              and 1<=x["correct_gold_native_rank_full49152"]<=NATIVE_VOCAB
                              and x["greedy_native_id"] in range(NATIVE_VOCAB)
                              for x in rr),
                      "no lost/spoofed original full49k source, no leaked answer")
                grouped[gain]={x["source_id"]:x for x in rr}
            check(set(grouped[0])==set(grouped[.25])==set(grouped[1]),
                  "strict physical paired Z2 same source at all predeclared gains")
            reference=grouped[.25]
            out={}
            for gain in GAINS:
                vals=grouped[gain]
                rows_one=list(vals.values())
                top1=sum(x["correct_gold_top1"] for x in rows_one)
                top5=sum(x["correct_gold_rank_top5"] for x in rows_one)
                out[f"native_rank4_gain_{gain:g}"]={
                    "n_ALREADY_SPENT_author_DEV":14,
                    "correct_gold_native_top1":top1,
                    "correct_gold_native_top5":top5,
                    "mean_gold_second_or_third_rank_full49152":
                        round(statistics.mean(x["correct_gold_native_rank_full49152"]
                                              for x in rows_one),4),
                    "median_gold_second_or_third_rank_full49152":
                        statistics.median(x["correct_gold_native_rank_full49152"]
                                          for x in rows_one),
                    "mean_gold_native_49k_NLL_nats":
                        round(statistics.mean(x["correct_gold_native_NLL_nats"]
                                              for x in rows_one),7),
                    "mean_wrong_pred_minus_correct_gold_logit_gap":
                        round(statistics.mean(x["wrong_top1_minus_gold_native_logit_gap"]
                                              for x in rows_one),6),
                    "EOS_wrong_greedy_first":sum(x["greedy_native_id"]==2 for x in rows_one),
                    "greedy_native_top1_ID_changed_vs_original_gain025":
                        sum(x["greedy_native_id"]!=reference[id]["greedy_native_id"]
                            for id,x in vals.items()),
                    "paired_gold_rank_better_than_gain025":
                        sum(x["correct_gold_native_rank_full49152"]<
                            reference[id]["correct_gold_native_rank_full49152"]
                            for id,x in vals.items()),
                    "paired_gold_rank_worse_than_gain025":
                        sum(x["correct_gold_native_rank_full49152"]>
                            reference[id]["correct_gold_native_rank_full49152"]
                            for id,x in vals.items()),
                    "not_NEWDEV_or_OWN_generation_accuracy":True,
                }
            output[state][f"teacherforced_previous_gold_position_{position+1}"]=out
    return output
