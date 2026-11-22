"""S119q2i8z predeclared ORIGINAL43M real-native 49152 causal source probes.

Original S142 43M physically on-path 12 GT12 + existing original
Core/NRA3/NPA1 & rank4 native language readout, with fully SHA verified
prior physical W parent, X A trained CE and X B trained nonEOS
hardnegative. All 14 author X DEV already scored (SPENT). Diagnostics
only: exactly 8 reversible toggles per original author prompt/state
at GOLD-PREVIOUS first native BPE (teacher-forced READONLY, not OWN).
No learning, model-selection, masked inference, new regions or blind test.
"""
from __future__ import annotations
from collections import Counter
from contextlib import nullcontext

STAGE="CALIBRI_S119Q2I8Z_FULL_ORIGINAL43M_GT12_VS_REGION_GATES_VS_NATIVE_HEAD_CONTEXT_CAUSAL"
VOCAB=49152
SOURCE_COUNT=14
MODELS=("W_PREVIOUS_EARLY23_X3_FROZEN",
        "X_FULLJOINT_EARLY23_X3_CE",
        "X_FULLJOINT_EARLY23_X3_HARDNEG")
CONDITIONS={
    "FULL_RECURRENT_GT12_ALL_REGIONS_HEAD_GAIN025":("GT_RECURRENT",1.,.25,(1,1,1)),
    "ZERO_ALL12_GT_GATES_ALL_REGIONS_HEAD_GAIN025":("GT_RECURRENT",0.,.25,(1,1,1)),
    "HEAD_GAIN_ZERO_GT12_AND_REGIONS_FITTED":("GT_RECURRENT",1.,0.,(1,1,1)),
    "ZERO_GT12_AND_ZERO_NATIVE_HEAD_REGIONS_ON":("GT_RECURRENT",0.,0.,(1,1,1)),
    "GT12_NO_INTERLAYER_CARRY_REGIONS_HEAD025":("GT_NOCARRY_LOCAL",1.,.25,(1,1,1)),
    "CORE_OFF_GT12_AND_NRA3_NPA1_HEAD025":("GT_RECURRENT",1.,.25,(0,1,1)),
    "NRA3_OFF_GT12_AND_CORE_NPA1_HEAD025":("GT_RECURRENT",1.,.25,(1,0,1)),
    "NPA1_OFF_GT12_AND_CORE_NRA3_HEAD025":("GT_RECURRENT",1.,.25,(1,1,0)),
}
FULL=next(iter(CONDITIONS))
EXPECTED_REAL_ORIGINAL_43M_FORWARDS=SOURCE_COUNT*len(MODELS)*len(CONDITIONS)

def check(ok,msg):
    if not ok:
        raise ValueError("S119q2i8z strict original43M CAUSAL context check: "+msg)

def config(label):
    check(label in CONDITIONS and len(CONDITIONS)==8,"only registered controls")
    route,gt_scale,gain,regions=CONDITIONS[label]
    check(route in ("GT_RECURRENT","GT_NOCARRY_LOCAL")
          and gt_scale in (0.,1.) and gain in (0.,.25)
          and len(regions)==3
          and all(x in (0,1) for x in regions),
          "true 12GT+three original component gates+native rank4 readout limited toggles")
    return route,gt_scale,gain,regions

def aggregate(rows):
    check(len(rows)==EXPECTED_REAL_ORIGINAL_43M_FORWARDS,
          "all 336 original full49152 actual teacher forced native rank probes must exist")
    results={}
    for model in MODELS:
        model_rows=[r for r in rows if r["model"]==model]
        check(len(model_rows)==SOURCE_COUNT*len(CONDITIONS),
              "exact same 14 already scored X synthetic RU sources per trained model")
        ids={r["source_id"] for r in model_rows}
        check(len(ids)==SOURCE_COUNT and
              Counter(r["condition"] for r in model_rows)=={
                  condition:SOURCE_COUNT for condition in CONDITIONS},
              "each registered component/head/gate ablation scored for each old author example")
        by_id={}
        for r in model_rows:
            check(r["condition"] in CONDITIONS
                  and r["gold_49152_BPE_rank"] in range(1,VOCAB+1)
                  and r["source_is_ALREADY_SPENT_synthetic_DEV"] is True
                  and r["teacher_forced_PREVIOUS_gold_BPE_not_own"] is True
                  and r["gold_BPE_target_native_id"]!=2,
                  "only true original43M native49152 gold BPE2 rank no blind or EOS target")
            by_id.setdefault(r["source_id"],{})[r["condition"]]=r
        check(all(set(v)==set(CONDITIONS) for v in by_id.values()),
              "exact 8 controls for every same X author source")
        full={i:d[FULL] for i,d in by_id.items()}
        model_out={}
        for label in CONDITIONS:
            rr=[d[label] for d in by_id.values()]
            count_better=sum(d[label]["gold_49152_BPE_rank"]<
                             full[i]["gold_49152_BPE_rank"] for i,d in by_id.items())
            count_worse=sum(d[label]["gold_49152_BPE_rank"]>
                            full[i]["gold_49152_BPE_rank"] for i,d in by_id.items())
            count_equal=SOURCE_COUNT-count_better-count_worse
            model_out[label]={
                "already_spent_DEV_n":SOURCE_COUNT,
                "gold_second_BPE_top1":sum(x["gold_top1"] for x in rr),
                "gold_second_BPE_top5":sum(x["gold_top5"] for x in rr),
                "mean_gold_second_native_rank":round(sum(x["gold_49152_BPE_rank"]
                                                        for x in rr)/SOURCE_COUNT,6),
                "mean_gold_second_NLL_nats":round(sum(x["gold_49152_BPE_NLL"]
                                                       for x in rr)/SOURCE_COUNT,6),
                "wrong_gold_second_native_EOS_pred":sum(x["winner_is_EOS"] for x in rr),
                "top1_winner_CHANGED_vs_FULL_same_source":sum(
                    d[label]["predicted_BPE_id"]!=full[i]["predicted_BPE_id"]
                    for i,d in by_id.items()),
                "ablation_better_gold_rank_than_FULL":count_better,
                "ablation_worse_gold_rank_than_FULL":count_worse,
                "ablation_same_gold_rank_as_FULL":count_equal,
                "mean_abs_native_logit_Linf_CHANGE_vs_FULL":round(
                    sum(x["original_full49152_logit_Linf_delta_vs_FULL"]
                        for x in rr)/SOURCE_COUNT,6),
                "new_DEV_is_SPENT_not_blind":True,
            }
        results[model]=model_out
    return results
