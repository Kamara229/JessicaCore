"""Z48 READ-ONLY pinned Z47 first independent internal DEV28 failure taxonomy.

Reads EXACTLY one immutable completed Z47 report artifact, NEVER imports
torch or loads original S142/GT12, never scores DEV again, never optimizes
on SPENT DEV28. Classifies the observed first2 improvements alongside
cross-prompt native BPE2 collapse and premature EOS failure to guide a
*precommitted TRAIN-only* next design (with entirely fresh future DEV).

The Z46 28 DEV are consumed and NOT eligible for training or retesting.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from math import log2
from pathlib import Path

STAGE = "S119Q2I8Z48_READONLY_PINNED_Z47_SPENT_NEWDEV28_FIRST3_COLLAPSE_EOS_FORENSIC"
Z47_STAGE = "S119Q2I8Z47_PHYSICAL_REAL43M_FRESH28_DEV_BPE2_ONPOLICY_AND_CROSSPROMPT_DIVERSITY"
SOURCE_SHA = "c4489970fbb0868f47c15fe6cc47b348ec4c533e094e0775f2275e00bd4bc58c"
TRAIN_SHA = "a66833e8f4e83e4ba080dc3db0b3032a3315bcfc2fe7cee473e7e9ea7569023e"
MODES = ("ORIGINAL_NATIVE_GREEDY", "HUMAN_OWNFIRST_TRAIN_ONLY",
         "WRONG17_OWNFIRST_TRAIN_CONTROL")
KINDS = (
    "sentence_formulation", "case_agreement", "agent_action_object",
    "time_causality", "negation_scope", "arithmetic_units",
    "semantic_paraphrase",
)

def ensure(condition, message):
    if not condition:
        raise ValueError("Z48 source-pinned spent Z47 DEV28 only: "+message)

def diversity(ids):
    ensure(len(ids)>0 and all(type(x) is int and 0<=x<49152 for x in ids),
           "actual second native vocabulary ID list")
    n=len(ids)
    counts=Counter(ids)
    m=max(counts.values())
    return {
        "n":n,
        "distinct_IDs":len(counts),
        "empirical_crossprompt_entropy_bits":round(-sum(
            (v/n)*log2(v/n) for v in counts.values()),6),
        "majority_ID_count":m,
        "majority_fraction":round(m/n,6),
        "top_native_ID_counts":sorted(
            [{"native_ID":int(k),"count":v} for k,v in counts.items()],
            key=lambda x:(-x["count"],x["native_ID"]))[:5]
    }

def summarize(report):
    ensure(report["stage"]==Z47_STAGE and
           report["candidate_accepted"] is False and report["output_gate"]==0 and
           report["Z46_NEW28_before_physical_SHA256"]==SOURCE_SHA and
           report["previous_Z40_LEGAL_TRAIN84_SHA256"]==TRAIN_SHA and
           report["new_DEV28_now_SPENT_not_external_independent_blind"] is True and
           report["no_old_SPENT_DEV_teacher_forcing_optimizer_or_scoring"] is True and
           report["no_training_of_original_or_616J7_EVAL128_GOLD_bpp_CB59_CPU_certification"] is True and
           report["original43M_GT12_Core_NRA3_NPA1_Z10_and_all_shadow_weights_post_DEV_unchanged"] is True and
           report["true_original43M_prior_TRAIN_only_forwards"]==168 and
           report["true_original43M_total_physically_paid_forwards"]==308,
           "strict prior Z46 pinned 28 DEV and physically successful Z47 308 S142 forwards")
    rows=report["new_internal_DEV28_per_prompt_ONLY"]
    metrics=report["new_internal_DEV28_posthoc_autonomous_FIRST3_with_original_BPE3"]
    ensure(len(rows)==28 and
           len({x["source_id"] for x in rows})==28 and
           len({x["prompt_SHA256"] for x in rows})==28 and
           Counter(x["kind"] for x in rows)==Counter({kind:4 for kind in KINDS}) and
           sum(r["paid_original43M_native_forwards"] for r in rows)==140 and
           set(metrics["modes"])==set(MODES) and
           metrics["all_precommitted_nonpromoting_thresholds_pass"] is False,
           "all 28 distinct NEW previously spent DEV once, seven kinds x4, 140 own forwards")
    domain=defaultdict(list)
    baseline_ids=[]
    rows_score={mode:[] for mode in MODES}
    gains=Counter()
    eos=Counter()
    for row in rows:
        ensure(row["source_id"].startswith("S119Q2I8Z46_") and
               row["new_Z46_internal_DEV_now_SPENT"] is True and
               row["new_DEV_optimizer_steps"]==0 and
               set(row["results"])==set(MODES),
               "Z46 per-prompt consumed DEV, all three fixed arms, no optimization")
        first_by_arm=set()
        for mode in MODES:
            r=row["results"][mode]
            ids=r["native_first3_ids"]
            ensure(1<=len(ids)<=3 and
                   all(type(v) is int and 0<=v<49152 for v in ids) and
                   type(r["correct_first"]) is bool and
                   type(r["correct_first2"]) is bool and
                   type(r["correct_first3"]) is bool and
                   r["correct_first3"]<=r["correct_first2"]<=r["correct_first"] and
                   (r["correct_first2"] is False or len(ids)>=2) and
                   (r["correct_first3"] is False or len(ids)==3) and
                   r["premature_EOS_first3"]==(2 in ids) and
                   len(ids)>=2, "actual native 1–3 self-generated, no ghost padding")
            first_by_arm.add(ids[0])
            rows_score[mode].append(r)
            if len(ids)>=2 and ids[1]==2: eos[(mode,"eos_at_second")]+=1
            if len(ids)>=3 and ids[2]==2: eos[(mode,"eos_at_third")]+=1
        ensure(len(first_by_arm)==1 and
               row["first_correct"]==row["results"][MODES[0]]["correct_first"],
               "same unchanged native BPE1 across original, human and wrong control")
        original=row["results"][MODES[0]]
        human=row["results"][MODES[1]]
        wrong=row["results"][MODES[2]]
        ensure(all(row["results"][mode]["correct_first"]==row["first_correct"]
                   for mode in MODES) and
               (human["second"]["changed"]==
                   (human["native_first3_ids"][1]!=original["native_first3_ids"][1])) and
               (wrong["second"]["changed"]==
                   (wrong["native_first3_ids"][1]!=original["native_first3_ids"][1])),
               "same first and true BPE2 change counterfactual against original")
        pair=(original["correct_first2"],human["correct_first2"])
        gains[{(False,True):"human_corrects_original_first2_error",
               (True,False):"human_regresses_original_first2",
               (True,True):"human_preserves_original_first2_correct",
               (False,False):"both_first2_wrong"}[pair]]+=1
        o_eos=original["premature_EOS_first3"]
        h_eos=human["premature_EOS_first3"]
        eos[{(True,False):"human_resolves_original_early_EOS",
             (False,True):"human_introduces_early_EOS",
             (True,True):"both_early_EOS",
             (False,False):"neither_early_EOS"}[(o_eos,h_eos)]]+=1
        baseline_ids.append(original["native_first3_ids"][1])
        domain[row["kind"]].append(row)
    score_modes={}
    for mode in MODES:
        rr=rows_score[mode]
        m=metrics["modes"][mode]
        ids=[r["native_first3_ids"][1] for r in rr]
        d=diversity(ids)
        expected={
            "n":28,
            "correct_first_native":sum(r["correct_first"] for r in rr),
            "correct_first2_native":sum(r["correct_first2"] for r in rr),
            "correct_first3_native":sum(r["correct_first3"] for r in rr),
            "premature_EOS_first3":sum(r["premature_EOS_first3"] for r in rr),
            "native_second_distinct_IDs":d["distinct_IDs"],
            "native_second_crossprompt_empirical_entropy_bits":
                d["empirical_crossprompt_entropy_bits"]
        }
        ensure(all(m[key]==value for key,value in expected.items()),
               "archived Z47 public summary exactly reconciles with 28 individual DEV results")
        score_modes[mode]={**expected,"second_diversity_detail":d}
    ensure(
        [score_modes[x]["correct_first2_native"] for x in MODES]==[1,5,0] and
        [score_modes[x]["correct_first3_native"] for x in MODES]==[0,1,0] and
        [score_modes[x]["native_second_distinct_IDs"] for x in MODES]==[8,3,12] and
        [score_modes[x]["premature_EOS_first3"] for x in MODES]==[11,4,5] and
        [score_modes[x]["correct_first_native"] for x in MODES]==[26,26,26],
        "pin completed Z47 actual first2/first3/EOS and diversity outcome")
    domain_stats={}
    for kind,items in sorted(domain.items()):
        domain_stats[kind]={
            "n":len(items),
            "original_correct_first2":sum(r["results"][MODES[0]]["correct_first2"]
                                         for r in items),
            "human_correct_first2":sum(r["results"][MODES[1]]["correct_first2"]
                                      for r in items),
            "wrong_correct_first2":sum(r["results"][MODES[2]]["correct_first2"]
                                      for r in items),
            "human_early_EOS_first3":sum(r["results"][MODES[1]]["premature_EOS_first3"]
                                         for r in items),
            "human_BPE2_distinct_IDs":len({
                r["results"][MODES[1]]["native_first3_ids"][1] for r in items}),
            "original_BPE2_distinct_IDs":len({
                r["results"][MODES[0]]["native_first3_ids"][1] for r in items}),
        }
    return {
        "stage":STAGE,"pinned_source_stage":Z47_STAGE,
        "previous_spent_DEV28_SHA256":SOURCE_SHA,
        "archived_real_original43M_native_forwards":308,
        "new_original43M_native_forwards":0,
        "new_optimizer_steps":0,
        "DEV28_considered_SPENT_never_fit_or_rescore":True,
        "pinned_original_vs_human_vs_wrong_native_FIRST3":score_modes,
        "paired_original_HUMAN_first2_changes":dict(sorted(gains.items())),
        "paired_original_HUMAN_premature_EOS_changes":dict(sorted(
            (k,v) for k,v in eos.items() if not isinstance(k,tuple))),
        "EOS_at_native_BPE2_or_BPE3_by_mode":{
            mode:{
                "at_second":eos[(mode,"eos_at_second")],
                "at_third":eos[(mode,"eos_at_third")]}
            for mode in MODES},
        "seven_domain_within_newDEV_descriptives":domain_stats,
        "crossprompt_entropy_HUMAN_minus_original_bits":round(
            score_modes[MODES[1]]["second_diversity_detail"][
                "empirical_crossprompt_entropy_bits"]-
            score_modes[MODES[0]]["second_diversity_detail"][
                "empirical_crossprompt_entropy_bits"],6),
        "conclusion":"Z47_FAILED_PREDECLARED_CROSSPROMPT_DIVERSITY_AND_ZERO_EARLY_EOS: NO_PROMOTION; propose TRAIN_ONLY regularization on FRESH futura data, no spent DEV use",
        "candidate_accepted":False,"output_gate":0,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    report=json.loads(args.report.read_text(encoding="utf8"))
    result=summarize(report)
    ensure(not args.out.exists(),"no overwriting published research result")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,
                                   indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Z48_READONLY_FROZEN_SPENT_DEV28 "+json.dumps(
        result,ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
