"""Z48 forensic: READ ONLY previously SPENT Z47 28 internal DEV outcomes.

The Z46 precommitted heldout was already scored ONCE in completed physical
GitHub Actions run #45372181111. This audit reads the immutable report
artifact only. No model, weights, tokenizer, prompts, new training, DEV
scoring, guessed grades, EVAL128/J7/GOLD or deploy/research promotion.

It quantifies the empirical cross-prompt native BPE2 distribution and
specific paired gains/losses to diagnose why a precommitted quality + entropy
+ EOS composite gate failed. Observations on SPENT DEV must NOT tune a
replacement ranker. Any new method must be trained/evaluated on LEGAL TRAIN
and independently precommitted before the NEXT entirely unseen DEV.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

Z47_STAGE="S119Q2I8Z47_PHYSICAL_REAL43M_FRESH28_DEV_BPE2_ONPOLICY_AND_CROSSPROMPT_DIVERSITY"
Z46_SHA="c4489970fbb0868f47c15fe6cc47b348ec4c533e094e0775f2275e00bd4bc58c"
LEGAL84_SHA="a66833e8f4e83e4ba080dc3db0b3032a3315bcfc2fe7cee473e7e9ea7569023e"
MODES=("ORIGINAL_NATIVE_GREEDY","HUMAN_OWNFIRST_TRAIN_ONLY",
       "WRONG17_OWNFIRST_TRAIN_CONTROL")
KINDS=("sentence_formulation","case_agreement","agent_action_object",
       "time_causality","negation_scope","arithmetic_units",
       "semantic_paraphrase")

def require(ok,reason):
    if not ok: raise ValueError("Z48 immutable SPENT DEV28 only: "+reason)

def empirical(ids):
    require(ids and all(type(i) is int and 0<=i<49152 for i in ids),
            "strict native49152 ID sequence")
    count=Counter(ids)
    n=len(ids)
    entropy=-sum((c/n)*math.log2(c/n) for c in count.values())
    return {
        "n":n,
        "distinct_native_IDs":len(count),
        "crossprompt_empirical_entropy_bits":round(entropy,6),
        "most_common_native_ID_frequency":max(count.values()),
        "most_common_native_ID_share":round(max(count.values())/n,6),
        "top3_native_IDs_total":sum(k for _,k in count.most_common(3)),
        "second_BPE_Herfindahl":round(sum((c/n)**2 for c in count.values()),6),
        "native_ID_frequency_histogram":dict(sorted(
            ((str(token),count) for token,count in count.items()),
            key=lambda x:(-x[1],int(x[0]))))
    }

def one_kind(rows):
    count=len(rows)
    require(count==4,"each originally registered DEV kind has four SPENT rows")
    b=[r["results"][MODES[0]] for r in rows]
    h=[r["results"][MODES[1]] for r in rows]
    w=[r["results"][MODES[2]] for r in rows]
    return {
        "n":count,
        "original_first2":sum(r["correct_first2"] for r in b),
        "human_first2":sum(r["correct_first2"] for r in h),
        "wrong17_first2":sum(r["correct_first2"] for r in w),
        "human_third":sum(r["correct_first3"] for r in h),
        "original_unique_second":len({r["native_first3_ids"][1] for r in b}),
        "human_unique_second":len({r["native_first3_ids"][1] for r in h}),
        "human_early_EOS_first3":sum(r["premature_EOS_first3"] for r in h),
    }

def audit(report):
    require(report["stage"]==Z47_STAGE and
            report["Z46_NEW28_before_physical_SHA256"]==Z46_SHA and
            report["previous_Z40_LEGAL_TRAIN84_SHA256"]==LEGAL84_SHA and
            report["candidate_accepted"] is False and report["output_gate"]==0 and
            report["new_DEV28_now_SPENT_not_external_independent_blind"] is True and
            report["no_old_SPENT_DEV_teacher_forcing_optimizer_or_scoring"] is True and
            report["original43M_GT12_Core_NRA3_NPA1_Z10_and_all_shadow_weights_post_DEV_unchanged"] is True and
            report["true_original43M_prior_TRAIN_only_forwards"]==168 and
            report["true_original43M_new_DEV28_native_forwards"]==140 and
            report["true_original43M_total_physically_paid_forwards"]==308,
            "one successful physical Z47 source-sha Z46 + frozen original S142 provenance")
    score=report["new_internal_DEV28_posthoc_autonomous_FIRST3_with_original_BPE3"]
    rows=report["new_internal_DEV28_per_prompt_ONLY"]
    require(len(rows)==28 and
            len({x["source_id"] for x in rows})==28 and
            len({x["prompt_SHA256"] for x in rows})==28 and
            sum(x["paid_original43M_native_forwards"] for x in rows)==140 and
            all(x["new_Z46_internal_DEV_now_SPENT"] is True and
                x["new_DEV_optimizer_steps"]==0 and
                x["paid_original43M_native_forwards"]==5 and
                set(x["results"])==set(MODES) for x in rows),
            "exactly previously SPENT Z46 NEW28, 140 physical DEV forwards")
    groups=defaultdict(list)
    for row in rows:
        require(row["kind"] in KINDS and
                row["source_id"].startswith("S119Q2I8Z46_"+row["kind"]+"_"),
                "same previously registered 7 kinds 4 prompts")
        groups[row["kind"]].append(row)
        decisions=row["results"]
        original=decisions[MODES[0]]
        tokens=original["native_first3_ids"]
        require(len(tokens)==3 and all(type(x) is int and 0<=x<49152
                for x in tokens) and
                original["second"]["alpha"]==0 and
                original["second"]["changed"] is False,
                "original native greedy BPE1/2/3 correct attribution")
        for mode in MODES:
            d=decisions[mode]
            t=d["native_first3_ids"]
            require(len(t)==3 and all(type(x) is int and 0<=x<49152 for x in t) and
                    t[0]==tokens[0] and
                    d["correct_first"]==row["first_correct"] and
                    type(d["correct_first2"]) is bool and
                    type(d["correct_first3"]) is bool and
                    (not d["correct_first3"] or d["correct_first2"]) and
                    d["premature_EOS_first3"]==(2 in t) and
                    d["second"]["candidate_original_top64"] is True and
                    d["second"]["alpha"]>=0 and
                    d["second"]["changed"]==(t[1]!=tokens[1]),
                    "strict on-policy BPE1 same; BPE2 candidate; BPE3 not oracle")
    require(set(groups)==set(KINDS) and
            all(len(rows)==4 for rows in groups.values()),
            "7 complete original Z46 4-row strata, no reconstructed NEW DEV")
    modes={}
    for mode in MODES:
        m=[r["results"][mode] for r in rows]
        ids=[x["native_first3_ids"][1] for x in m]
        modes[mode]={
            "correct_first":sum(x["correct_first"] for x in m),
            "correct_first2":sum(x["correct_first2"] for x in m),
            "correct_first3":sum(x["correct_first3"] for x in m),
            "premature_EOS_first3":sum(x["premature_EOS_first3"] for x in m),
            "second_changed_vs_original":
                sum(x["second"]["changed"] for x in m),
            "per_prompt_entropy_guard_active":
                sum(x["second"]["alpha"]>0 for x in m),
            "crossprompt_BPE2":empirical(ids),
        }
        original_score=score["modes"][mode]
        require(
            modes[mode]["correct_first"]==original_score["correct_first_native"] and
            modes[mode]["correct_first2"]==original_score["correct_first2_native"] and
            modes[mode]["correct_first3"]==original_score["correct_first3_native"] and
            modes[mode]["premature_EOS_first3"]==original_score["premature_EOS_first3"] and
            modes[mode]["second_changed_vs_original"]==original_score["per_prompt_second_control_applied"] and
            modes[mode]["per_prompt_entropy_guard_active"]==original_score["per_prompt_second_guard_nonzero"] and
            modes[mode]["crossprompt_BPE2"]["distinct_native_IDs"]==
                original_score["native_second_distinct_IDs"] and
            modes[mode]["crossprompt_BPE2"]["crossprompt_empirical_entropy_bits"]==
                original_score["native_second_crossprompt_empirical_entropy_bits"],
            "recomputed original Z47 published score and entropy exact parity")
    b=[r["results"][MODES[0]] for r in rows]
    h=[r["results"][MODES[1]] for r in rows]
    paired={
        "human_fixes_baseline_wrong_first2":sum(
            not orig["correct_first2"] and human["correct_first2"]
            for orig,human in zip(b,h)),
        "human_breaks_baseline_correct_first2":sum(
            orig["correct_first2"] and not human["correct_first2"]
            for orig,human in zip(b,h)),
        "both_original_and_human_correct_first2":sum(
            orig["correct_first2"] and human["correct_first2"]
            for orig,human in zip(b,h)),
        "both_original_and_human_wrong_first2":sum(
            not orig["correct_first2"] and not human["correct_first2"]
            for orig,human in zip(b,h)),
        "human_early_EOS_at_BPE3":sum(
            2 in human["native_first3_ids"][2:]
            for human in h),
        "human_early_EOS_at_BPE2":sum(
            human["native_first3_ids"][1]==2 for human in h),
    }
    require(sum(paired[k] for k in (
        "human_fixes_baseline_wrong_first2",
        "human_breaks_baseline_correct_first2",
        "both_original_and_human_correct_first2",
        "both_original_and_human_wrong_first2"))==28,
        "paired change exact 28")
    require(modes[MODES[1]]["correct_first2"]==5 and
            modes[MODES[0]]["correct_first2"]==1 and
            modes[MODES[2]]["correct_first2"]==0 and
            modes[MODES[1]]["crossprompt_BPE2"]["distinct_native_IDs"]==3 and
            modes[MODES[0]]["crossprompt_BPE2"]["distinct_native_IDs"]==8 and
            modes[MODES[1]]["premature_EOS_first3"]==4 and
            modes[MODES[0]]["premature_EOS_first3"]==11 and
            score["all_precommitted_nonpromoting_thresholds_pass"] is False,
            "read-only genuinely failed initial prereg Z46 safeguards")
    failed=[name for name,passed in score[
        "precommitted_diagnostic_thresholds"].items() if not passed]
    require(set(failed)=={
        "human_second_distinct_at_least_75percent_original",
        "human_crossprompt_entropy_drop_at_most_0_25_bits",
        "human_no_premature_EOS_first3",
        }, "unchanged documented 3 prereg failure points; never claim success")
    return {
        "stage":"S119Q2I8Z48_READONLY_PINNED_Z47_SPENT_DEV28_BPE2_DIVERSITY_EOS_FORENSIC",
        "prior_Z47_stage":Z47_STAGE,
        "Z46_NEW28_becomes_SPENT_already":True,
        "no_new_original43M_native_forwards":True,
        "reference_Z47_prior_paid_43M_native_forwards":308,
        "no_optimizer_no_weight_updates_no_new_data":True,
        "no_DEV_remediation_or_hyperparameter_search":True,
        "original_Z46_preregister_gates_FAILED":failed,
        "per_mode":modes,
        "paired_first2_forensic":paired,
        "per_kind_previous_DEV_ONLY":{
            k:one_kind(groups[k]) for k in KINDS
        },
        "limitations":"READONLY POSTHOC on SPENT internally authored DEV28; do not use for training, hyperparameters, threshold selection, validation, acceptance or promotion",
        "candidate_accepted":False,"output_gate":0,
        "next_safe_action":"precommit alternate on-policy anti-collapse objective on OLD LEGAL TRAIN84 only; use independent NEW heldout for later comparison",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=audit(json.loads(a.report.read_text(encoding="utf-8")))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                     encoding="utf8")
    print("S119Q2I8Z48_READONLY_SPENT28_FORENSIC "+
          json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
if __name__=="__main__":
    main()
