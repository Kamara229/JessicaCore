"""Z42 immutable archival audit of physically completed Z41 S142 43M BPE2–4.

No model weights, no training and no new model forward. The input is solely
archived GitHub Actions Z41 report JSON on *already spent* author DEV14.
Distinguishes per-PROMPT conditional entropy trust region (probabilities)
from across-PROMPT empirical diversity of sampled/greedy token IDs, and
reports whether interventions actually flipped original native decisions.
Do NOT use spent DEV for architecture tuning or promotion.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from math import comb, log2
from pathlib import Path

STAGE="CALIBRI_S119Q2I8Z41_REAL43M_NATIVE49152_CAUSAL_BPE234_ENTROPY_GUARDED_84TRAIN14DEV"
SHA="5891c8122853e39a5441c63da91bfa70138c7941c027331992e5d8b4e09f4fdb"
BASE="ORIGINAL_NATIVE_GREEDY"
TRUE="HUMAN_GOLD_BPE234"
WRONG="WRONG_OFFSET17_BPE234"
MODES=(BASE,TRUE,WRONG)
ARMS=("Z10_FULLJOINT_EARLY23_X3_CE","Z10_FULLJOINT_EARLY23_X3_CONTEXT_TOP8_BPE2")
ALPHAS=(1.,.75,.5,.375,.25,.125,.0625,.03125,0.)
POSITIONS=(2,3,4)
ENTROPY_GUARD_FLOOR=-.05

def require(ok,why):
    if not ok:
        raise ValueError("Z42 immutable Z41 physical report fail-closed: "+why)

def empirical(ids):
    """Frequency entropy across prompts, NOT per-prompt model distribution."""
    if not ids:
        return {"count":0,"distinct":0,"empirical_entropy_bits":None,
                "largest_token_fraction":None}
    c=Counter(ids)
    n=len(ids)
    return {"count":n,"distinct":len(c),
            "empirical_entropy_bits":round(-sum((v/n)*log2(v/n) for v in c.values()),6),
            "largest_token_fraction":round(max(c.values())/n,6)}

def sign_p(up,down):
    require(type(up) is int and type(down) is int and up>=0 and down>=0,
            "paired nonnegative discordance")
    n=up+down
    return min(1.,2.*sum(comb(n,k) for k in range(min(up,down)+1))/2**n) if n else 1.

def inspect_arm(arm, label):
    rows=arm["NEW_DEV14_per_prompt_first6"]
    require(len(rows)==14 and len({r["source_id"] for r in rows})==14 and
            len({r["prompt_SHA256"] for r in rows})==14, "14 unique SPENT DEV prompts")
    summary=arm["NEW_DEV14_full_autonomous_first6_native"]
    require(summary["original43M_real_native_TRAIN_forwards"]==252 and
            summary["brand_NEW_synthetic_DEV14"]==14 and
            summary["no_DEV_teacher_forcing"] is True and
            summary["only_native_BPE2_BPE3_BPE4_can_be_reranked"] is True and
            summary["native_BPE5_BPE6_unchanged_ON_policy"] is True and
            summary["targetblind_native_entropy_and_KL_guard"] is True,
            "one previous physical run, no dev teacher forcing")
    totals={m:{
        "correct_first1":0,"correct_first2":0,"correct_first3":0,
        "correct_first4":0,"correct_first6":0,"early_EOS_first4":0,
        "first2_correct_but_BPE3_wrong":0,
        "token_ids_by_position":{2:[],3:[],4:[]},
        "steps_by_position":{p:{"attempts":0,"alpha_nonzero":0,"alpha_zero":0,
            "changed_original_greedy":0,"entropy_delta_bits":[],"alphas":[]} for p in POSITIONS}
    } for m in MODES}
    pairs=[]
    actual_forwards=0
    for row in rows:
        require(row["DEV_gold_was_never_in_training_or_candidate_selection"] is True and
                row["NEW_Z40_DEV_NOW_SPENT_internal_synthetic"] is True,
                "SPENT DEV never trained and source integrity")
        gold=row["human_gold_native_first4_POSTHOC"]
        require(len(gold)==4 and all(type(x) is int for x in gold),"POSTHOC first4 golden IDs")
        modes=row["auto_generated_modes_no_teacher_forcing"]
        require(set(modes)==set(MODES),"all three same-source real neural modes")
        first=set()
        paired={}
        counts=0
        for mode in MODES:
            d=totals[mode]
            rec=modes[mode]
            ids=rec["autonomous_first_six_native_ids"]
            require(1<=len(ids)<=6 and all(type(x) is int for x in ids),"real own native first6")
            first.add(ids[0])
            steps=rec["per_second_to_fourth_real_forward"]
            require(len(steps)==len(ids)-1 and rec["true_full_S142_native_forwards"]==len(ids),
                    "real physical forward count / step parity")
            counts+=len(steps)
            correct={k:len(ids)>=k and ids[:k]==gold[:k] for k in (1,2,3,4)}
            correct[6]=bool(rec["first_six_native_correct"])
            for k in (1,2,3,4):
                require(rec[{1:"first_native_correct",2:"first_two_native_correct",
                             3:"first_three_native_correct",4:"first_four_native_correct"}[k]]==correct[k],
                        "native exact-prefix archived score consistency")
            require(not correct[6] or correct[4],"six correct implies four correct")
            for k in (1,2,3,4,6):
                d["correct_first"+str(k)]+=int(correct[k])
            d["early_EOS_first4"]+=int(rec["did_emit_EOS_early"])
            d["first2_correct_but_BPE3_wrong"]+=int(correct[2] and not correct[3])
            paired[mode]={2:correct[2],3:correct[3],4:correct[4]}
            for pos in POSITIONS:
                if len(ids)>=pos:
                    d["token_ids_by_position"][pos].append(ids[pos-1])
            for idx,step in enumerate(steps,start=2):
                require(step["native_position_1based"]==idx and
                        step["chosen_native_id"]==ids[idx-1] and
                        step["chose_EOS"]==(ids[idx-1]==2),
                        "original native self-prefix actual chosen token")
                changed=(step["chosen_native_id"]!=step["original_greedy_native_id"])
                require(step["shadow_changed_original_greedy"]==changed,
                        "actual changed-vs-greedy parity")
                alpha=step["targetblind_entropy_guard_alpha"]
                delta=step["targetblind_entropy_vs_original_bits"]
                if mode==BASE or idx>4:
                    require(not changed and alpha==0 and delta==0,
                            "unmodified original native greedy outside BPE2..4")
                else:
                    require(step["candidate_original_top64"] is True and
                            any(abs(alpha-a)<1e-8 for a in ALPHAS) and
                            delta>=ENTROPY_GUARD_FLOOR-0.000001,
                            "targetblind top64 entropy guard no gold injection")
                    require(not(alpha==0 and changed),
                            "zero-alpha guard must preserve original greedy")
                if idx in POSITIONS:
                    z=d["steps_by_position"][idx]
                    z["attempts"]+=1
                    z["changed_original_greedy"]+=int(changed)
                    z["alphas"].append(float(alpha))
                    z["entropy_delta_bits"].append(float(delta))
                    z["alpha_nonzero"]+=int(alpha>0)
                    z["alpha_zero"]+=int(alpha==0)
        require(len(first)==1,"same original unmodified native first BPE")
        require(row["actual_original43M_full_native_forwards"]==1+counts and
                1<=1+counts<=16,"count original first BPE only once across A/B modes")
        actual_forwards+=1+counts
        pairs.append(paired)
    score=summary["matched_autonomous_first6_native_generated"]
    output={}
    for mode in MODES:
        d=totals[mode]
        require(score[mode]["correct_FIRST"]==d["correct_first1"] and
                score[mode]["correct_FIRST_TWO"]==d["correct_first2"] and
                score[mode]["correct_FIRST_THREE"]==d["correct_first3"] and
                score[mode]["correct_FIRST_FOUR"]==d["correct_first4"] and
                score[mode]["correct_FIRST_SIX"]==d["correct_first6"] and
                score[mode]["premature_EOS_in_first4"]==d["early_EOS_first4"],
                "archived physical first1-6 summary matches per-prompt replay")
        ids=d.pop("token_ids_by_position")
        d["empirical_cross_prompt_native_ID_diversity"]={
            str(pos):empirical(ids[pos]) for pos in POSITIONS}
        require(d["empirical_cross_prompt_native_ID_diversity"]["2"]["distinct"]==
                score[mode]["distinct_second_native_BPE_ids"],
                "native BPE2 diversity parity physical")
        stats=d.pop("steps_by_position")
        d["per_position_actual_native_interventions"]={
            str(pos):{**{k:v for k,v in data.items() if k not in
                       ("alphas","entropy_delta_bits")},
                      "mean_guard_alpha":round(sum(data["alphas"])/len(data["alphas"]),6)
                        if data["alphas"] else None,
                      "mean_conditional_top64_entropy_delta_bits":round(
                          sum(data["entropy_delta_bits"])/len(data["entropy_delta_bits"]),6)
                          if data["entropy_delta_bits"] else None,
                      "minimum_conditional_top64_entropy_delta_bits":round(
                          min(data["entropy_delta_bits"]),6)
                          if data["entropy_delta_bits"] else None}
            for pos,data in stats.items()}
    for mode in (TRUE,WRONG):
        p={}
        for length in (2,3,4):
            up=sum(int(rec[mode][length] and not rec[BASE][length]) for rec in pairs)
            down=sum(int(rec[BASE][length] and not rec[mode][length]) for rec in pairs)
            p["first"+str(length)]={"new_correct":up,"became_wrong":down,
                 "exact_p_two_sided_descriptive_not_significance":sign_p(up,down)}
        output[mode]["paired_vs_original"]=p
        original_second=output[BASE]["empirical_cross_prompt_native_ID_diversity"]["2"]["distinct"]
        second=output[mode]["empirical_cross_prompt_native_ID_diversity"]["2"]["distinct"]
        output[mode]["BPE2_diversity_fraction_vs_baseline"]=round(second/original_second,6)
    require(actual_forwards==summary["original43M_real_native_DEV_forwards"],
            "actual decoder forward sum parity")
    return {"arm":label,"real_original43M_native_DEV_forwards":actual_forwards,
            "scores":output}

def audit(report):
    require(report["stage"]==STAGE and report["output_gate"]==0 and
            report["candidate_accepted"] is False and
            report["Z40_PRECOMMITTED_14_BRANDNEW_DEV_SHA256"]==SHA and
            report["Z40_brandNEW_author_DEV14_NOW_SPENT_not_independently_blind"] is True,
            "pinned original physical Z41 gate0 source SHA, SPENT DEV")
    for key in ("strict_ONLY_native_BPE2_3_4_candidate_decoding_intervention",
       "targetblind_no_gold_entropy_and_KL_guard",
       "legal84_TRAIN_only_gold_prefix_native_BPE2_3_4",
       "original_S142_43M_GT12_Core_NRA3_NPA1_and_Z10_A_B_unchanged",
       "no_gold_in_candidate_selection_on_new_DEV",
       "no_previous_SPENT_DEV_used_in_TRAIN",
       "no_616J7_EVAL128_GOLD_bpp_CB59_CPU_certification",
       "no_MAIN_LIVE_Render_Supabase_APK_Jessica_changes"):
        require(report[key] is True,key)
    require(report["original43M_49k_native_FORWARDS"]==913,
            "source native forwards physical 913")
    arms=report["actual_original43M_A_B_real_native49k_BPE2_only_rank32"]
    require(set(arms)==set(ARMS),"both true original S142 prior A and B frozen physical")
    result={name:inspect_arm(arms[name],name) for name in ARMS}
    require(sum(r["real_original43M_native_DEV_forwards"]+252 for r in result.values())==913,
            "two arms 913 frozen model original decoder pass parity")
    return {
      "stage":"S119Q2I8Z42_READONLY_Z41_SPENT14_ENTROPY_CAUSAL_FAILURE_AUDIT",
      "physical_original43M_decoder_forwards_in_immutable_Z41":913,
      "new_original43M_decoder_forwards_or_optimizer_steps":0,
      "physical_previous_author_DEV_SHA256_now_spent":SHA,
      "not_new_benchmark_and_not_independent_blind":True,
      "original_with_wronglabel_and_true_train_controls":result,
      "warning":"cross_prompt_empirical_ID_entropy != per_prompt_next_token_entropy",
      "training_prefix":"gold teacher-prefix on 84 legal TRAIN only",
      "DEV_prefix":"on-policy own prefix only",
      "candidate_accepted":False,"output_gate":0,
      "decision":"NO_PROMOTION; NO_REUSE_SPENT_DEV_FOR_HYPERPARAMS; TEST_TRAIN_OWN_PREFIX_COVERAGE_AND_CAUSAL_BPE3_PLUS_WITH_NEW_PREREG"
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    arg=p.parse_args()
    report=json.loads(arg.input.read_text(encoding="utf8"))
    result=audit(report)
    arg.out.parent.mkdir(parents=True,exist_ok=True)
    arg.out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                       encoding="utf8")
    print(json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
