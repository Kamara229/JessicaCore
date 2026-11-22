"""S119q2i8v: first wrong continuation on REAL 43M native 49152 BPE.

Source-only posthoc diagnostic of successful completed #957 read-only 84
physical original S142 43M / 12GT K vs T(CE vs relative EOSmargin),
14 already scored/seen small synthetic T author DEV x second+third human
gold BPE each; never misrepresent teacher-forced previous gold BPE
as own-native language generation.

Categorize whether each wrong top1 native token is REAL EOS (native ID2)
or OTHER incorrect 49k BPE. This isolates whether poor continuation
is simply premature EOS or also lack of correct nonEOS distinctions.
No new model weights, gradients, logits reruns, teacher calls, hyperparam
tuning, blind test, promotion, 616J7/EVAL128, production.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

STAGE="CALIBRI_S119Q2I8V_TRUE43M_49K_NATIVE_EOS_VS_WRONG_OTHER_CONTINUATION_FORENSIC"
U_STAGE="CALIBRI_S119Q2I8U_READONLY_TRUE43M_K_VS_T_FULLJOINT_CE_EOS_NATIVE_BPE2BPE3_RANK"
T_STAGE="CALIBRI_S119Q2I8T_REAL43M_FULL_GT12_CORE_NRA3_NPA1_NATIVEHEAD_CE_VS_EOSMARGIN_G025"
U_REPORT="S119Q2I8U_TRUE43M_T_CE_VS_EOSMARGIN_GOLD_2ND3RD_BPE_NATIVE49K_G0.json"
T_REPORT="S119Q2I8T_REAL43M_FULL_JOINT_27TRAIN_CE_VS_EOS_MARGIN_G025.json"
VOCAB=49152
EOS=2
POSITIONS=(1,2)
LABELS=("K_FROZEN_EARLY23_KL_43M",
        "T_FULLJOINT_CE_43M",
        "T_FULLJOINT_CE_PLUS_EOS_RELATIVE_43M")
ARMS=(
    "K_EARLY23_KL_EXISTING_GAIN_0p25",
    "FULL_JOINT_CE_GAIN_0p25",
    "FULL_JOINT_CE_PLUS_EOS_RELATIVE_GAIN_0p25",
)

def check(ok,msg):
    if not ok:
        raise ValueError("S119q2i8v STRICT FAIL-CLOSED: "+str(msg))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def choice_category(record):
    gold=record["gold_native_id"]
    guess=record["greedy_original_native_id"]
    check(type(gold) is int and type(guess) is int
          and 0<=gold<VOCAB and 0<=guess<VOCAB
          and gold!=EOS,
          "human second/third gold native BPE, not terminal EOS")
    if guess==gold: return "CORRECT_NONTERMINAL_GOLD"
    if guess==EOS: return "WRONG_PREMATURE_EOS"
    return "WRONG_OTHER_NON_EOS_NATIVE_BPE"

def analyze(u,t):
    check(u["stage"]==U_STAGE and t["stage"]==T_STAGE
          and u["candidate_accepted"] is False
          and t["candidate_accepted"] is False
          and u["output_gate"]==t["output_gate"]==0
          and u["true_original43M_full_GT12_native49152_forwards"]==84
          and u["true_source135M_teacher_calls"]==0
          and u["teacher_forced_gold_PREVIOUS_tokens_ONLY_diagnostic_NOT_own"] is True
          and u["no_training_or_backward_or_optimizer_or_promotion"] is True
          and t["matched_27TRAIN_SOURCE_ONLY_and_14DEV_NO_gradient"] is True
          and u["T_14_SYNTHETIC_ALREADY_SPENT_DEV_source_sha256"]==
              t["new_T_14_DEV_SHA256"],
          "SHA-matched T 27TRAIN+14DEV and U REAL43M 84 forward posthoc gate0")
    scored=t["new_14_INTERNAL_DEV_and_old6_spent_OWN_native_BPE_by_arm"]
    check(all(a in scored for a in ARMS),"three matched original T OWN baseline and A/B")
    for arm in ARMS:
        own=scored[arm]["NEW_DEV"]["scores"]
        check(own["DEV_n"]==14
              and own["correct_first_native_BPE"]==14
              and own["first_correct_but_second_wrong"]==14
              and own["correct_full_native_ANSWER_and_EOS"]==0,
              "per-T actual own14 first correct then all wrong second, NOT rank proxy")
    rows=u["per_sample_gold_prefix_only_SECOND_THIRD_native_BPE_posthoc"]
    check(len(rows)==28,"exact 14 observed T prompts * 2 full-native gold BPE positions")
    by_pos={}
    raw=[]
    for pos in POSITIONS:
        group=[r for r in rows if r["native_gold_answer_index_0based"]==pos]
        check(len(group)==14 and len({r["source_id"] for r in group})==14,
              "exact 14 unique previously scored T synthetic author prompts")
        per_label={a:[] for a in LABELS}
        for r in group:
            check(r["teacher_forced_GOLD_PREFIX_DIAGNOSTIC_only"] is True
                  and r["is_sealed_or_independent_blind_eval"] is False
                  and r["native_gold_answer_index_0based"]==pos,
                  "full native gold PREVIOUS token only, no independent language claim")
            scores=r["three_source_original43m_native_results"]
            check(set(scores)==set(LABELS),"full native original K+T CE+T EOSmargin only")
            target=scores[LABELS[0]]["gold_native_id"]
            check(type(target) is int and target!=EOS,
                  "second and third human BPE target never EOS")
            datum={
                "source_id":r["source_id"],"domain":r["domain"],
                "native_answer_gold_BPE_position_1based":pos+1,
                "source_prompt_sha256":r["source_prompt_SHA256"],
                "gold_native_id":target,
                "source_is_spent_internal_synthetic_not_blind":True,
                "original43M_49k_native_gold_prefix_rank_inputs_ONLY":{},
            }
            for arm in LABELS:
                x=scores[arm]
                guess=x["greedy_original_native_id"]
                rank=x["correct_gold_native_rank_full49152"]
                top5=x["top5_native_ids"]
                check(x["gold_native_id"]==target
                      and type(guess) is int and 0<=guess<VOCAB
                      and type(rank) is int and 1<=rank<=VOCAB
                      and len(top5)==5 and len(set(top5))==5
                      and all(type(v) is int and 0<=v<VOCAB for v in top5)
                      and top5[0]==guess
                      and x["correct_gold_rank_top5"]==(rank<=5)
                      and x["gold_native_top1"]==(guess==target)
                      and math.isfinite(x["gold_log_probability_native_49152"]),
                      "genuine full49152 native distribution exact rank/top5/greedy")
                per_label[arm].append(x)
                datum["original43M_49k_native_gold_prefix_rank_inputs_ONLY"][arm]={
                    "actual_greedy_original_native_BPE_id":guess,
                    "gold_rank_of_49152":rank,
                    "wrong_top1_category":choice_category(x),
                    "gold_log_probability_nats":x["gold_log_probability_native_49152"],
                }
            raw.append(datum)
        stats={}
        for label in LABELS:
            xs=per_label[label]
            cats=Counter(choice_category(x) for x in xs)
            stats[label]={
                "spent_synthetic_source_count":14,
                "gold_top1_correct":cats["CORRECT_NONTERMINAL_GOLD"],
                "wrong_top1_premature_EOS":cats["WRONG_PREMATURE_EOS"],
                "wrong_top1_other_native_nonEOS":cats["WRONG_OTHER_NON_EOS_NATIVE_BPE"],
                "nonEOS_native_winners_unique_count":len(set(
                    x["greedy_original_native_id"] for x in xs
                    if x["greedy_original_native_id"]!=EOS)),
                "mean_full49152_gold_rank":round(statistics.mean(
                    x["correct_gold_native_rank_full49152"] for x in xs),5),
                "mean_full49152_gold_NLL_nats":round(-statistics.mean(
                    x["gold_log_probability_native_49152"] for x in xs),6),
            }
            check(sum(cats.values())==14 and
                  stats[label]["gold_top1_correct"]+
                  stats[label]["wrong_top1_premature_EOS"]+
                  stats[label]["wrong_top1_other_native_nonEOS"]==14,
                  "exact all 14 full original43M decoder native heads partitioned")
        a={r["source_id"]:r for r in raw if r["native_answer_gold_BPE_position_1based"]==pos+1}
        change=sum(v["original43M_49k_native_gold_prefix_rank_inputs_ONLY"][LABELS[1]][
                       "actual_greedy_original_native_BPE_id"]!=v[
                       "original43M_49k_native_gold_prefix_rank_inputs_ONLY"][LABELS[2]][
                       "actual_greedy_original_native_BPE_id"] for v in a.values())
        by_pos[f"gold_BPE_{pos+1}"]={
            "true_original43M_native49152_K_vs_TCE_vs_Tmargin":stats,
            "same_source_TCE_vs_Tmargin_greedy_top1_native_BPE_changed_count":change,
            "same_source_TCE_vs_Tmargin_greedy_top1_native_BPE_identical_count":14-change,
            "already_spent_T_DEV_not_independent_blind_or_MODEL_SELECTION":True,
        }
    check(len(raw)==28 and set(x["source_id"] for x in raw)==
          set(x["source_id"] for x in rows if x["native_gold_answer_index_0based"]==1),
          "all T prompts preserved across gold second and third BPE diagnostics")
    return {
        "stage":STAGE,
        "source_completed_physical_U_gold_BPE2BPE3_stage":U_STAGE,
        "source_completed_physical_T_fulljoint_stage":T_STAGE,
        "source_T_14_spent_author_DEV_sha256":t["new_T_14_DEV_SHA256"],
        "source_U_true43M_49k_native_forwards":84,
        "physical43M_forward_in_this_second_order_audit":0,
        "native_vocab":VOCAB,"native_EOS_ID":EOS,
        "genuine_NATIVE_second_and_third_gold_BPE_wrong_top1_EOS_vs_OTHER":by_pos,
        "all_14_spent_T_author_dev_per_sample_ids_NO_text":raw,
        "gold_previous_BPE_NOT_own_native_generation":True,
        "source_own_native14_first_correct_second_wrong_all_3_arms":True,
        "no_model_training_no_teacher_no_blind_or_model_selection":True,
        "no_616J7_EVAL128_GOLD_bpp_CPU_or_production":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"IDENTIFY_REAL_NATIVE_WRONG_TOP1_AFTER_FIRST_CORRECT_BPE_POSTHOC_ONLY",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--u-dir",type=Path,required=True)
    p.add_argument("--t-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    check(a.u_dir.is_dir() and a.t_dir.is_dir() and not a.out.exists(),
          "completed source-only artifacts no overwrite")
    u_path=a.u_dir/U_REPORT;t_path=a.t_dir/T_REPORT
    check(u_path.is_file() and t_path.is_file(),
          "completed physical U and T original43M reports not found")
    u=json.loads(u_path.read_text(encoding="utf8"))
    t=json.loads(t_path.read_text(encoding="utf8"))
    check(u["completed_T_fulljoint_actual_43M_REPORT_SHA256"]==sha(t_path),
          "actual U report SHA pins previous completed genuine T fulljoint 43M report")
    output=analyze(u,t)
    output["completed_source_U_report_SHA256"]=sha(u_path)
    output["completed_source_T_report_SHA256"]=sha(t_path)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,ensure_ascii=False,indent=2,
                                sort_keys=True)+"\n",encoding="utf8")
    print("S119Q2I8V_NATIVE49K_WRONG_SECOND_THIRD_BPE_EOS_VS_OTHER "+json.dumps({
        "per_gold_position":output["genuine_NATIVE_second_and_third_gold_BPE_wrong_top1_EOS_vs_OTHER"],
        "source_43M_forwards_reused_not_rerun":84,
        "candidate_accepted":False
    },ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
