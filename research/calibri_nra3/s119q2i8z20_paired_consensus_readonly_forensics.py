"""S119q2i8z20: READ-ONLY actual paired error/complementarity forensic of Z19.

Consumes *only* already completed physical original43M Z19 JSON artifact;
does not reload original43M, execute model, train, update, or change
hyperparameters on spent synthetic DEV14. Reports whether native h320
FINAL vs L3/L7/L11/final fixed equal mix are different predictions
and where they help/hurt on EXACT same 14 previously spent DEV samples.

This is NOT new independent DEV, a new model, or proof of semantics.
POSTHOC "either correct" is ORACLE UPPER BOUND, NEVER a deployable rule.
No 616 J7/EVAL128/GOLD, bpp, CB59 CPU, LIVE, Render or Supabase.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

from research.calibri_nra3.s119q2i8z18_fresh28train14dev_multilayer_source import (
    BASELINE,ALTERNATE,STAGE as Z18_SOURCE_STAGE,verify,rows,
)
# Literal immutable stage/report IDs, so this read-only forensic does
# NOT import torch, transformers, GitHub artifact model loader or original43M.
PHYSICAL_STAGE="CALIBRI_S119Q2I8Z19_ORIGINAL43M_FIXED_MULTILAYER_H320_VS_FINAL_RANK2_NATIVE64"
PHYSICAL_REPORT="S119Q2I8Z19_ORIGINAL43M_L3_L7_L11_FINAL_VS_LAST_28TRAIN14DEV_G0.json"

STAGE="CALIBRI_S119Q2I8Z20_ACTUAL_Z19_GOLD_POSTHOC_PAIRED_CONSENSUS_FORENSICS"
REPORT="S119Q2I8Z20_ACTUAL43M_FINAL_VS_MULTI_PAIRED_ERROR_CONSENSUS_G0.json"
ARMS=("Z10_FULLJOINT_EARLY23_X3_CE",
      "Z10_FULLJOINT_EARLY23_X3_CONTEXT_TOP8_BPE2")
MODES=(BASELINE,ALTERNATE)
KINDS=("sentence_formulation","case_agreement","agent_action_object",
       "time_causality","negation_scope","arithmetic_units","semantic_paraphrase")
EXPECTED_Z18_TRAIN_SHA="7fe5c8111c10108a2332b1e0c82a08c23797c66620e64e3eff530bb365f88cb4"
EXPECTED_Z18_DEV_SHA="089e55c5dbd3b3a759c17f52e37217c25bb2e39e10a5d9e75c38c4fbd606fd86"

def check(ok,why):
    if not ok:
        raise ValueError("Z20 fail-closed previous physical data-only: "+why)

def full_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def wilson95(successes,n):
    check(type(successes) is int and type(n) is int
          and n>0 and 0<=successes<=n,"Wilson binary size")
    z=1.95996398454
    p=successes/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z/den*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [round(max(0,center-half),6),
            round(min(1,center+half),6)]

def paired_exact_discordance_p(final_only,multi_only):
    """Two-sided exact paired sign/binomial p-value, exploratory small n."""
    n=final_only+multi_only
    if n==0:return 1.0
    k=min(final_only,multi_only)
    return round(min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n),8)

def metrics_for_arm(src,expected_sha):
    check(isinstance(src,dict)
          and isinstance(src.get("new_DEV14_per_source"),list)
          and len(src["new_DEV14_per_source"])==14,
          "genuine original43M Z19 14 paired posthoc rows")
    items=src["new_DEV14_per_source"]
    check({x["source_id"] for x in items}==set(expected_sha)
          and all(x["prompt_SHA256"] and x["source_id"] in expected_sha
                  and type(x["own_FIRST_correct"]) is bool
                  and type(x["gold_SECOND_reachable_in_original_top64"]) is bool
                  and type(x["original_greedy_SECOND_correct"]) is bool
                  and x["gold_only_posthoc_never_in_candidate_or_ranker_selection"] is True
                  and x["new_Z18_DEV_now_SPENT_not_independent"] is True
                  and x["real_original43M_full_native_forwards"]==2
                  and set(x["choices"])==set(MODES)
                  for x in items),
          "exact locked identities/paired labels and no target contamination")
    by_domain=defaultdict(list)
    counts=Counter()
    original_ids=[]
    selected={mode:[] for mode in MODES}
    per_source=[]
    for x in items:
        a,b=x["choices"][BASELINE],x["choices"][ALTERNATE]
        check(x["kind"] in KINDS and expected_sha[x["source_id"]]==x["kind"]
              and all(type(z["correct"]) is bool and
                      type(z["changed_original_greedy"]) is bool and
                      type(z["premature_EOS"]) is bool and
                      type(z["second_native_id"]) is int and
                      0<=z["second_native_id"]<49152
                      and z["changed_original_greedy"]==
                          (z["second_native_id"]!=x["original_greedy_SECOND_id"])
                      and z["premature_EOS"]==(z["second_native_id"]==2)
                      for z in (a,b)),
              "TRUE matched native BPE source and correct choice consistency")
        f=bool(a["correct"])
        m=bool(b["correct"])
        available=x["gold_SECOND_reachable_in_original_top64"]
        check(not(f or m) or available,
              "no correct BPE2 absent from original shortlist")
        tag=("both_correct" if f and m else
             "final_only_correct" if f else
             "multilayer_only_correct" if m else "neither_correct")
        same=a["second_native_id"]==b["second_native_id"]
        if same:check(f==m,"same native ID cannot have different gold result")
        counts[tag]+=1
        counts["agreed_choice" if same else "disagreed_choice"]+=1
        if same:
            counts["agreed_correct" if f else "agreed_incorrect"]+=1
        counts["own_first_correct"]+=int(x["own_FIRST_correct"])
        counts["human_gold_in_original_top64"]+=int(available)
        counts["original_native_greedy_BPE2_correct"]+=int(
            x["original_greedy_SECOND_correct"])
        if not available:
            counts["human_gold_missing_top64"]+=1
        if available and not(f or m):
            counts["gold_accessible_but_both_fail"]+=1
        if x["own_FIRST_correct"] is False:
            check(not (f or m or available or x["original_greedy_SECOND_correct"]),
                  "first own wrong => no correct second continuation")
        for mode,z in ((BASELINE,a),(ALTERNATE,b)):
            counts[mode+"_correct"]+=int(z["correct"])
            counts[mode+"_changed_original_greedy"]+=int(z["changed_original_greedy"])
            counts[mode+"_EOS"]+=int(z["premature_EOS"])
            selected[mode].append(z["second_native_id"])
        original_ids.append(x["original_greedy_SECOND_id"])
        by_domain[x["kind"]].append((f,m,available))
        per_source.append({
            "source_id":x["source_id"],"kind":x["kind"],
            "own_FIRST_correct":x["own_FIRST_correct"],
            "gold_in_original_top64":available,
            "greedy_BPE2_correct":x["original_greedy_SECOND_correct"],
            "FINAL_correct":f,"MULTI_correct":m,
            "prediction_agreement":same,
            "paired_outcome":tag,
            "POSTHOC_labels_only":True,
        })
    check(set(by_domain)==set(KINDS)
          and all(len(by_domain[k])==2 for k in KINDS)
          and len(per_source)==14
          and sum(counts[k] for k in (
              "both_correct","final_only_correct",
              "multilayer_only_correct","neither_correct"))==14
          and counts[BASELINE+"_correct"]==
              counts["both_correct"]+counts["final_only_correct"]
          and counts[ALTERNATE+"_correct"]==
              counts["both_correct"]+counts["multilayer_only_correct"],
          "exact same 14 paired source consistency")
    check(src["new_DEV14_after_training_once"]["own_FIRST_correct"]==
          counts["own_first_correct"]
          and src["new_DEV14_after_training_once"][
              "gold_SECOND_in_original_top64"]==
              counts["human_gold_in_original_top64"]
          and src["new_DEV14_after_training_once"][
              "original_greedy_SECOND_correct"]==
              counts["original_native_greedy_BPE2_correct"]
          and all(src["new_DEV14_after_training_once"][
              "same_source_matched_modes"][mode]["correct_SECOND"]==
              counts[mode+"_correct"] for mode in MODES),
          "Z19 completed official original full decoder scores cross-check")
    unique={m:len(set(ids)) for m,ids in selected.items()}
    concentration={m:{
        "unique_native_BPE2_ids":unique[m],
        "largest_single_native_BPE2_count":max(Counter(ids).values()),
        "largest_single_token_share":round(max(Counter(ids).values())/14,5),
        "simpson_concentration":round(sum((v/14)**2
                              for v in Counter(ids).values()),6),
    } for m,ids in selected.items()}
    result={
        "source_n":14,"counts":dict(counts),
        "final_only_second_correct":counts[BASELINE+"_correct"],
        "multilayer_second_correct":counts[ALTERNATE+"_correct"],
        "paired_delta_multilayer_minus_final_correct":
            counts["multilayer_only_correct"]-counts["final_only_correct"],
        "paired_exact_two_sided_discordance_p_NOT_validation":
            paired_exact_discordance_p(
                counts["final_only_correct"],counts["multilayer_only_correct"]),
        "Wilson95_exploratory_final":wilson95(counts[BASELINE+"_correct"],14),
        "Wilson95_exploratory_multilayer":wilson95(counts[ALTERNATE+"_correct"],14),
        "posthoc_ORACLE_any_shadow_correct_upper_bound_NOT_a_rule":
            counts["both_correct"]+counts["final_only_correct"]+
            counts["multilayer_only_correct"],
        "candidate_coverage_upper_bound_original_top64":
            counts["human_gold_in_original_top64"],
        "agreement_precision_POSTHOC_if_agree":(
            round(counts["agreed_correct"]/counts["agreed_choice"],6)
            if counts["agreed_choice"] else None),
        "selections_concentration":concentration,
        "by_domain_2_sources_each":{
            domain:{
                "final_correct":sum(a for a,b,present in groups),
                "multilayer_correct":sum(b for a,b,present in groups),
                "gold_second_accessible_original_top64":
                    sum(present for a,b,present in groups),
                "source_n":2,
            }
            for domain,groups in sorted(by_domain.items())
        },
        "per_source_POSTHOC_descriptive_only":per_source,
    }
    check(result["posthoc_ORACLE_any_shadow_correct_upper_bound_NOT_a_rule"]
          <=result["candidate_coverage_upper_bound_original_top64"],
          "non-deployable oracle impossible above original candidates")
    return result

def run(z19path,out):
    z19path=Path(z19path);out=Path(out)
    check(z19path.is_file() and z19path.name==PHYSICAL_REPORT,
          "actual completed Z19 physical JSON name only")
    check(not out.exists(),"no overwrite old research results")
    fresh=verify()
    check(fresh["stage"]==Z18_SOURCE_STAGE
          and fresh["TRAIN_SHA256"]==EXPECTED_Z18_TRAIN_SHA
          and fresh["DEV_SHA256"]==EXPECTED_Z18_DEV_SHA,
          "previously source-locked, now SPENT Z18 28TRAIN14DEV exact SHA")
    report=json.loads(z19path.read_text(encoding="utf8"))
    check(report["stage"]==PHYSICAL_STAGE
          and report["precommitted_Z18_new_TRAIN28_SHA256"]==EXPECTED_Z18_TRAIN_SHA
          and report["precommitted_Z18_new_DEV14_SHA256"]==EXPECTED_Z18_DEV_SHA
          and report["DEV14_NOW_SPENT_not_independently_blind"] is True
          and report["original43M_A_B_real_full_native_forwards"]==168
          and report["same_native49152_topK"]==64
          and report["same_per_shadow_trainable_params"]==1280
          and report["no_DEV_gradient_or_gold_in_candidate_selection"] is True
          and report["candidate_accepted"] is False
          and report["output_gate"]==0
          and set(report["TRUE_matched_A_B_SHADOW_FINAL_VS_MULTILAYER"])==set(ARMS),
          "MUST use real completed original43M Z19 A/B fixed matched evidence")
    source_ids={r["id"]:r["kind"] for r in rows("DEV")}
    results={
        arm:metrics_for_arm(
            report["TRUE_matched_A_B_SHADOW_FINAL_VS_MULTILAYER"][arm],
            source_ids)
        for arm in ARMS
    }
    all_ids=[r["source_id"] for r in results[ARMS[0]][
        "per_source_POSTHOC_descriptive_only"]]
    check(set(all_ids)==set(source_ids),"all original Z18 DEV unchanged")
    cross={}
    rows_by_arm={
        arm:{x["source_id"]:x for x in results[arm]["per_source_POSTHOC_descriptive_only"]}
        for arm in ARMS
    }
    for mode in ("FINAL_correct","MULTI_correct"):
        a,b=ARMS
        cross[mode]={
            "A_and_B_correct":sum(rows_by_arm[a][i][mode]
                                  and rows_by_arm[b][i][mode] for i in source_ids),
            "A_only_correct":sum(rows_by_arm[a][i][mode]
                                 and not rows_by_arm[b][i][mode] for i in source_ids),
            "B_only_correct":sum(not rows_by_arm[a][i][mode]
                                 and rows_by_arm[b][i][mode] for i in source_ids),
            "neither_correct":sum(not rows_by_arm[a][i][mode]
                                 and not rows_by_arm[b][i][mode] for i in source_ids),
        }
    output={
        "stage":STAGE,"physical_source_Z19_report_sha256":full_sha256(z19path),
        "source_Z19_physical_original43M_all12GT_done_NOT_reexecuted":True,
        "preregistered_Z18_TRAIN28_SHA256":EXPECTED_Z18_TRAIN_SHA,
        "already_SPENT_Z18_DEV14_SHA256":EXPECTED_Z18_DEV_SHA,
        "original_physical_model_forwards_in_Z19":168,
        "new_original43M_forwards_in_Z20":0,
        "new_training_or_fit_in_Z20":0,
        "A_B_paired_FORensics":results,
        "cross_A_B_source_overlap":cross,
        "posthoc_oracle_lookahead_requiring_human_gold_PROHIBITED_as_inference":True,
        "NO_new_independent_holdout":True,
        "no_DECODING_rules_selected_or_promoted_from_spent_data":True,
        "no_original_model_or_hidden_weights_modified":True,
        "no_616_J7_EVAL128_GOLD_bpp_CB59_CPU_certification":True,
        "no_MAIN_LIVE_Render_Supabase_APK_Jessica_changes":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"SPENT_14_PAIRED_FORENSIC_ONLY_DO_NOT_TUNE_OR_DEPLOY",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(json.dumps(output,ensure_ascii=False,
                                     indent=2,sort_keys=True)+"\n",encoding="utf8")
    compact={
        arm:{
            "paired_counts":results[arm]["counts"],
            "second_correct_FINAL":results[arm]["final_only_second_correct"],
            "second_correct_MULTILAYER":results[arm]["multilayer_second_correct"],
            "paired_delta":results[arm]["paired_delta_multilayer_minus_final_correct"],
            "both_correct_oracle":results[arm][
                "posthoc_ORACLE_any_shadow_correct_upper_bound_NOT_a_rule"],
            "original_gold_in_top64":results[arm][
                "candidate_coverage_upper_bound_original_top64"],
            "paired_p":results[arm]["paired_exact_two_sided_discordance_p_NOT_validation"],
            "consensus_agreement_precision":results[arm][
                "agreement_precision_POSTHOC_if_agree"],
            "diversity":results[arm]["selections_concentration"],
        }
        for arm in ARMS
    }
    print("S119Q2I8Z20_COMPLETED_ACTUAL_Z19_PAIRED_CONSENSUS_FORENSICS "+
          json.dumps({"A_B":compact,"cross_A_B":cross,
                      "new_actual43M_forwards":0,
                      "candidate_accepted":False,"output_gate":0},
                     ensure_ascii=False,sort_keys=True),flush=True)
    return output

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--z19-report",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    x=parser.parse_args()
    run(x.z19_report,x.out)
