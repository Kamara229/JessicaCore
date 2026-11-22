"""S119q2i8z1: exact paired source-by-source original 43M 12GT causal audit.

NO NEW forward / training: load already SUCCESSFUL physical 336
original S142 43M native49k forensic Z results, compare same 14
ALREADY SPENT synthetic X DEV source across 3 physically saved
neural FP16 trained W/X states × 8 reversible original GT12, native
rank4, CORE/NRA3/NPA1 and recurrence toggles. Count paired native
gold-second NLL and rank change directions and seven domain stability.
Descriptive small seen synthetic set, NOT independent blind, no
preference tuning or candidate promotion. No model gradients or weights.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from research.calibri_nra3.s119q2i8z_context_ablation_contract import (
    STAGE as Z_STAGE,CONDITIONS,MODELS,FULL,
    EXPECTED_REAL_ORIGINAL_43M_FORWARDS,
)
from research.calibri_nra3.s119q2i8x_new_ru_train_dev_source import (
    verify as x_source_verify,rows as x_rows,
)
STAGE="CALIBRI_S119Q2I8Z1_ACTUAL_ORIGINAL43M_X_SAME_SOURCE_PAIRED_CAUSAL_NLL_SIGNS"
SOURCE_REPORT="S119Q2I8Z_TRUE43M_GOLD_SECOND_NATIVE49K_CONTEXT_CAUSAL_336_PROBES_G0.json"
REPORT="S119Q2I8Z1_ORIGINAL43M_SAME_SOURCE_14DEV_CAUSAL_PAIRED_NLL_SIGN_COUNTS_G0.json"
EPS=1e-6

def check(ok,msg):
    if not ok:raise ValueError("S119q2i8z1 fail-closed: "+msg)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def paired_stats(report):
    src=x_source_verify()
    check(report["stage"]==Z_STAGE
          and report["candidate_accepted"] is False
          and report["output_gate"]==0
          and report["real_original43M_full_native49152_gold_second_prefix_actual_physical_forwards"]==
              EXPECTED_REAL_ORIGINAL_43M_FORWARDS==336
          and report["original_frozen43M_and_all_three_research_FP16_model_and_GT12_weights_preserved"] is True
          and report["already_spent_X14_author_internal_DEV_gold_PREVIOUS_BPE_only"] is True
          and report["new_X14_now_SPENT_source_SHA256"]==src["X14_DEV_sha256"]
          and report["registered_8_component_and_native_readout_conditions"]==list(CONDITIONS),
          "immutable valid SHA-identified completed original43M actual Z 336 native49k")
    rows=report["per_source_14x3x8_actual_original43M_logit_rank_and_ce_records"]
    check(len(rows)==336,"same exact three FP16 physical original43M ×14 spent author DEV×8 registered")
    xdev=x_rows("DEV")
    domains={r["id"]:r["kind"] for r in xdev}
    check(len(domains)==14 and len(set(domains.values()))==7,
          "exact previously scored seven original author categories of 14 synthetic prompts")
    out={}
    for state in MODELS:
        selected=[r for r in rows if r["model"]==state]
        check(len(selected)==112,"one full trained child ×14×8 physical")
        perid=defaultdict(dict)
        for r in selected:
            check(r["source_id"] in domains
                  and r["source_is_ALREADY_SPENT_synthetic_DEV"] is True
                  and r["teacher_forced_PREVIOUS_gold_BPE_not_own"] is True
                  and r["condition"] in CONDITIONS
                  and r["gold_BPE_target_native_id"] not in (2,None),
                  "only already used original43M author X gold-second native teacherforced")
            check(r["condition"] not in perid[r["source_id"]],
                  "no double-counted replay to inflate 14 synthetic cases")
            perid[r["source_id"]][r["condition"]]=r
        check(set(perid)==set(domains) and
              all(set(items)==set(CONDITIONS) for items in perid.values()),
              "exact each same X source compared to all 8 toggles and same weights")
        percondition={}
        for condition in CONDITIONS:
            pairs=[];improve=worsen=same=0
            for source_id,items in perid.items():
                base=items[FULL];other=items[condition]
                delta=float(other["gold_49152_BPE_NLL"])-float(
                    base["gold_49152_BPE_NLL"])
                rank_diff=other["gold_49152_BPE_rank"]-base["gold_49152_BPE_rank"]
                check(abs(delta)<40 and abs(rank_diff)<=49151,
                      "finite FULL vs one strictly reversible 43M native49k toggle")
                improve+=delta<-EPS
                worsen+=delta>EPS
                same+=abs(delta)<=EPS
                pairs.append({
                    "source_id":source_id,"domain":domains[source_id],
                    "NLL_ablation_minus_FULL_nats":round(delta,7),
                    "gold_native_rank_ablation_minus_FULL":rank_diff,
                    "native_winner_CHANGED":other["predicted_BPE_id"]!=base["predicted_BPE_id"],
                    "gold_second_top5_FULL":bool(base["gold_top5"]),
                    "gold_second_top5_ablated":bool(other["gold_top5"]),
                })
            diffs=[x["NLL_ablation_minus_FULL_nats"] for x in pairs]
            kinds={}
            for kind in sorted(set(domains.values())):
                ks=[p["NLL_ablation_minus_FULL_nats"] for p in pairs if p["domain"]==kind]
                check(len(ks)==2,"exact 7 domains×2 each")
                kinds[kind]={
                    "2_already_spent_source_count":2,
                    "mean_ablation_minus_FULL_gold_second_NLL_nats":round(
                        statistics.mean(ks),7),
                    "n_ablation_improves_vs_FULL":sum(x<-EPS for x in ks),
                    "n_ablation_harms_vs_FULL":sum(x>EPS for x in ks),
                }
            check(improve+worsen+same==14,
                  "paired within-source signs must exactly partition 14")
            percondition[condition]={
                "same_original43M_trained_FP16_state":state,
                "n_already_scored_author_synthetic_DEV":14,
                "ablation_NLL_improves_lower_better_vs_FULL":improve,
                "ablation_NLL_harms_higher_worse_vs_FULL":worsen,
                "ablation_NLL_effectively_unchanged_vs_FULL":same,
                "mean_paired_gold_second_NLL_ablation_minus_FULL_nats":
                    round(statistics.mean(diffs),7),
                "median_paired_gold_second_NLL_ablation_minus_FULL_nats":
                    round(statistics.median(diffs),7),
                "same_source_native_BPE_top1_changes":sum(
                    p["native_winner_CHANGED"] for p in pairs),
                "seven_original_author_category_paired_NLL_signs":kinds,
                "per_X14_spent_source_paired_loss_deltas":pairs,
                "NOT_independent_blind_or_model_selection":True,
            }
        out[state]=percondition
    return {
        "stage":STAGE,
        "source_true_original43M_Z_stage":Z_STAGE,
        "source_X14_now_SPENT_sha256":src["X14_DEV_sha256"],
        "reused_original43M_full_native49152_physical_forwards":336,
        "new_actual_original43M_model_forwards_this_paired_analysis":0,
        "three_real_physical_421793_FP16_trained_state_and_12GT_model_paired_ablation_signs":out,
        "gold_previous_BPE_source_only_NOT_real_OWN_generation":True,
        "no_training_gradient_optimizer_or_tuning_from_DEV":True,
        "original_616J7_EVAL128_GOLD_bpp_CPU_not_scored":True,
        "candidate_accepted":False,
        "output_gate":0,
        "decision":"DESCRIPTIVE_PAIRED_NLL_ABLATION_SIGN_RESEARCH_ONLY_NO_PROMOTION",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--z-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    file=a.z_dir/SOURCE_REPORT
    check(file.is_file() and not a.out.exists(),"completed Z read-only report only")
    z=json.loads(file.read_text(encoding="utf8"))
    value=paired_stats(z)
    value["completed_z_336_original43M_REPORT_SHA256"]=sha(file)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
                      encoding="utf8")
    print("S119Q2I8Z1_336_ORIGINAL43M_PAIRED_CAUSAL_NLL "+json.dumps({
        state:{name:{
            "paired_improve":d["ablation_NLL_improves_lower_better_vs_FULL"],
            "paired_harm":d["ablation_NLL_harms_higher_worse_vs_FULL"],
            "paired_tie":d["ablation_NLL_effectively_unchanged_vs_FULL"],
            "mean_NLL_ablation_minus_full":d[
                "mean_paired_gold_second_NLL_ablation_minus_FULL_nats"],
            "median_NLL_ablation_minus_full":d[
                "median_paired_gold_second_NLL_ablation_minus_FULL_nats"]
        } for name,d in v.items()} for state,v in value[
            "three_real_physical_421793_FP16_trained_state_and_12GT_model_paired_ablation_signs"
        ].items()},ensure_ascii=False,sort_keys=True),flush=True)
if __name__=="__main__":
    main()
