"""S119q2i8z7 strict source-only causal wrong-gold-margin readout audit.

Reuses #969 proven 84 REAL original S142 43M 12GT original full native
49152 forwards, split learned raw head into contextual rank4 and static
native vocabulary bias. Counts signs of their contribution to the
actual FULL greedy winner-vs-human gold next SECOND/THIRD native token.
Gold PREVIOUS human context on 14 already spent synthetic Z2 author DEV.
This audit does NOT use new model passes, training or unscored DEV.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from research.calibri_nra3.s119q2i8z6_head_context_bias_contract import (
    STAGE as Z6_STAGE,STATES,POSITIONS,VARIANTS,
    EXPECTED_PHYSICAL,EXPECTED_ANALYTIC,summarize,
)
from research.calibri_nra3.s119q2i8z2_new_ru_train_dev_source import verify as source_verify

STAGE="CALIBRI_S119Q2I8Z7_ORIGINAL43M_NATIVE49152_FULL_WRONG_WINNER_CONTEXT_RANK4_VS_STATIC_BIAS"
SOURCE_REPORT="S119Q2I8Z6_TRUE43M_NATIVE49152_LEARNED_BIAS_VS_CONTEXT_RANK4_84REAL_336VIRTUAL_G0.json"
REPORT="S119Q2I8Z7_REAL_ORIGINAL43M_CONTEXT_RANK4_VS_BIAS_WRONG_GOLD_MARGIN_SIGNS_G0.json"
GAIN=.25

def check(ok,msg):
    if not ok: raise ValueError("S119q2i8z7 fail closed: "+msg)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def analyze(z6):
    source=source_verify()
    check(z6["stage"]==Z6_STAGE
          and z6["candidate_accepted"] is False
          and z6["output_gate"]==0
          and z6["source_Z2_14_author_DEV_NOW_SPENT_sha256"]==
              source["Z2_14_DEV_sha256"]
          and z6["physical_original_S142_43M_all12GT_native49152_full_forwards"]==
              EXPECTED_PHYSICAL==84
          and z6["analytical_original49k_same_true_hidden_and_same_source_virtual_variants"]==
              EXPECTED_ANALYTIC==336
          and z6["three_complete421793_original_neural_FP16_and_real12GT_fitted_SHA_verified"] is True
          and z6["exact_learned_native_rank4_plus_bias_decomposition_recomputed"] is True
          and z6["all_three_source_full_original43M_and_12GT_immutable"] is True
          and z6["same14_small_internal_synthetic_Z2_DEV_ALREADY_SPENT_not_blind"] is True
          and z6["gold_previous_human_second_third_BPE_NOT_real_OWN_autoregression"] is True
          and z6["native_output_gate_original_trained_gain"]==GAIN,
          "ONLY SHA-proven original S142 43M+12GT Z6 84 REAL, 336 analytic native49k")
    records=z6["all_336_FULL49152_original43M_actual_hidden_virtual_component_ranks"]
    check(summarize(records)==z6["same14_BPE2_BPE3_source_variants_full_native49152_NLL_rank"],
          "recompute entire original 336 same-source complete49152 readout stats")
    rows=z6["decomposition_actual_full49k_winner_gold_bias_rank4_logit_gaps"]
    check(len(rows)==EXPECTED_PHYSICAL,"exact 84 original real 43M physical winner-gold contrasts")
    groups={}
    for row in rows:
        check(row["state"] in STATES
              and row["position_native_zero_based"] in POSITIONS
              and row["source_PREVIOUS_GOLD_only_ALREADY_SPENT"] is True,
              "source must be true previously scored Z2 human GOLD previous BPE2/3")
        key=(row["state"],row["position_native_zero_based"],row["source_id"])
        check(key not in groups,"no duplicate same-source winner margin readout")
        groups[key]=row
        check(abs(row["true_FITTED_full_winner_minus_gold_logit"]-(
              row["pretrained_original_43M_winner_minus_gold_logit"]+
              GAIN*row["learned_vocab_global_BIAS_winner_minus_gold_raw"]+
              GAIN*row["learned_CONTEXT_rank4_winner_minus_gold_raw"]))<.00011
              and row["true_FITTED_full_winner_minus_gold_logit"]>=-0.0001,
              "exact original43M decomposition of actual full WINNER-minus-correct-gold logits")
    out={}
    for state in STATES:
        out[state]={}
        for pos in POSITIONS:
            rr=[r for key,r in groups.items() if key[:2]==(state,pos)]
            check(len(rr)==14 and len({r["source_id"] for r in rr})==14,
                  "exact 14 same previously scored synthetic Z2 author prompts")
            ctx=[GAIN*r["learned_CONTEXT_rank4_winner_minus_gold_raw"] for r in rr]
            bias=[GAIN*r["learned_vocab_global_BIAS_winner_minus_gold_raw"] for r in rr]
            base=[r["pretrained_original_43M_winner_minus_gold_logit"] for r in rr]
            full=[r["true_FITTED_full_winner_minus_gold_logit"] for r in rr]
            out[state][f"gold_previous_BPE_position_{pos+1}"]={
                "same14_now_SPENT_synthetic_authored_Z2":14,
                "actual_full_native49152_wrong_winner_over_gold_mean_margin":
                    round(statistics.mean(full),7),
                "mean_pretrained_43M_native_winner_minus_gold_margin":
                    round(statistics.mean(base),7),
                "mean_additive_CONTEXT_rank4_winner_minus_gold_margin":
                    round(statistics.mean(ctx),7),
                "mean_additive_GLOBAL_vocab_bias_winner_minus_gold_margin":
                    round(statistics.mean(bias),7),
                "learned_context_rank4_SHRINKS_wrong_winner_gold_gap":sum(x<-1e-6 for x in ctx),
                "learned_context_rank4_ENLARGES_wrong_winner_gold_gap":sum(x>1e-6 for x in ctx),
                "learned_global_vocab_bias_SHRINKS_wrong_winner_gold_gap":sum(x<-1e-6 for x in bias),
                "learned_global_vocab_bias_ENLARGES_wrong_winner_gold_gap":sum(x>1e-6 for x in bias),
                "already_current_wrong_gold_full_margin_zero_correct_native_greedy":sum(
                    x<1e-6 for x in full),
                "rank4_context_term_has_GREATER_absolute_gap_contribution_than_global_bias":sum(
                    abs(a)>abs(b) for a,b in zip(ctx,bias)),
                "per_synthetic_now_spent_source_margin_decomposition":[{
                    "source_id":r["source_id"],
                    "current_FITTED_native_greedy_id":r["true_full_native_winner_ID"],
                    "correct_next_native_gold_id":r["correct_native_gold_ID"],
                    "pretrained_base_winner_minus_gold_margin":round(v,7),
                    "learned_context_rank4_contribution_to_wrong_gold_margin":round(c,7),
                    "learned_global_bias_contribution_to_wrong_gold_margin":round(b,7),
                    "true_reconstructed_full_winner_minus_gold_margin":round(f,7),
                } for r,v,c,b,f in zip(rr,base,ctx,bias,full)],
            }
    check(len(groups)==EXPECTED_PHYSICAL and set(out)==set(STATES),
          "all source/position/physical model splits complete, never selective")
    return {
        "stage":STAGE,
        "reused_original_S142_43M_native49152_12GT_REAL_forward_calls":84,
        "new_actual_original43M_decoder_forwards_in_this_stage":0,
        "reused_virtual_full49152_native_component_rank_readouts":336,
        "full_native49k_original_fitted_head_bias_vs_context_wrong_winner_margin_signs":out,
        "source_Z2_14_SYNTHETIC_DEV_ALREADY_SPENT_sha256":source["Z2_14_DEV_sha256"],
        "only_teacherforced_correct_GOLD_PREVIOUS_BPE_not_OWN":True,
        "all_old_FP16_research_models_original43M_and_original_12GT_untouched":True,
        "no_training_gradient_optimizer_or_blind_test":True,
        "no_616J7_EVAL128_GOLD_bpp_CPU_TESTED":True,
        "no_main_LIVE_Render_Supabase_APK_Jessica_chat":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"DESCRIPTIVE_WRONG_TOKEN_OLD43M_CONTEXT_RANK4_VS_BIAS_SIGN_ONLY_NO_PROMOTION",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--z6-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    source=a.z6_dir/SOURCE_REPORT
    check(source.is_file() and not a.out.exists(),
          "genuine completed Z6 original43M research JSON only, no overwrites")
    obj=analyze(json.loads(source.read_text(encoding="utf8")))
    obj["source_84_real_43M_336_virtual_G0_SHA256"]=sha(source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(obj,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
                     encoding="utf8")
    print("S119Q2I8Z7_TRUE43M_CONTEXT_VS_BIAS_MARGIN_CAUSES "+json.dumps({
        state:{position:{k:v for k,v in data.items()
                         if k!="per_synthetic_now_spent_source_margin_decomposition"}
               for position,data in positions.items()}
        for state,positions in obj[
            "full_native49k_original_fitted_head_bias_vs_context_wrong_winner_margin_signs"
        ].items()},sort_keys=True,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
