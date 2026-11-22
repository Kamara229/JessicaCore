"""S119q2i8y: physical FP16 + paired SELF native 49k posthoc X audit.

READ-ONLY completed successful S119q2i8x report + saved full 421793 FP16
trained Core/GT12/NRA3/NPA1/rank8/native49k head. No original pretrained
model forwards, no gradients, no optimizer or fresh DEV/GOLD. The 14 new
X DEV have been scored and are now SPENT, not independent blinded test.
Classify genuine A/B numeric full FP16 and 12GT differences versus
unchanged 53,751 S115 protected neural and exact matched pair of
actual self-generated native BPE outputs / second-gold ranks.
No selection, promotion, production, LIVE/main/Render/Supabase.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from research.calibri_nra3.s119q2i8m_fp16_early23_delta_forensics import (
    checked_safetensors,quantized_diff,
)

STAGE="CALIBRI_S119Q2I8Y_ACTUAL_S119Q2I8X_FULL43M_FP16_HARDNEG_PAIRED_OWN_AUDIT"
SOURCE_STAGE="CALIBRI_S119Q2I8X_TRUE43M_FRESH21TRAIN_FULL_GT12_NATIVE_CE_EARLY23X3_VS_HARDNEG"
SOURCE_REPORT="S119Q2I8X_TRUE43M_FULLJOINT_27TRAIN_EARLY23X3_CE_VS_HARDNEG_NATIVE49K_G025.json"
NEURAL_A="S119Q2I8X_TRUE43M_FULLJOINT_NATIVE_EARLY23X3_CE_FP16_G0.safetensors"
NEURAL_B="S119Q2I8X_TRUE43M_FULLJOINT_NATIVE_EARLY23X3_HARDNEG_FP16_G0.safetensors"
GT_A="S119Q2I8X_TRUE43M_FULLJOINT_EARLY23X3_CE_GT12_FP16_G0.safetensors"
GT_B="S119Q2I8X_TRUE43M_FULLJOINT_EARLY23X3_HARDNEG_GT12_FP16_G0.safetensors"
OLD="W_PREVIOUS_EARLY23_X3_FROZEN"
A="X_FULLJOINT_EARLY23_X3_CE"
B="X_FULLJOINT_EARLY23_X3_HARDNEG"
TRAIN_A="ALL_NATIVE_CE_EARLY23_X3"
TRAIN_B="ALL_NATIVE_CE_EARLY23_X3_HARDNEG"
COUNTS=(("NEW_DEV",14),("OLD_DEV",6))
VOCAB=49152

def check(ok,reason):
    if not ok:raise ValueError("S119q2i8y research proof failed: "+reason)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def paired_own(a,b,split,n):
    one=a[split]["OWN_samples"];two=b[split]["OWN_samples"]
    check(len(one)==len(two)==n and
          len({x["source_id"] for x in one})==n and
          {x["source_id"] for x in one}==
          {x["source_id"] for x in two},
          "exact original source identity and true self-native output n")
    lookup={x["source_id"]:x for x in two}
    per=[]
    for x in one:
        y=lookup[x["source_id"]]
        ids=x["generated_own_native_BPE_ids"]
        other=y["generated_own_native_BPE_ids"]
        for v in (x,y):
            seq=v["generated_own_native_BPE_ids"]
            check(isinstance(seq,list) and 1<=len(seq)<=40
                  and all(type(k) is int and 0<=k<VOCAB for k in seq)
                  and v["source_internal_DEV_NOT_independent_blind"] is True
                  and v["true_native_full_EOS_correct"] is False,
                  "real full native 49k autoregressive own BPE tokens, not gold prompt")
        first=next((k+1 for k,(u,v) in enumerate(zip(ids,other))
                    if u!=v),None)
        if first is None and len(ids)!=len(other):
            first=min(len(ids),len(other))+1
        per.append({
            "source_id":x["source_id"],
            "same_self_generated_native_BPE_sequence":first is None,
            "first_different_generated_native_BPE_position_1based":first,
            "A_first_two_native_correct":x["first_two_gold_BPE_correct"],
            "B_first_two_native_correct":y["first_two_gold_BPE_correct"],
            "A_first_three_native_correct":x["first_three_gold_BPE_correct"],
            "B_first_three_native_correct":y["first_three_gold_BPE_correct"],
            "A_EOS_emitted":bool(x["native_EOS"]),
            "B_EOS_emitted":bool(y["native_EOS"]),
            "A_native_generated_length":len(ids),
            "B_native_generated_length":len(other),
        })
    return {
        "source_count":n,
        "same_A_B_real_OWN_BPE_sequences":sum(z["same_self_generated_native_BPE_sequence"]
                                               for z in per),
        "different_A_B_real_OWN_BPE_sequences":sum(
            not z["same_self_generated_native_BPE_sequence"] for z in per),
        "A_correct_first_two":sum(z["A_first_two_native_correct"] for z in per),
        "B_correct_first_two":sum(z["B_first_two_native_correct"] for z in per),
        "A_correct_first_three":sum(z["A_first_three_native_correct"] for z in per),
        "B_correct_first_three":sum(z["B_first_three_native_correct"] for z in per),
        "A_native_EOS_emitted":sum(z["A_EOS_emitted"] for z in per),
        "B_native_EOS_emitted":sum(z["B_EOS_emitted"] for z in per),
        "paired_source_rows":per,
        "new_DEV_spent_no_independent_blind_or_model_selection":True,
    }

def paired_gold_second(a,b):
    aa=a["source_rows"];bb=b["source_rows"]
    check(len(aa)==len(bb)==14 and
          len({z["source_id"] for z in aa})==14 and
          {z["source_id"] for z in aa}=={z["source_id"] for z in bb},
          "same exact 14 SPENT author X DEV original49152 gold SECOND native BPE")
    lut={z["source_id"]:z for z in bb}
    better=worse=tie=0
    for z in aa:
        y=lut[z["source_id"]]
        old=z["gold_second_rank_full49152"]
        new=y["gold_second_rank_full49152"]
        check(type(old) is int and type(new) is int and
              1<=old<=VOCAB and 1<=new<=VOCAB and
              z["gold_previous_native_BPE_teacherforced_diagnostic_not_OWN"] is True and
              y["gold_previous_native_BPE_teacherforced_diagnostic_not_OWN"] is True,
              "full 49k REAL gold PREVIOUS BPE rank NOT generation or independent DEV")
        better+=new<old
        worse+=new>old
        tie+=new==old
    check(better+worse+tie==14,"all paired second-native rankings compared")
    return {"margin_B_better_lower_gold_rank_than_A":better,
            "margin_B_worse_higher_gold_rank_than_A":worse,
            "margin_B_equal_gold_rank_to_A":tie,
            "n_already_spent_source_pairs":14,
            "posthoc_annotated_NOT_blind_or_training":True}

def run(report,base):
    check(report["stage"]==SOURCE_STAGE
          and report["candidate_accepted"] is False and report["output_gate"]==0
          and report["matched_true43M_all_joint_368051_A_B_from_identical_W_FP16_and_GT12"] is True
          and report["previous_W_fulljoint_neural_421793_and_GT12_never_modified"] is True
          and report["no_new_regions_and_NO_EOS_logits_mask_in_decoder"] is True
          and report["prior_protected_S115_53751_unchanged"] is True
          and report["actual_original_S142_43M_12GT_native_49152_training_forwards"]==1350
          and report["actual_original_S142_43M_12GT_native_49152_OWN_forwards"]==1355
          and report["actual_original_S142_43M_12GT_native_49152_posthoc_gold_second_forwards"]==42,
          "only successful completed TRUE S142 physical original43M 12 GT native X")
    tr=report["matched_27TRAIN_BOTH_real_original43M_all_joints"]
    check(set(tr)=={TRAIN_A,TRAIN_B}
          and all(tr[k]["real_all_neural_and_gt_Adam_updates"]==27
                  and tr[k]["trainable_parameter_count"]==368051
                  and tr[k]["train_only_true_SECOND_THIRD_native_gold_CE_weight"]==3.
                  for k in tr)
          and tr[TRAIN_A]["hardnegative_ONLY_correct_gold_vs_wrong_nonEOS_BPE2_3_calls"]==0
          and tr[TRAIN_B]["hardnegative_ONLY_correct_gold_vs_wrong_nonEOS_BPE2_3_calls"]>27
          and tr[TRAIN_B]["hardnegative_TRAIN_only_margin_loss_lambda"]==.5,
          "A/B same original 27× 49k physical TRAIN, only B λ=.5 hard wrong native nonEOS")
    meta=report["persisted_A_B_original43M_FULL_neural_and_12GT_FP16_gate0"]
    check(set(meta)=={"CE_EARLY23_X3_joint_FP16",
                      "EARLY23_X3_HARDNEG_joint_FP16",
                      "CE_EARLY23_X3_GT12","EARLY23_X3_HARDNEG_GT12"},
          "all REAL original FP16 421793 A/B neural and 12 gates, not gradient proxy")
    neural_a=checked_safetensors(base,NEURAL_A,meta["CE_EARLY23_X3_joint_FP16"])
    neural_b=checked_safetensors(base,NEURAL_B,meta["EARLY23_X3_HARDNEG_joint_FP16"])
    gates_a=checked_safetensors(base,GT_A,meta["CE_EARLY23_X3_GT12"],gate=True)
    gates_b=checked_safetensors(base,GT_B,meta["EARLY23_X3_HARDNEG_GT12"],gate=True)
    neural=quantized_diff(neural_a,neural_b)
    gt=quantized_diff(gates_a,gates_b)
    check(neural["all_saved_params"]==421793
          and gt["all_saved_params"]==12
          and neural["changed_fp16_params"]>0
          and neural["regions"]["S115_protected_neural_state"][
              "parameters_different"]==0,
          "REAL physical trained FP16 delta nonzero while original protected S115 identical")
    own=report["fresh_14X_INTERNAL_DEV_plus6_old_FULL_OWN_native_autoregressive"]
    check(set(own)=={OLD,A,B}
          and all(k in own for k in (OLD,A,B))
          and all(own[k]["NEW_DEV"]["scores"]["DEV_n"]==14
                  and own[k]["NEW_DEV"]["scores"][
                      "correct_full_native_ANSWER_and_EOS"]==0 for k in own),
          "all three physical original 43M full native OWN source controls")
    pairs={split:paired_own(own[A],own[B],split,n) for split,n in COUNTS}
    ranks=report["posthoc_gold_previous_native_second_full49152_ranks_NOT_OWN"]
    check(set(ranks)=={OLD,A,B},"three true original native gold SECOND rank controls")
    return {
        "stage":STAGE,
        "TRUE_43M_actual_fp16_full_neural_A_vs_B":neural,
        "TRUE_12_GT_gates_fp16_A_vs_B":gt,
        "paired_actual_original43M_OWN_autoregressive_full_native_BPE":pairs,
        "paired_prev_gold_human_SECOND_original49152_rank":{
            "A_CE_vs_B_CEplus_nonEOS_hardnegative":paired_gold_second(ranks[A],ranks[B])},
        "input_new_X14_AUTHOR_DEV_is_now_SPENT_not_independent":True,
        "real_original43M_native_forward_calls_in_this_READONLY_audit":0,
        "no_new_model_optimization_no_teacher_and_no_original_J7_EVAL128":True,
        "original_S142_all_GT_Core_NRA3_NPA1_training_source_stays_research":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"ACTUAL_TRUE_FP16_HARDNEG_AB_LANGUAGE_NOT_RESTORED_NO_PROMOTION",
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--x-artifact-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    opts=ap.parse_args()
    check(opts.x_artifact_dir.is_dir() and
          not opts.out.exists(),"completed original physical X path and no output overwrite")
    path=opts.x_artifact_dir/SOURCE_REPORT
    check(path.is_file(),"original physical X complete 43M research report not present")
    out=run(json.loads(path.read_text(encoding="utf8")),opts.x_artifact_dir)
    out["source_completed_43M_AB_X_report_SHA256"]=sha(path)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    opts.out.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,
                                   indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Y_TRUE43M_FP16_NEURAL_GT12_AND_OWN_PAIRED_AB "+json.dumps({
        "neural_FP16_changed_original":out["TRUE_43M_actual_fp16_full_neural_A_vs_B"][
            "changed_fp16_params"],
        "GT12_FP16_changed_gates":out["TRUE_12_GT_gates_fp16_A_vs_B"][
            "changed_fp16_params"],
        "new_X14_actual_OWN_paired":{k:v for k,v in
            out["paired_actual_original43M_OWN_autoregressive_full_native_BPE"][
                "NEW_DEV"].items() if k!="paired_source_rows"},
        "new_X14_gold_second_rank_paired":
            out["paired_prev_gold_human_SECOND_original49152_rank"],
        "candidate_accepted":False,
    },sort_keys=True,ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
