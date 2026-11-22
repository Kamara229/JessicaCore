"""S119q2i8z3: SHA-verified saved original S142 43M full FP16 physical
X/Z2 trained neural 421793, original fitted 12 Green Tunnel gates
and same-source actual native49k OWN and gold second BPE paired audit.

Compares S119q2i8z2 A normal human BPE2/3 CE×3 vs B SAME + fixed
TRAIN-only λ2 on learned existing Core midlayer tanh gate. W trained
parent, original 43M and protected S115 frozen. The 14 new Z2 author
synthetic DEV have ALREADY BEEN SCORED; NEVER independent blind.
No model inference/updates, no train or evaluation leaks or promotion.
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

STAGE="CALIBRI_S119Q2I8Z3_TRUE43M_ACTUAL_FP16_CORE_GATE_L2_VS_CE_AND_PAIRED_OWN"
SOURCE_STAGE="CALIBRI_S119Q2I8Z2_TRUE43M_NEW21TRAIN_FULL_GT12_CE_EARLY23X3_VS_TRAINONLY_CORE_GATE_L2"
SOURCE_REPORT="S119Q2I8Z2_TRUE43M_FULLJOINT_27TRAIN_NATIVE_CE_EARLY23_X3_VS_CORE_GATE_L2_G025.json"
NEURAL_A="S119Q2I8Z2_TRUE43M_FULLJOINT_NATIVE_CE_EARLY23_X3_FP16_G0.safetensors"
NEURAL_B="S119Q2I8Z2_TRUE43M_FULLJOINT_NATIVE_CORE_GATE_L2_FP16_G0.safetensors"
GT_A="S119Q2I8Z2_TRUE43M_FULLJOINT_NATIVE_CE_EARLY23_X3_GT12_FP16_G0.safetensors"
GT_B="S119Q2I8Z2_TRUE43M_FULLJOINT_NATIVE_CORE_GATE_L2_GT12_FP16_G0.safetensors"
OLD="W_PREVIOUS_EARLY23_X3_FROZEN"
A="Z2_FULLJOINT_EARLY23_X3_CE"
B="Z2_FULLJOINT_EARLY23_X3_CORE_TRAINONLY_L2"
TRAIN_A="ALL_NATIVE_CE_EARLY23_X3"
TRAIN_B="ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE"
COUNTS=(("NEW_DEV",14),("OLD_DEV",6))
VOCAB=49152

def check(ok,reason):
    if not ok:raise ValueError("S119q2i8z3 research physical FP16 proof failed: "+reason)

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
    aa=a["per_source_read_only_teacherforced_previous_gold_BPE"]
    bb=b["per_source_read_only_teacherforced_previous_gold_BPE"]
    check(len(aa)==len(bb)==14 and
          len({z["source_id"] for z in aa})==14 and
          {z["source_id"] for z in aa}=={z["source_id"] for z in bb},
          "same exact 14 SPENT author X DEV original49152 gold SECOND native BPE")
    lut={z["source_id"]:z for z in bb}
    better=worse=tie=0
    for z in aa:
        y=lut[z["source_id"]]
        old=z["correct_gold_second_native_rank_full49152"]
        new=y["correct_gold_second_native_rank_full49152"]
        check(type(old) is int and type(new) is int and
              1<=old<=VOCAB and 1<=new<=VOCAB and
              z["given_gold_first_BPE_ONLY_for_readonly_diagnostic"] is True and
              y["given_gold_first_BPE_ONLY_for_readonly_diagnostic"] is True,
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
          and report["candidate_accepted"] is False
          and report["output_gate"]==0
          and report["matched_27_true43M_alljoint_neural_and_GT12_A_B_from_same_W_FP16"] is True
          and report["same_human_BPE2andBPE3_CE3x_in_both_arms"] is True
          and report["no_inference_manual_CORE_gate_overrides"] is True
          and report["frozen_original43M_S115_protected_W_parent_unchanged"] is True
          and report["existing_CORE_gate_only_extra_train_regularizer_lambda"]==2.
          and report["true43M_49k_full_native_TRAIN_forwards"]==1384
          and report["true43M_49k_full_native_OWN_forwards"]==1361
          and report["true43M_49k_gold_second_teacherforced_only_forwards"]==42,
          "only actual successful completed original 43M Z2 gate0 full physical A/B")
    tr=report["true43M_full_GT12_Core_NRA3_NPA1_native49152_27TRAIN_AB"]
    check(set(tr)=={TRAIN_A,TRAIN_B}
          and all(tr[k]["real_all_neural_and_gt_Adam_updates"]==27
                  and tr[k]["trainable_parameter_count"]==368051
                  and tr[k]["train_only_true_SECOND_THIRD_native_gold_CE_weight"]==3.
                  and tr[k]["Core_gate_training_ONLY_regularizer_count"]==27
                  for k in tr)
          and tr[TRAIN_A]["Core_gate_L2_lambda_PER_27TRAIN_answer"]==0
          and tr[TRAIN_B]["Core_gate_L2_lambda_PER_27TRAIN_answer"]==2
          and tr[TRAIN_A]["learned_existing_CORE_component_tanh_GATE_before_27TRAIN"]==
              tr[TRAIN_B]["learned_existing_CORE_component_tanh_GATE_before_27TRAIN"],
          "same W start and matched 27 true original43M 368051 Adam, B ONLY Core train L2")
    meta=report["neural_all421793_plus_separate12GT_FP16_saved"]
    check(set(meta)=={"CE_EARLY23_X3_FP16","SHRINK_CORE_GATE_FP16",
                      "CE_EARLY23_X3_GT12","SHRINK_CORE_GATE_GT12"},
          "all physical trained neural 421793 and full12 GT fitted fp16 file metadata")
    neural_a=checked_safetensors(base,NEURAL_A,meta["CE_EARLY23_X3_FP16"])
    neural_b=checked_safetensors(base,NEURAL_B,meta["SHRINK_CORE_GATE_FP16"])
    gates_a=checked_safetensors(base,GT_A,meta["CE_EARLY23_X3_GT12"],gate=True)
    gates_b=checked_safetensors(base,GT_B,meta["SHRINK_CORE_GATE_GT12"],gate=True)
    neural=quantized_diff(neural_a,neural_b)
    gt=quantized_diff(gates_a,gates_b)
    check(neural["all_saved_params"]==421793
          and gt["all_saved_params"]==12
          and neural["changed_fp16_params"]>0
          and neural["regions"]["S115_protected_neural_state"][
              "parameters_different"]==0,
          "REAL physical trained FP16 delta nonzero while original protected S115 identical")
    own=report["true43M_SELF_OWN_newZ2_14DEV_plus_OLD6_each_three_states"]
    check(set(own)=={OLD,A,B}
          and all(k in own for k in (OLD,A,B))
          and all(own[k]["NEW_DEV"]["scores"]["DEV_n"]==14
                  and own[k]["NEW_DEV"]["scores"][
                      "correct_full_native_ANSWER_and_EOS"]==0 for k in own),
          "all three physical original 43M full native OWN source controls")
    pairs={split:paired_own(own[A],own[B],split,n) for split,n in COUNTS}
    ranks=report["true43M_gold_SECOND_BPE_original49k_rank_only_POSTHOC"]
    check(set(ranks)=={OLD,A,B},"three true original native gold SECOND rank controls")
    return {
        "stage":STAGE,
        "TRUE_43M_actual_fp16_full_neural_A_vs_B":neural,
        "TRUE_12_GT_gates_fp16_A_vs_B":gt,
        "paired_actual_original43M_OWN_autoregressive_full_native_BPE":pairs,
        "paired_prev_gold_human_SECOND_original49152_rank":{
            "A_CE_vs_B_CEplus_trainonly_CoreGate_L2":paired_gold_second(ranks[A],ranks[B])},
        "input_new_Z2_14_AUTHOR_DEV_is_now_SPENT_not_independent":True,
        "TRAIN_ONLY_CORE_gate_start_and_end":{k:{"before":v["learned_existing_CORE_component_tanh_GATE_before_27TRAIN"],"after":v["learned_existing_CORE_component_tanh_GATE_after_27TRAIN"],"lambda":v["Core_gate_L2_lambda_PER_27TRAIN_answer"]} for k,v in tr.items()},
        "real_original43M_native_forward_calls_in_this_READONLY_audit":0,
        "no_new_model_optimization_no_teacher_and_no_original_J7_EVAL128":True,
        "original_S142_all_GT_Core_NRA3_NPA1_training_source_stays_research":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"ACTUAL_TRUE43M_FP16_CORE_GATE_L2_AB_NO_LANGUAGE_PROMOTION",
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--z2-artifact-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    opts=ap.parse_args()
    check(opts.z2_artifact_dir.is_dir() and
          not opts.out.exists(),"completed original physical X path and no output overwrite")
    path=opts.z2_artifact_dir/SOURCE_REPORT
    check(path.is_file(),"original physical X complete 43M research report not present")
    out=run(json.loads(path.read_text(encoding="utf8")),opts.z2_artifact_dir)
    out["source_completed_43M_AB_Z2_report_SHA256"]=sha(path)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    opts.out.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,
                                   indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Z3_TRUE43M_FP16_COREGATE_L2_AND_OWN_PAIRED_AB "+json.dumps({
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
