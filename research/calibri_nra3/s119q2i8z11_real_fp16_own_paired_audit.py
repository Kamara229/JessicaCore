"""S119q2i8z11: read-only physical FP16 and OWN-native paired Z10 forensic.

Strict SHA check of completed successful #45115966394 artifact (original
S142 true43M / 12 GT / Core NRA3 NPA1 / native49152), paired full child
421793 FP16 A/B weights and the 12 separately learned GT gate scalars.
Compare the SAME 14 already-spent Z9 synthetic author DEV and six OLD
DEV genuine self-generated native BPE traces; separate human-gold-first
BPE2 ranks and NLL are TEACHER-FORCED forensic, never OWN language proof.
NO new model forward, no gradients, training or model selection/promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from research.calibri_nra3.s119q2i8m_fp16_early23_delta_forensics import (
    checked_safetensors, quantized_diff,
)
from research.calibri_nra3.s119q2i8z9_new_ru_train_dev_source import verify as source_verify

STAGE = "CALIBRI_S119Q2I8Z11_SHA_PINNED_TRUE43M_CONTEXTTOP8_FP16_OWN_PAIRED"
SOURCE_STAGE = "CALIBRI_S119Q2I8Z10_TRUE43M_GT12_CE_EARLY23X3_VS_CONTEXT_TOP8_BPE2"
SOURCE_REPORT = "S119Q2I8Z10_TRUE43M_FULLJOINT_27TRAIN_NATIVE_CE_EARLY23_X3_VS_CONTEXT_TOP8_BPE2_G025.json"
NEURAL_A = "S119Q2I8Z10_TRUE43M_FULLJOINT_NATIVE_CE_EARLY23_X3_FP16_G0.safetensors"
NEURAL_B = "S119Q2I8Z10_TRUE43M_FULLJOINT_NATIVE_CONTEXT_TOP8_BPE2_FP16_G0.safetensors"
GT_A = "S119Q2I8Z10_TRUE43M_FULLJOINT_NATIVE_CE_EARLY23_X3_GT12_FP16_G0.safetensors"
GT_B = "S119Q2I8Z10_TRUE43M_FULLJOINT_NATIVE_CONTEXT_TOP8_BPE2_GT12_FP16_G0.safetensors"
OLD = "W_PREVIOUS_EARLY23_X3_FROZEN"
A = "Z10_FULLJOINT_EARLY23_X3_CE"
B = "Z10_FULLJOINT_EARLY23_X3_CONTEXT_TOP8_BPE2"
TRAIN_A = "ALL_NATIVE_CE_EARLY23_X3"
TRAIN_B = "ALL_NATIVE_CE_EARLY23_X3_CONTEXT_TOP8"
VOCAB = 49152
MAX_NATIVE = 40


def check(value, reason):
    if not value:
        raise ValueError("Z11 true original43M physical paired audit failed: " + reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_own(a: dict, b: dict, split: str, n: int) -> dict:
    x = a[split]["OWN_samples"]
    y = b[split]["OWN_samples"]
    check(len(x) == len(y) == n
          and len({e["source_id"] for e in x}) == n
          and {e["source_id"] for e in x} == {e["source_id"] for e in y},
          "strict source-by-source native OWN pair identity")
    look = {e["source_id"]: e for e in y}
    records = []
    for u in x:
        v = look[u["source_id"]]
        left, right = u["generated_own_native_BPE_ids"], v["generated_own_native_BPE_ids"]
        for item, seq in ((u,left),(v,right)):
            check(isinstance(seq,list) and 1 <= len(seq) <= MAX_NATIVE
                  and all(type(i) is int and 0 <= i < VOCAB for i in seq)
                  and item["source_internal_DEV_NOT_independent_blind"] is True
                  and item["true_native_full_EOS_correct"] is False
                  and item["eval_arm"] in (A,B),
                  "native self-generated 49k IDs, source spent, no exact full correct")
        first = next((idx+1 for idx, (m,k) in enumerate(zip(left,right))
                      if m!=k), None)
        if first is None and len(left)!=len(right):
            first = min(len(left),len(right))+1
        records.append({
            "source_id": u["source_id"],
            "source_prompt_SHA256": u["original_prompt_SHA256"],
            "same_full_native_SELF_sequence": first is None,
            "first_A_vs_B_generated_BPE_divergence_1based": first,
            "A_first_gold_BPE": bool(u["first_gold_BPE_correct"]),
            "B_first_gold_BPE": bool(v["first_gold_BPE_correct"]),
            "A_first_TWO_gold_BPE": bool(u["first_two_gold_BPE_correct"]),
            "B_first_TWO_gold_BPE": bool(v["first_two_gold_BPE_correct"]),
            "A_first_THREE_gold_BPE": bool(u["first_three_gold_BPE_correct"]),
            "B_first_THREE_gold_BPE": bool(v["first_three_gold_BPE_correct"]),
            "A_native_EOS": bool(u["native_EOS"]),
            "B_native_EOS": bool(v["native_EOS"]),
            "A_valid_UTF8": bool(u["valid_full_UTF8_bytes"]),
            "B_valid_UTF8": bool(v["valid_full_UTF8_bytes"]),
            "A_generated_native_length": len(left),
            "B_generated_native_length": len(right),
        })
    check(len(records)==n and
          len({x["source_prompt_SHA256"] for x in records})==n,
          "one strict SHA per paired native sample")
    counts = {
        "source_count": n,
        "same_A_B_full_OWN_BPE_sequences": sum(v["same_full_native_SELF_sequence"] for v in records),
        "different_A_B_full_OWN_BPE_sequences": sum(not v["same_full_native_SELF_sequence"] for v in records),
        "A_first_two_correct": sum(v["A_first_TWO_gold_BPE"] for v in records),
        "B_first_two_correct": sum(v["B_first_TWO_gold_BPE"] for v in records),
        "A_first_three_correct": sum(v["A_first_THREE_gold_BPE"] for v in records),
        "B_first_three_correct": sum(v["B_first_THREE_gold_BPE"] for v in records),
        "A_native_EOS": sum(v["A_native_EOS"] for v in records),
        "B_native_EOS": sum(v["B_native_EOS"] for v in records),
        "A_valid_UTF8": sum(v["A_valid_UTF8"] for v in records),
        "B_valid_UTF8": sum(v["B_valid_UTF8"] for v in records),
        "paired_sample_details": records,
    }
    check(counts["same_A_B_full_OWN_BPE_sequences"] +
          counts["different_A_B_full_OWN_BPE_sequences"]==n,
          "every native SELF pair accounted for")
    return counts


def paired_gold_second(a: dict, b: dict) -> dict:
    x = a["per_source_read_only_teacherforced_previous_gold_BPE"]
    y = b["per_source_read_only_teacherforced_previous_gold_BPE"]
    check(len(x)==len(y)==14 and
          len({v["source_id"] for v in x})==14 and
          {v["source_id"] for v in x}=={v["source_id"] for v in y},
          "exact 14 paired Z9 synthetic DEV gold-previous FULL49152 BPE2 ranks")
    look = {v["source_id"]: v for v in y}
    changes=[]
    for u in x:
        v=look[u["source_id"]]
        r0,r1=u["correct_gold_second_native_rank_full49152"],v["correct_gold_second_native_rank_full49152"]
        n0,n1=u["gold_second_NLL_nats"],v["gold_second_NLL_nats"]
        check(type(r0) is int and type(r1) is int and 1 <= r0 <= VOCAB
              and 1 <= r1 <= VOCAB and math.isfinite(n0) and math.isfinite(n1)
              and 0<=n0<1000 and 0<=n1<1000
              and u["given_gold_first_BPE_ONLY_for_readonly_diagnostic"] is True
              and v["given_gold_first_BPE_ONLY_for_readonly_diagnostic"] is True
              and u["source_prompt_sha256"]==v["source_prompt_sha256"],
              "matched teacherforced BPE2 49k rank+NLL without DEV training")
        changes.append({
            "source_id": u["source_id"],
            "A_gold_rank": r0, "B_gold_rank": r1,
            "B_minus_A_NLL_nats": round(n1-n0,9),
            "B_rank_improved":r1<r0,
            "B_rank_worsened":r1>r0,
            "B_rank_equal":r1==r0,
            "B_NLL_better":n1<n0,
            "B_NLL_worse":n1>n0,
        })
    return {
        "n_spent_paired_gold_previous_SECOND_BPE":14,
        "B_gold_rank_better":sum(v["B_rank_improved"] for v in changes),
        "B_gold_rank_worse":sum(v["B_rank_worsened"] for v in changes),
        "B_gold_rank_equal":sum(v["B_rank_equal"] for v in changes),
        "B_gold_NLL_better":sum(v["B_NLL_better"] for v in changes),
        "B_gold_NLL_worse":sum(v["B_NLL_worse"] for v in changes),
        "mean_B_minus_A_gold_NLL_nats":round(sum(v["B_minus_A_NLL_nats"] for v in changes)/14,9),
        "paired_records":changes,
        "NEVER_OWN_or_independent_blind":True,
    }


def run(report:dict, base:Path) -> dict:
    s=source_verify()
    check(report["stage"]==SOURCE_STAGE
          and report["candidate_accepted"] is False
          and report["output_gate"]==0
          and report["matched_27_true43M_alljoint_neural_and_GT12_A_B_from_same_W_FP16"] is True
          and report["same_human_BPE2andBPE3_CE3x_in_both_arms"] is True
          and report["no_inference_manual_CORE_gate_overrides"] is True
          and report["frozen_original43M_S115_protected_W_parent_unchanged"] is True
          and report["second_gold_BPE2_trainonly_contextual_top8_margin_weight"]==.20
          and report["new_Z9_21_TRAIN_sha256"]==s["Z9_21_TRAIN_sha256"]
          and report["new_Z9_14_DEV_sha256"]==s["Z9_14_DEV_sha256"]
          and report["true43M_49k_full_native_TRAIN_forwards"]==1346
          and report["true43M_49k_full_native_OWN_forwards"]==1707
          and report["true43M_49k_gold_second_teacherforced_only_forwards"]==42,
          "exact successful Z10 43M physical run with frozen source+sha")
    train=report["true43M_full_GT12_Core_NRA3_NPA1_native49152_27TRAIN_AB"]
    check(set(train)=={TRAIN_A,TRAIN_B}
          and all(v["real_all_neural_and_gt_Adam_updates"]==27
                  and v["trainable_parameter_count"]==368051
                  and v["train_only_true_SECOND_THIRD_native_gold_CE_weight"]==3
                  and v["second_BPE_context_trainonly_margin_count"]==27
                  for v in train.values())
          and train[TRAIN_A]["context_top8_aux_weight_if_B"]==0.
          and train[TRAIN_B]["context_top8_aux_weight_if_B"]==.20,
          "matched 27 Adam native CE vs B contextual top8 TRAIN second-only")
    meta=report["neural_all421793_plus_separate12GT_FP16_saved"]
    check(set(meta)=={"CE_EARLY23_X3_FP16","CONTEXT_TOP8_BPE2_FP16",
                      "CE_EARLY23_X3_GT12","CONTEXT_TOP8_BPE2_GT12"},
          "all 421793 FP16 child +12 GT physical file and sha metadata")
    aa=checked_safetensors(base,NEURAL_A,meta["CE_EARLY23_X3_FP16"])
    bb=checked_safetensors(base,NEURAL_B,meta["CONTEXT_TOP8_BPE2_FP16"])
    ga=checked_safetensors(base,GT_A,meta["CE_EARLY23_X3_GT12"],gate=True)
    gb=checked_safetensors(base,GT_B,meta["CONTEXT_TOP8_BPE2_GT12"],gate=True)
    neural=quantized_diff(aa,bb)
    gt=quantized_diff(ga,gb)
    check(neural["all_saved_params"]==421793 and
          gt["all_saved_params"]==12 and
          neural["changed_fp16_params"]>0 and
          neural["regions"]["S115_protected_neural_state"]["parameters_different"]==0,
          "real neural trained state DIFFERENT while exact protected S115 unchanged")
    own=report["true43M_SELF_OWN_newZ10_14DEV_plus_OLD6_each_three_states"]
    check(set(own)=={OLD,A,B} and
          all(own[v]["NEW_DEV"]["scores"]["DEV_n"]==14
              and own[v]["NEW_DEV"]["scores"]["correct_full_native_ANSWER_and_EOS"]==0
              for v in (OLD,A,B)),
          "three true ORIGINAL43M physical own native sequences no completed answer")
    pa=paired_own(own[A],own[B],"NEW_DEV",14)
    pb=paired_own(own[A],own[B],"OLD_DEV",6)
    ranks=report["true43M_gold_SECOND_BPE_original49k_rank_only_POSTHOC"]
    check(set(ranks)=={OLD,A,B},"all read-only full-native49k human-gold BPE2 controls")
    gold=paired_gold_second(ranks[A],ranks[B])
    check(pa["A_first_two_correct"]==pa["B_first_two_correct"]==0
          and pa["A_first_three_correct"]==pa["B_first_three_correct"]==0
          and gold["B_gold_rank_better"]+gold["B_gold_rank_worse"]+
              gold["B_gold_rank_equal"]==14,
          "no first TWO correct true self native BPE despite teacherforced small NLL")
    return {
        "stage":STAGE,
        "source_completed_43M_stage":SOURCE_STAGE,
        "source_Z9_TRAIN_sha256":s["Z9_21_TRAIN_sha256"],
        "source_Z9_DEV_sha256":s["Z9_14_DEV_sha256"],
        "FP16_all421793_true_neural_A_vs_B":neural,
        "FP16_all12_GT_gates_A_vs_B":gt,
        "TRUE_original43M_full_native_SELF_paired":{"NEW_DEV":pa,"OLD_DEV":pb},
        "gold_previous_second_FULL49152_teacherforced_only":gold,
        "real_original43M_model_forwards_THIS_READONLY_stage":0,
        "new_model_gradient_or_optimizer_steps_THIS_stage":0,
        "no_616J7_EVAL128_GOLD_bpp_CPU_or_promotion":True,
        "original_S142_frozen_no_pretrained_or_LIVE_changes":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"NO_PROVEN_NATIVE_LANGUAGE_RESTORATION_CONTEXT_TOP8_SECOND_BPE",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--z10-artifact-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    p=ap.parse_args()
    check(p.z10_artifact_dir.is_dir() and not p.out.exists(),
          "completed source only and no overwrite")
    file=p.z10_artifact_dir/SOURCE_REPORT
    check(file.is_file(),"only exact physically completed Z10 original43M artifact")
    result=run(json.loads(file.read_text(encoding="utf8")),p.z10_artifact_dir)
    result["source_completed_Z10_report_SHA256"]=sha(file)
    p.out.parent.mkdir(parents=True,exist_ok=True)
    p.out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Z11_ORIGINAL43M_FP16_OWN_PAIRED "+json.dumps({
        "neural_fp16_changed":result["FP16_all421793_true_neural_A_vs_B"]["changed_fp16_params"],
        "GT12_fp16_changed":result["FP16_all12_GT_gates_A_vs_B"]["changed_fp16_params"],
        "Z9_NEW14_TRUE_OWN_paired":{k:v for k,v in
            result["TRUE_original43M_full_native_SELF_paired"]["NEW_DEV"].items()
            if k!="paired_sample_details"},
        "gold_previous_BPE2_teacherforced_paired":{k:v for k,v in
            result["gold_previous_second_FULL49152_teacherforced_only"].items()
            if k!="paired_records"},
        "candidate_accepted":False,"output_gate":0,
    },ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
