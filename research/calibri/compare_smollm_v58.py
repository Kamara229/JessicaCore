#!/usr/bin/env python3
"""Independent cross-CPU CALIBRI v58 actual SmolLM2 135M checkpoint verifier.

No pretrained model is loaded by this third runner; the two independent
CPU executions already stored full source CB58, repair CR47 and heldout
answer predictions. Their physical hashes/rates and TRAIN-only gate
math are checked from independently produced reports.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

METHODS=("base_int6_pre","source_sse_balanced_repair")
TASKS=("SciQ","BoolQ")


def run(a,b):
    errors=[]
    common=("experiment","new_model","original_model_revision",
            "trained_instruction_model","architecture",
            "unique_original_parameter_count",
            "source_tied_weight_aliases","training_dataset_revision",
            "source_WikiText_train_revision","TRAIN_real_2x16_task_text_SHA256",
            "first_read_after_gate_HELDOUT_2x32_SHA256")
    for field in common:
        if a[field]!=b[field]:
            errors.append("independent CPUs disagree on source/training/test hash "+field)
    if a["new_model"]!="HuggingFaceTB/SmolLM2-135M-Instruct":
        errors.append("wrong newly designated stronger open instruction model")
    if not a["source_tied_weight_aliases"]:
        errors.append("missing actual tied trained source aliases")
    evidence={}
    for host,data in (("ubuntu22",a),("ubuntu24",b)):
        params=data["unique_original_parameter_count"]
        base=data["entire_physical_checkpoint"]
        patch=data["source_only_physical_FP16_sparse_repair"]
        max_bytes=data["full_entire_model_MAX_6_50bpw_bytes"]
        reserved=data["reserved_other_post_bytes"]
        if not (120_000_000<params<150_000_000):
            errors.append(host+" is not 135M UNIQUE pretrained source parameters")
        if not (70_000_000<base["bytes"]<max_bytes):
            errors.append(host+" missing physical full model signed INT6 source")
        if (len(base["sha256"])!=64 or len(patch["sha256"])!=64 or
            patch["bytes"]<100_000 or patch["selected_original_rows"]<=0):
            errors.append(host+" missing source or correction SHA/physical bytes")
        if (base["bytes"]+patch["bytes"]+reserved>max_bytes or
            max_bytes!=math.floor(6.5*params/8)):
            errors.append(host+" full source+patch exceeds actual unique 6.50bpw")
        if (data["source_only_physical_FP16_sparse_repair"]["full_model_plus_patch_bytes"]
            !=base["bytes"]+patch["bytes"]):
            errors.append(host+" physical source+patch accounting mismatch")
        if base["bytes"]+(
            patch["bytes"] if data["selected_method"]==METHODS[1] else 0
            )!=data["selected_actual_stored_bytes"]:
            errors.append(host+" chosen physical correction bytes not actual")
        gate=data["one_shot_TRAIN_task_gate"]
        if set(gate["TRAIN_only_full_task_metrics"])!=set(METHODS):
            errors.append(host+" missing actual TRAIN question evaluation")
        else:
            train=gate["TRAIN_only_full_task_metrics"]
            if any(set(train[m])!=set(TASKS) for m in METHODS):
                errors.append(host+" incomplete original TRAIN real QA")
            else:
                margin={}
                correct={}
                for task in TASKS:
                    if train[METHODS[0]][task]["n"]!=16 or train[METHODS[1]][task]["n"]!=16:
                        errors.append(host+" real TRAIN label count not 16 "+task)
                    margin[task]=(train[METHODS[1]][task]["mean_gold_logprob_margin"]-
                                  train[METHODS[0]][task]["mean_gold_logprob_margin"])
                    correct[task]=train[METHODS[1]][task]["correct"]-train[
                         METHODS[0]][task]["correct"]
                allowed=(sum(correct.values())>=0 and min(correct.values())>=0
                         and min(margin.values())>0
                         and math.fsum(margin.values())/2>=0.001)
                expected=METHODS[1] if allowed else METHODS[0]
                if (allowed!=gate["eligible_for_expensive_repair"] or
                    expected!=gate["chosen"] or expected!=data["selected_method"] or
                    any(abs(margin[t]-gate["training_family_margin_deltas"][t])>1e-12
                        for t in TASKS) or
                    correct!=gate["training_exact_correct_deltas"]):
                    errors.append(host+" incorrect preregistered TRAIN-only choice")
        task=data["heldout_all_INT6_methods_task_accuracy"]
        orig=data["heldout_teacher_real_task_accuracy"]
        if set(task)!=set(METHODS) or set(orig)!=set(TASKS):
            errors.append(host+" missing first-use answer quality variants")
        for key,items in (("teacher",orig),*task.items()):
            if set(items)!=set(TASKS):
                errors.append(host+" missing one unseen task "+key)
                continue
            for name,row in items.items():
                if (row["n"]!=32 or len(row["per_item_correct"])!=32
                    or row["correct"]!=sum(row["per_item_correct"])
                    or len(row["predicted_option_indices"])!=32
                    or abs(row["accuracy"]-row["correct"]/32)>1e-12):
                    errors.append(host+" corrupt first-use answer score "+key+"/"+name)
        warm=data["actual_full_model_speed"]
        ratio=warm["chosen_to_original_warm_total_ratio"]
        if (not math.isfinite(ratio) or ratio<=0 or
            warm["user_20pct_max_slowdown_pass"]!=(ratio<=1.20)):
            errors.append(host+" wrong real warm whole-model speed")
        for which in ("original_uncompressed_teacher","from_physical_files_chosen"):
            for metric in ("prefill_ms","decode32_ms","total_prefill_and_decode32_ms"):
                raw=warm[which][metric]["raw"]
                if len(raw)!=7 or any(not math.isfinite(x) or x<=0 for x in raw):
                    errors.append(host+" missing 7 true full-model timing results")
        evidence[host]={
            "new_instruction_model":data["new_model"],
            "trained_source_unique_parameters":params,
            "full_physical_CB58_int6_bytes":base["bytes"],
            "source_derived_FP16_correction_bytes":patch["bytes"],
            "max_full_model_physical_bpw":6.5,
            "actual_selected_physical_bpw":data["selected_actual_stored_bpw"],
            "TRAIN_only_method":data["selected_method"],
            "real_warm_selected_to_teacher_ratio":ratio,
            "user_20pct_warm_limit_pass":warm["user_20pct_max_slowdown_pass"],
            "unseen_actual_zero_shot_correct_out_of_32":{
                n:{"teacher":orig[n]["correct"],
                   "plain_INT6":task[METHODS[0]][n]["correct"],
                   "extra_FP16":task[METHODS[1]][n]["correct"]}
                for n in TASKS},
            "shared_tied_source_aliases":data["source_tied_weight_aliases"]}
    source_match=(a["entire_physical_checkpoint"]["sha256"]==
                  b["entire_physical_checkpoint"]["sha256"])
    patch_match=(a["source_only_physical_FP16_sparse_repair"]["sha256"]==
                 b["source_only_physical_FP16_sparse_repair"]["sha256"])
    if not source_match:errors.append("full CB58 source nonidentical across CPU hosts")
    if not patch_match:errors.append("full CR47 sidecar source score not bitwise deterministic")
    if evidence["ubuntu22"]["TRAIN_only_method"]!=evidence["ubuntu24"]["TRAIN_only_method"]:
        errors.append("independent hosts disagreed on frozen TRAIN selection")
    for variant in ("heldout_teacher_real_task_accuracy",):
        for task in TASKS:
            if (a[variant][task]["per_item_correct"]!=
                b[variant][task]["per_item_correct"]):
                errors.append("teacher first-use predictions vary "+task)
    for method in METHODS:
        for task in TASKS:
            if (a["heldout_all_INT6_methods_task_accuracy"][method][task]["per_item_correct"]!=
                b["heldout_all_INT6_methods_task_accuracy"][method][task]["per_item_correct"]):
                errors.append("source restored INT6 predictions vary "+method+"/"+task)
    return {"PASS":not errors,"errors":errors,
            "cross_cpu_physical_CB58_bitwise_equal":source_match,
            "cross_cpu_physical_CR47_bitwise_equal":patch_match,
            "independent_cpu_outcomes":evidence,
            "limitations":[
                "Actual instruction-finetuned SmolLM2 135M is different from production Jessica.",
                "Only 16 TRAIN and 32 HELDOUT examples per task; no population quality claim.",
                "Original BF16 pretrained source is evaluated after conversion to FP32 for CPU paired equality.",
                "Physical INT6 checkpoint uses full dense FP32 hot-RAM decode, not native packed INT6 matrix multiplication.",
                "All artifacts from isolated research branch, no production service changes."]}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True)
    p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    result=run(a,b)
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result["PASS"]:
        raise SystemExit(1)
