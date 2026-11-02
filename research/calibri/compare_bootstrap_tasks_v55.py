#!/usr/bin/env python3
"""Independent v55 verification of real INT6 CRC/SHA, bootstrap TRAIN
gate and four previously untouched supervised choice-answer task results.

No PyTorch import; verifier recomputes fixed gate from emitted NLL
values, rather than simply trusting the original decision function.
"""
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path

METHODS=("base_int6_pre","sse_cap_06pct","sse_cap_12pct","sse_cap_24pct")
FAMILIES={
    "WikiText":("WikiText_TRAIN_I_16384_18432","WikiText_TRAIN_J_18432_20480"),
    "AGNews":("AGNews_TRAIN_1536_1664","AGNews_TRAIN_1664_1792"),
    "SciQ":("SciQ_TRAIN_512_640","SciQ_TRAIN_640_768"),
    "BoolQ":("BoolQ_TRAIN_256_384","BoolQ_TRAIN_384_512"),
}
TASKS=("SciQ_TEST_128_160","ARCEasy_TEST_0_32",
       "BoolQ_VALIDATION_256_288","HellaSwag_VALIDATION_256_288")
REPS=1024
SEED=55
LOWER=102


def actual_frozen_selection(report,errors,host):
    gate=report["one_shot_train_only_gate"]
    observed=gate["all_training_NLL_and_PPL"]
    names=set(x for windows in FAMILIES.values() for x in windows)
    if ({k:tuple(v) for k,v in gate["family_window_mapping"].items()}
        !=FAMILIES):
        errors.append(host+": changed four-family frozen TRAIN windows")
    if set(report["eight_TRAIN_gate_token_SHA256"])!=names:
        errors.append(host+": missing 8 disjoint TRAIN SHA256 fingerprints")
    if set(observed)!=set(METHODS) or any(set(observed[k])!=names for k in METHODS):
        errors.append(host+": incomplete TRAIN-only model NLL data")
        return "invalid"
    results={}
    base=observed["base_int6_pre"]
    for method in METHODS[1:]:
        source=observed[method]
        gains={w:base[w]["nll"]-source[w]["nll"] for w in names}
        fam={k:math.fsum(gains[w] for w in win)/2 for k,win in FAMILIES.items()}
        rng=random.Random(SEED)
        ordered=tuple(FAMILIES)
        scores=[]
        for _ in range(REPS):
            this=[]
            for _ in ordered:
                family=ordered[rng.randrange(len(ordered))]
                w=FAMILIES[family][rng.randrange(2)]
                this.append(gains[w])
            scores.append(math.fsum(this)/4)
        scores.sort()
        decile=scores[LOWER]
        wins=sum(x>0 for x in fam.values())
        valid=(wins>=3 and min(fam.values())>=-0.0080 and decile>=0.0005)
        row=gate["candidate_training_stability"][method]
        if (row["paired_training_gain_by_window"]!=gains or
            row["equal_family_mean_NLL_gains"]!=fam or
            row["positive_corpus_families"]!=wins or
            abs(row["bootstrap_tenth_percentile_NLL_gain"]-decile)>1e-12 or
            row["passes_frozen_heuristic_gate"]!=valid or
            row["seed"]!=SEED or row["bootstrap_draws"]!=REPS):
            errors.append(host+": actual TRAIN bootstrap evidence mismatch "+method)
        results[method]=(valid,decile,math.fsum(fam.values())/4)
    valid=[k for k in METHODS[1:] if results[k][0]]
    chosen=(max(valid,key=lambda k:(results[k][1],results[k][2],
                 -METHODS.index(k))) if valid else "base_int6_pre")
    if chosen!=gate["chosen"] or chosen!=report["selected_method"]:
        errors.append(host+": actual selection not frozen bootstrap decision")
    return chosen


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True);p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    errors=[]
    fixed_keys=("experiment","real_pretrained_model","source_revision",
        "source_checkpoint_fp32_SHA256","source_all_parameter_count",
        "wiki_revision","ag_news_revision","sciq_training_revision",
        "boolq_training_revision","external_task_source_revision_by_dataset",
        "first_use_task_definition_and_record_SHA256",
        "WikiText_TRAIN_A_SHA256","eight_TRAIN_gate_token_SHA256")
    for key in fixed_keys:
        if a[key]!=b[key]:
            errors.append("crosshost pretrained model or actual source/text mismatch: "+key)
    all_host={}
    for host,d in (("ubuntu22",a),("ubuntu24",b)):
        baseline=d["base_INT6_complete_source"]
        if baseline["bytes"]!=10803661 or len(baseline["sha256"])!=64:
            errors.append(host+": full genuine physical trained CB46 changed")
        proof=d["deterministic_row_budget"]
        sides=proof["plans"]
        if set(sides)!=set(METHODS[1:]):
            errors.append(host+": missing physical correction alternatives")
        for m in METHODS[1:]:
            side=sides[m]
            if side["bytes"] not in (610751,610715,610875):
                errors.append(host+": original physical correction byte budget changed "+m)
            if (baseline["bytes"]+side["bytes"]+
                proof["reserved_other_correction_bytes"]>
                proof["max_all_model_plus_all_correction_bytes"]):
                errors.append(host+": exceeded full model 6.5bpw ceiling "+m)
        selected=actual_frozen_selection(d,errors,host)
        teacher=d["original_teacher_first_use_task_accuracy"]
        results=d["all_candidates_first_use_task_accuracy"]
        met=d["first_use_task_definition_and_record_SHA256"]
        if (set(teacher)!=set(TASKS) or set(results)!=set(METHODS) or
            set(met["all_task_text_sha256"])!=set(TASKS) or
            set(d["external_task_source_revision_by_dataset"])!=set(TASKS)):
            errors.append(host+": missing one first-use independent heldout task")
        for m,taskdict in (("teacher",teacher),*results.items()):
            if set(taskdict)!=set(TASKS):
                errors.append(host+": missing tested tasks for "+m)
                continue
            for name,entry in taskdict.items():
                if (entry["n"]!=32 or
                    len(entry["per_example_correct"])!=32 or
                    len(entry["predicted_label_indices"])!=32 or
                    entry["correct"]!=sum(entry["per_example_correct"]) or
                    not 0<=entry["correct"]<=32 or
                    abs(entry["accuracy"]-entry["correct"]/32)>1e-12):
                    errors.append(host+": incorrect recorded 32-item answer accuracy "+m+"/"+name)
        real=d["actual_full_model_warm_speed"]
        if (not math.isfinite(real["chosen_divided_by_teacher"]) or
            real["chosen_divided_by_teacher"]<=0):
            errors.append(host+": invalid complete model real latency ratio")
        for typ in ("original_teacher","chosen"):
            for key in ("prefill_ms","decode32_ms","total_prefill_and_decode32_ms"):
                numbers=real[typ][key]["raw"]
                if len(numbers)!=7 or any((not math.isfinite(t) or t<=0)
                                          for t in numbers):
                    errors.append(host+": missing actual warm total samples")
        by_task={n:{
            "teacher_correct":teacher[n]["correct"],
            "plain_INT6_correct":results["base_int6_pre"][n]["correct"],
            "selected_correct":results[selected][n]["correct"],
            "selected_minus_plain_INT6_correct":
                results[selected][n]["correct"]-
                results["base_int6_pre"][n]["correct"]
            } for n in TASKS}
        all_host[host]={
            "TRAIN_only_chosen":selected,
            "selected_effective_real_bpw":d["selected_actual_bpw"],
            "total_warm_latency_div_teacher":real["chosen_divided_by_teacher"],
            "user_20pct_slowdown_limit_pass":real["passes_plus20pct_user_limit"],
            "first_use_real_question_answer_accuracy":by_task,
            "true_corrected_sidecar_sizes":{m:sides[m]["bytes"]
                                            for m in METHODS[1:]}
        }
    same_source=a["base_INT6_complete_source"]["sha256"]==b[
        "base_INT6_complete_source"]["sha256"]
    if not same_source:errors.append("different full physical CB46 source hash")
    physical={}
    for m in METHODS[1:]:
        x=a["deterministic_row_budget"]["plans"][m]
        y=b["deterministic_row_budget"]["plans"][m]
        physical[m]=(x["sha256"]==y["sha256"] and x["bytes"]==y["bytes"]
                     and x["module_counts"]==y["module_counts"]
                     and x["selected_real_rows"]==y["selected_real_rows"])
        if not physical[m]:
            errors.append("physically inconsistent source-only trained CR47 correction: "+m)
    if all_host["ubuntu22"]["TRAIN_only_chosen"]!=all_host["ubuntu24"]["TRAIN_only_chosen"]:
        errors.append("crosshost TRAIN bootstrap model selection differs")
    for m in ("original_teacher",*METHODS):
        section=("original_teacher_first_use_task_accuracy" if m=="original_teacher"
                 else "all_candidates_first_use_task_accuracy")
        for name in TASKS:
            arow=a[section][name] if m=="original_teacher" else a[section][m][name]
            brow=b[section][name] if m=="original_teacher" else b[section][m][name]
            if arow["per_example_correct"]!=brow["per_example_correct"]:
                errors.append("crosshost measured task predictions disagree: "+m+"/"+name)
    report={
        "PASS":not errors,"errors":errors,
        "all_physical_source_and_sidecar_bitwise_SHA_identical":
            {"CB46":same_source,**physical},
        "outcomes_by_independent_CPU":all_host,
        "limitations":[
            "TRAIN heuristic bootstrap is NOT an unbiased population confidence interval.",
            "Gold-label task accuracy on 32 records each is too small to establish general task capability or statistical significance.",
            "The original pretrained Pythia14m is not Jessica and was not instruction-fine-tuned for these QA tasks.",
            "On-disk INT6 checkpoint is expanded into dense FP32 hot RAM at cold load.",
            "No main/Jessica chat/Render/Supabase/APK/settings/secrets changed."],
    }
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
