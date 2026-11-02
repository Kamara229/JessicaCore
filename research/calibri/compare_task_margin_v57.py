#!/usr/bin/env python3
"""CALIBRI v57 independent verifier: same physical INT6 bytes across CPUs,
recompute NLL preregistered bootstrap and task-margin TRAIN gate from
raw scores, check all heldout 128-item answer outcomes and warm speed.
No pretrained PyTorch model needs to be loaded by this checker.
"""
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path

METHODS=("base_int6_pre","sse_cap_06pct","sse_cap_12pct","sse_cap_24pct")
TASKS=("SciQ","ARCEasy","BoolQ","HellaSwag")
FAMILIES={
    "WikiText":("WikiText_TRAIN_I_16384_18432","WikiText_TRAIN_J_18432_20480"),
    "AGNews":("AGNews_TRAIN_1536_1664","AGNews_TRAIN_1664_1792"),
    "SciQ":("SciQ_TRAIN_512_640","SciQ_TRAIN_640_768"),
    "BoolQ":("BoolQ_TRAIN_256_384","BoolQ_TRAIN_384_512"),
}


def nll_gate(report,errors,host):
    gate=report["one_shot_train_only_gate"]
    data=gate["all_training_NLL_and_PPL"]
    names={n for wins in FAMILIES.values() for n in wins}
    if {k:tuple(v) for k,v in gate["family_window_mapping"].items()}!=FAMILIES:
        errors.append(host+" altered NLL TRAIN family window definitions")
    if set(report["eight_TRAIN_gate_token_SHA256"])!=names:
        errors.append(host+" missing preregistered NLL TRAIN hashes")
    if set(data)!=set(METHODS) or any(set(data[m])!=names for m in METHODS):
        errors.append(host+" incomplete original TRAIN NLL measurements")
        return "INVALID"
    results={}
    for method in METHODS[1:]:
        changes={n:data["base_int6_pre"][n]["nll"]-data[method][n]["nll"]
                 for n in names}
        family={f:math.fsum(changes[n] for n in windows)/2
                for f,windows in FAMILIES.items()}
        r=random.Random(55)
        fr=tuple(FAMILIES)
        draws=[]
        for _ in range(1024):
            vals=[]
            for _ in fr:
                key=fr[r.randrange(4)]
                vals.append(changes[FAMILIES[key][r.randrange(2)]])
            draws.append(math.fsum(vals)/4)
        draws.sort()
        decile=draws[102]
        approved=(sum(x>0 for x in family.values())>=3
                  and min(family.values())>=-0.008 and decile>=0.0005)
        proof=gate["candidate_training_stability"][method]
        if (proof["paired_training_gain_by_window"]!=changes
            or proof["equal_family_mean_NLL_gains"]!=family
            or abs(proof["bootstrap_tenth_percentile_NLL_gain"]-decile)>1e-12
            or proof["passes_frozen_heuristic_gate"]!=approved):
            errors.append(host+" corrupted independently reconstructed NLL bootstrap "+method)
        results[method]=(approved,decile,math.fsum(family.values())/4)
    viable=[m for m in METHODS[1:] if results[m][0]]
    chosen=max(viable,key=lambda m:(results[m][1],results[m][2],
             -METHODS.index(m))) if viable else "base_int6_pre"
    if chosen!=gate["chosen"]:
        errors.append(host+" wrong preregistered NLL bootstrap choice")
    return chosen


def margin_gate(report,original_choice,errors,host):
    gate=report["real_TASK_TRAIN_only_margin_gate"]
    d=gate["full_64_each_task_TRAIN_metrics"]
    if set(d)!=set(METHODS) or any(set(d[k])!=set(TASKS) for k in METHODS):
        errors.append(host+" incomplete four TRAIN task families")
        return "INVALID"
    if gate["NLL_bootstrap_preapproval"]!=original_choice:
        errors.append(host+" NLL prerequisite not preserved")
    base=d["base_int6_pre"]
    outcomes={}
    for method in METHODS[1:]:
        families={}
        correct={}
        for task in TASKS:
            x=base[task]["gold_logprob_margin_per_item"]
            y=d[method][task]["gold_logprob_margin_per_item"]
            if len(x)!=64 or len(y)!=64:
                errors.append(host+" incomplete paired 64 TRAIN margins "+task)
                return "INVALID"
            families[task]=math.fsum(b-a for a,b in zip(x,y))/64
            correct[task]=d[method][task]["correct"]-base[task]["correct"]
        r=random.Random(57)
        samples=[]
        for _ in range(1024):
            values=[]
            for _ in TASKS:
                task=TASKS[r.randrange(4)]
                segment=r.randrange(4)
                x=base[task]["gold_logprob_margin_per_item"]
                y=d[method][task]["gold_logprob_margin_per_item"]
                values.append(math.fsum(y[i]-x[i] for i in
                                range(segment*16,(segment+1)*16))/16)
            samples.append(math.fsum(values)/4)
        samples.sort()
        lower=samples[102]
        mean=math.fsum(families.values())/4
        approved=report["one_shot_train_only_gate"][
             "candidate_training_stability"][method]["passes_frozen_heuristic_gate"]
        ok=(approved and sum(v>0 for v in families.values())>=3
            and mean>=0.001 and min(families.values())>=-0.020
            and lower>=0.0005 and min(correct.values())>=-1)
        doc=gate["TRAIN_margin_evidence"][method]
        if (doc["family_mean_relative_answer_margin_gain"]!=families
            or doc["TRAIN_correct_change_by_family"]!=correct
            or abs(doc["family_block_bootstrap_lower_decile_gain"]-lower)>1e-12
            or abs(doc["equal_family_macro_margin_gain"]-mean)>1e-12
            or doc["NLL_preapproval"]!=approved or doc["eligible"]!=ok
            or doc["family_block_bootstrap_samples"]!=1024):
            errors.append(host+" independent paired margin gate disagrees "+method)
        outcomes[method]=(ok,lower,mean)
    accept=[m for m in METHODS[1:] if outcomes[m][0]]
    chosen=max(accept,key=lambda m:(outcomes[m][1],outcomes[m][2],
              -METHODS.index(m))) if accept else "base_int6_pre"
    if chosen!=gate["chosen"] or chosen!=report["selected_method"]:
        errors.append(host+" wrong frozen TRAIN-only task margin selection")
    return chosen


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True)
    p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    errors=[]
    for key in ("experiment","real_pretrained_model","source_revision",
        "source_checkpoint_fp32_SHA256","source_all_parameter_count",
        "WikiText_TRAIN_A_SHA256","eight_TRAIN_gate_token_SHA256",
        "external_task_source_revision_by_dataset",
        "TRAIN_task_definition_and_record_SHA256",
        "first_use_task_definition_and_record_SHA256"):
        if a[key]!=b[key]:
            errors.append("crosshost same trained model or text sources differ: "+key)
    if a["base_INT6_complete_source"]["sha256"]!=b[
         "base_INT6_complete_source"]["sha256"]:
        errors.append("physical full pretrained source CB46 differs across hosts")
    physical={}
    for method in METHODS[1:]:
        x=a["deterministic_row_budget"]["plans"][method]
        y=b["deterministic_row_budget"]["plans"][method]
        physical[method]=(x["bytes"]==y["bytes"] and
            x["sha256"]==y["sha256"] and
            x["selected_real_rows"]==y["selected_real_rows"] and
            x["module_counts"]==y["module_counts"])
        if not physical[method]:
            errors.append("nonreproducible correction CR47 "+method)
    evidence={}
    for host,report in (("ubuntu22",a),("ubuntu24",b)):
        base=report["base_INT6_complete_source"]
        if base["bytes"]!=10803661:
            errors.append(host+" original physically stored INT6 source changed")
        for method in METHODS[1:]:
            patch=report["deterministic_row_budget"]["plans"][method]
            if base["bytes"]+patch["bytes"]+15480>report[
              "deterministic_row_budget"]["max_all_model_plus_all_correction_bytes"]:
                errors.append(host+" full corrected rate above 6.5bpw "+method)
        first=nll_gate(report,errors,host)
        chosen=margin_gate(report,first,errors,host)
        teacher=report["original_teacher_first_use_task_accuracy"]
        arms=report["all_candidates_first_use_task_accuracy"]
        if set(teacher)!=set(TASKS) or set(arms)!=set(METHODS):
            errors.append(host+" missing blinded actual QA task results")
        for name,values in (("teacher",teacher),*arms.items()):
            if set(values)!=set(TASKS):
                errors.append(host+" incorrect evaluated task family "+name)
                continue
            for task,rec in values.items():
                n=128
                if rec["n"]!=n or len(rec["per_item_correct"])!=n or (
                    len(rec["gold_logprob_margin_per_item"])!=n
                    or len(rec["predicted_option_indices"])!=n
                    or rec["correct"]!=sum(rec["per_item_correct"])
                    or abs(rec["accuracy"]-rec["correct"]/n)>1e-12):
                    errors.append(host+" invalid actual answer count "+name+"/"+task)
        for arm,items in report["real_TASK_TRAIN_only_margin_gate"][
                "full_64_each_task_TRAIN_metrics"].items():
            for task,row in items.items():
                if (row["n"]!=64 or len(row["gold_logprob_margin_per_item"])!=64
                    or len(row["per_item_correct"])!=64 or
                    row["correct"]!=sum(row["per_item_correct"])):
                    errors.append(host+" invalid TRAIN task margin corpus "+arm+"/"+task)
        run=report["actual_full_model_warm_speed"]
        ratio=run["chosen_divided_by_teacher"]
        if not math.isfinite(ratio) or ratio<=0:
            errors.append(host+" missing measurable real full-model warm speed")
        for case in ("original_teacher","chosen"):
            for metric in ("prefill_ms","decode32_ms","total_prefill_and_decode32_ms"):
                vals=run[case][metric]["raw"]
                if len(vals)!=7 or any(not math.isfinite(t) or t<=0 for t in vals):
                    errors.append(host+" missing seven true warm repeats "+metric)
        evidence[host]={
            "TRAIN_bootstrap_choice":first,
            "TRAIN_margin_guard_chosen":chosen,
            "physical_selected_bpw":report["selected_actual_bpw"],
            "true_full_model_selected_div_original_latency":ratio,
            "passes_user_20pct_warm_slowdown_limit":run["passes_plus20pct_user_limit"],
            "heldout_128_item_correct":{task:{
                "teacher":teacher[task]["correct"],
                "plain_INT6":arms["base_int6_pre"][task]["correct"],
                "selected":arms[chosen][task]["correct"]}
                for task in TASKS}
        }
    if evidence["ubuntu22"]["TRAIN_margin_guard_chosen"]!=evidence[
        "ubuntu24"]["TRAIN_margin_guard_chosen"]:
        errors.append("different selected actual compressed model across independent CPUs")
    for method in ("original_teacher",*METHODS):
        for task in TASKS:
            key="original_teacher_first_use_task_accuracy" if method=="original_teacher" else "all_candidates_first_use_task_accuracy"
            x=a[key][task] if method=="original_teacher" else a[key][method][task]
            y=b[key][task] if method=="original_teacher" else b[key][method][task]
            if x["per_item_correct"]!=y["per_item_correct"]:
                errors.append("different physical-model heldout answer predictions "+method+"/"+task)
    result={"PASS":not errors,"errors":errors,
            "physical_sidecar_bitwise_identical":physical,
            "independent_host_measurements":evidence,
            "limitations":[
                "TRAIN bootstrap lower decile is a heuristic, not a population confidence bound.",
                "Tiny non-instruction-tuned Pythia14m is NOT actual production Jessica.",
                "128 heldout answers per task cannot establish broad task reliability.",
                "Physical INT6 disk model expands to FP32 hot RAM at cold load.",
                "Research-only branch; main, Render, Supabase, APK, ordinary Jessica chat untouched."]}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
