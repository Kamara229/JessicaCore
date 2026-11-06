"""CALIBRI v207: retrospective read-only original 135M frozen public DEV diagnostics.

Compare already SEALED v204 (16 steps) and v206 (48 steps) using the same
locked original FP32 answers, cases and grader, and the unchanged v186 base.
Never use PUBLIC DEV to update a source, choose a checkpoint, train, or select
training hyperparameters. Distinguish ungraded from machine-recognized wrong.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from v205.read_only import _grades, _read


def audit_steps(data, expected):
    rows = data["steps"]
    if data["actual_pilot_steps"] != expected or len(rows) != expected:
        raise AssertionError("Unexpected TRAIN pilot length")
    if [r["step"] for r in rows] != list(range(1, expected + 1)):
        raise AssertionError("TRAIN step numbering drift")
    if len({r["operator"] for r in rows}) != expected:
        raise AssertionError("Missing/duplicate physical packed matrix")
    kept = [r["step"] for r in rows if r["source_step_accepted"]]
    if len(kept) != data["accepted_steps_before_full_TRAIN_guard"]:
        raise AssertionError("TRAIN accepted step count inconsistent")
    rejects = Counter()
    for row in rows:
        if row["source_step_accepted"]:
            continue
        if not row["TRAIN_guard"]["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"]:
            rejects["sampled_8_first"] += 1
        full = row.get("per_step_TRAIN_full_24_guard")
        if full is not None and not full["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"]:
            rejects["global_24_first"] += 1
        if not row["per_step_TRAIN_sequence_guard"]["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"]:
            rejects["global_8_full_sequences"] += 1
        if not row["TRAIN_no_more_than_1percent_loss_increase"]:
            rejects["TRAIN_mix_loss"] += 1
    return {"accepted": kept, "rejected": [r["step"] for r in rows if not r["source_step_accepted"]],
            "operators": [r["operator"] for r in rows],
            "rejected_guard_reasons_overlapping": dict(sorted(rejects.items())),
            "final_full_guard_rollback": bool(data["all_pilot_mutations_rolled_back_on_full_TRAIN_fail"])}


def analyze(qa204, qa206, a204, a206, s204, s206):
    if qa204["cases"] != qa206["cases"]:
        raise AssertionError("Frozen public QA identities/questions/golds differ")
    if qa204["preexisting_v192_frozen_grader_sha256"] != qa206["preexisting_v192_frozen_grader_sha256"]:
        raise AssertionError("Locked original machine grader changed")
    fp204, fp206 = (_grades(q,"FP32_existing_135M") for q in (qa204,qa206))
    if fp204 != fp206:
        raise AssertionError("Original FP32 reference changed")
    cb204, cb206 = (_grades(q,"CB186") for q in (qa204,qa206))
    if cb204 != cb206:
        raise AssertionError("Starting paid v186 native QA changed")
    x, y = _grades(qa204, "GT204"), _grades(qa206, "GT206")
    if set(x) != set(y) or set(x) != set(fp204):
        raise AssertionError("Case ID mismatch")
    train204,train206 = audit_steps(a204,16),audit_steps(a206,48)
    if train204["operators"] != train206["operators"][:16]:
        raise AssertionError("First 16 canonical paid source operators not comparable")
    if not s204["completed"] or not s206["completed"]:
        raise AssertionError("Native source result incomplete")
    for s in (s204,s206):
        if s["whole_original_source_bpp"] > .7:
            raise AssertionError("Physical original 135M source exceeds bpp target")
        if s["native_CPU_speed_ratio_CB59"] < 0:
            raise AssertionError("Native CPU speed ratio invalid")
        if s["original_FP32_16_machine_correct"] != 6:
            raise AssertionError("Original FP32 frozen baseline differs")
    if s204["source_SHA256"] == s206["source_SHA256"]:
        raise AssertionError("Expected distinct genuinely paid source outcomes")
    transitions,domains=Counter(),defaultdict(lambda:Counter())
    details=[]
    for i in sorted(fp204):
        old,new=x[i],y[i]
        if old["domain"]!=new["domain"] or old["domain"]!=fp204[i]["domain"]:
            raise AssertionError("Frozen QA domain drift")
        k=old["machine_status"]+" -> "+new["machine_status"]
        transitions[k]+=1
        domain=old["domain"]
        domains[domain]["v204_"+old["machine_status"]]+=1
        domains[domain]["v206_"+new["machine_status"]]+=1
        details.append({"case_id":i,"domain":domain,
            "FP32_status":fp204[i]["machine_status"],
            "v186_status":cb204[i]["machine_status"],
            "v204_status":old["machine_status"],
            "v206_status":new["machine_status"],
            "v204_raw":old["raw"], "v206_raw":new["raw"],
            "full_answer_changed":old["raw"]!=new["raw"]})
    if sum(transitions.values())!=16:
        raise AssertionError("Incomplete immutable comparison")
    return {"experiment":"CALIBRI_v207_READ_ONLY_retrospective_public_DEV",
        "read_only_NO_model_inference_or_training":True,
        "PUBLIC_DEV_NOT_used_to_accept_changes_choose_checkpoints_or_hyperparameters":True,
        "original_pretrained_parameters":134515008,
        "frozen_16_cases":16,
        "FP32_confirmed_correct":sum(z["machine_status"]=="machine_correct" for z in fp204.values()),
        "v186_confirmed_correct":sum(z["machine_status"]=="machine_correct" for z in cb204.values()),
        "v204_confirmed_correct":sum(z["machine_status"]=="machine_correct" for z in x.values()),
        "v206_confirmed_correct":sum(z["machine_status"]=="machine_correct" for z in y.values()),
        "v204_to_v206_machine_confirmed_delta":sum(z["machine_status"]=="machine_correct" for z in y.values())-sum(z["machine_status"]=="machine_correct" for z in x.values()),
        "status_transitions_ungraded_not_equal_wrong":dict(sorted(transitions.items())),
        "per_domain_statuses":{k:dict(sorted(v.items())) for k,v in sorted(domains.items())},
        "verbatim_sealed_full_answers":details,
        "v204_TRAIN":train204,"v206_TRAIN":train206,
        "v204_source_SHA256":s204["source_SHA256"],
        "v206_source_SHA256":s206["source_SHA256"],
        "v204_physical_bpp":s204["whole_original_source_bpp"],
        "v206_physical_bpp":s206["whole_original_source_bpp"],
        "v204_native_CPU_ratio":s204["native_CPU_speed_ratio_CB59"],
        "v206_native_CPU_ratio":s206["native_CPU_speed_ratio_CB59"],
        "NO_LIVE_Jessica_main_Render_Supabase_APK":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("v204_artifact")
    parser.add_argument("v206_artifact")
    parser.add_argument("out")
    a=parser.parse_args()
    p204,p206=Path(a.v204_artifact),Path(a.v206_artifact)
    report=analyze(
        _read(p204/"REAL_GREEN_TUNNEL_new_public_DEV_16_full_QA.json"),
        _read(p206/"REAL_GREEN_TUNNEL_new_public_DEV_16_full_QA.json"),
        _read(p204/"ACTUAL_guarded_TRAIN_16_step_audit.json"),
        _read(p206/"ACTUAL_guarded_TRAIN_48_step_audit.json"),
        _read(p204/"v204_real_NATIVE_SOURCE_QA_summary.json"),
        _read(p206/"v206_real_NATIVE_SOURCE_QA_summary.json"))
    output=Path(a.out)
    output.mkdir(parents=True,exist_ok=True)
    (output/"SEALED_v204_v206_READ_ONLY_forensics.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"stage":"v207_read_only_forensics_completed",
         "FP32":report["FP32_confirmed_correct"],
         "v204":report["v204_confirmed_correct"],
         "v206":report["v206_confirmed_correct"],
         "transitions":report["status_transitions_ungraded_not_equal_wrong"],
         "TRAIN_v204_accepted":len(report["v204_TRAIN"]["accepted"]),
         "TRAIN_v206_accepted":len(report["v206_TRAIN"]["accepted"]),
         "TRAIN_v206_rejections":report["v206_TRAIN"]["rejected_guard_reasons_overlapping"],
         "DEV_never_guides_weight_training":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
