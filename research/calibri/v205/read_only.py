"""CALIBRI v205: READ-ONLY comparison of two already sealed 135M QA artifacts.

Forensics on PUBLIC DEV after sealing. This file contains NO optimizer, codebook
writer, model inference, sampling, new test generation, or model selection logic.
DEV answers are NOT permission to tune TRAIN or to claim independent blind TEST.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

STATUSES = ("machine_correct", "machine_wrong", "ungraded")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _grades(report, model):
    scores = report["scores"][model]
    raw = scores["verbatim_answers_and_grades"]
    result = {}
    for row in raw:
        case_id = int(row["case_id"])
        if case_id in result or row["machine_status"] not in STATUSES:
            raise AssertionError("Duplicated ID or unsupported locked-machine grade")
        result[case_id] = row
    if len(result) != 16 or scores["n"] != 16:
        raise AssertionError("Expected exactly 16 sealed PUBLIC DEV answers")
    count = Counter(r["machine_status"] for r in result.values())
    if count["machine_correct"] != scores["machine_confirmed"]:
        raise AssertionError("Stored locked-grader correct count mismatched raw grades")
    if count["machine_wrong"] != scores["machine_recognized_wrong"]:
        raise AssertionError("Stored locked-grader wrong count mismatched raw grades")
    if count["ungraded"] != scores["ungraded_not_known_wrong"]:
        raise AssertionError("Stored locked-grader ungraded count mismatched raw grades")
    return result


def _step_audit(audit):
    steps = audit["steps"]
    if len(steps) != 16 or audit["actual_pilot_steps"] != 16:
        raise AssertionError("Expected same frozen 16-step original 135M operator pilot")
    ids = [s["step"] for s in steps]
    if ids != list(range(1, 17)):
        raise AssertionError("Original operator steps are missing/reordered")
    accept = [s["step"] for s in steps if s["source_step_accepted"]]
    if len(accept) != audit["accepted_steps_before_full_TRAIN_guard"]:
        raise AssertionError("Reported accepted TRAIN steps inconsistent")
    return {"accepted_steps": accept,
            "rejected_steps": [s["step"] for s in steps if not s["source_step_accepted"]],
            "per_step_operator": [s["operator"] for s in steps],
            "final_full_TRAIN_rollback": audit["all_pilot_mutations_rolled_back_on_full_TRAIN_fail"]}


def analyze(qa200, qa204, train200, train204, summary200, summary204):
    if qa200["cases"] != qa204["cases"]:
        raise AssertionError("QA question/gold/case identities drift across sealed runs")
    for k in ("preexisting_v192_frozen_grader_sha256",):
        if qa200[k] != qa204[k]:
            raise AssertionError("Locked original grader fingerprint drift")
    original200 = _grades(qa200, "FP32_existing_135M")
    original204 = _grades(qa204, "FP32_existing_135M")
    if original200 != original204:
        raise AssertionError("Independent original FP32 public DEV answers drift")
    before,after = _grades(qa200,"GT200"),_grades(qa204,"GT204")
    baseline200,baseline204=_grades(qa200,"CB186"),_grades(qa204,"CB186")
    if baseline200 != baseline204:
        raise AssertionError("Immutable v186 original packed answers drift")
    expected=set(original200)
    if set(before)!=expected or set(after)!=expected:
        raise AssertionError("Candidate comparison must use identical frozen case IDs")
    trans=Counter(); rows=[]
    for i in sorted(expected):
        a,b=before[i],after[i]
        from_status,to_status=a["machine_status"],b["machine_status"]
        trans[from_status+" -> "+to_status]+=1
        rows.append({"case_id":i,"domain":a["domain"],
                     "FP32_machine_status":original200[i]["machine_status"],
                     "v186_machine_status":baseline200[i]["machine_status"],
                     "v200_machine_status":from_status,
                     "v204_machine_status":to_status,
                     "v200_actual_text":a["raw"],"v204_actual_text":b["raw"],
                     "answer_text_changed":a["raw"]!=b["raw"]})
    guards200,guards204=_step_audit(train200),_step_audit(train204)
    if guards200["per_step_operator"] != guards204["per_step_operator"]:
        raise AssertionError("Different frozen 16-source operator order: cannot infer effects")
    if not summary200["completed"] or not summary204["completed"]:
        raise AssertionError("Missing complete independent native QA certification")
    if summary200["source_SHA256"]==summary204["source_SHA256"]:
        raise AssertionError("Expected distinct physically sealed v200/v204 sources")
    bpp200=summary200["whole_original_source_bpp"]
    bpp204=summary204["whole_original_source_bpp"]
    if bpp200>0.7 or bpp204>0.7:
        raise AssertionError("Forensic source violates physical <=0.7 bpp policy")
    return {
        "experiment":"v205_read_only_sealed_PUBLIC_DEV_diagnostics",
        "read_only_no_weights_training_or_inference":True,
        "public_DEV_is_not_new_blind_test":True,
        "public_DEV_must_not_be_used_as_optimizer_or_checkpoint_selector":True,
        "frozen_case_count":16,
        "original_FP32_machine_correct":sum(x["machine_status"]=="machine_correct" for x in original200.values()),
        "v186_machine_correct":sum(x["machine_status"]=="machine_correct" for x in baseline200.values()),
        "v200_machine_correct":sum(x["machine_status"]=="machine_correct" for x in before.values()),
        "v204_machine_correct":sum(x["machine_status"]=="machine_correct" for x in after.values()),
        "v200_v204_status_transitions":dict(sorted(trans.items())),
        "v200_to_v204_machine_confirmed_change":sum(x["machine_status"]=="machine_correct" for x in after.values())-sum(x["machine_status"]=="machine_correct" for x in before.values()),
        "full_verbatim_answer_diffs":rows,
        "TRAIN_only_v200_guards":guards200,
        "TRAIN_only_v204_guards":guards204,
        "v200_source_SHA256":summary200["source_SHA256"],
        "v204_source_SHA256":summary204["source_SHA256"],
        "v200_physical_bpp":bpp200,"v204_physical_bpp":bpp204,
        "v200_CPU_speed_ratio_CB59":summary200["native_CPU_speed_ratio_CB59"],
        "v204_CPU_speed_ratio_CB59":summary204["native_CPU_speed_ratio_CB59"],
        "next_step_not_automatically_selected_by_DEV":True,
        "NO_Jessica_LIVE_main_Render_Supabase_APK":True
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("v200_artifact")
    p.add_argument("v204_artifact")
    p.add_argument("out")
    args=p.parse_args()
    left,right=Path(args.v200_artifact),Path(args.v204_artifact)
    result=analyze(
        _read(left/"REAL_GREEN_TUNNEL_new_public_DEV_16_full_QA.json"),
        _read(right/"REAL_GREEN_TUNNEL_new_public_DEV_16_full_QA.json"),
        _read(left/"ACTUAL_guarded_TRAIN_16_step_audit.json"),
        _read(right/"ACTUAL_guarded_TRAIN_16_step_audit.json"),
        _read(left/"v200_real_NATIVE_SOURCE_QA_summary.json"),
        _read(right/"v204_real_NATIVE_SOURCE_QA_summary.json"),
    )
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"SEALED_v200_v204_READ_ONLY_forensic.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"stage":"v205_forensics_completed",
      "FP32_16":result["original_FP32_machine_correct"],
      "v200_16":result["v200_machine_correct"],
      "v204_16":result["v204_machine_correct"],
      "status_transitions":result["v200_v204_status_transitions"],
      "TRAIN_v200_accepted":result["TRAIN_only_v200_guards"]["accepted_steps"],
      "TRAIN_v204_accepted":result["TRAIN_only_v204_guards"]["accepted_steps"],
      "PUBLIC_DEV_never_used_to_accept_training":True
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
