"""Read-only root-cause audit of completed v206->v212 on frozen PUBLIC DEV.

This is retrospective diagnostics ONLY. The public DEV cases are never used
to choose v213 training samples, interventions, checkpoints or hyperparameters.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(source):
    source = Path(source)
    qa = json.loads((source/"REAL_GREEN_TUNNEL_new_public_DEV_16_full_QA.json").read_text())
    audit = json.loads((source/"ACTUAL_guarded_TRAIN_24_step_audit.json").read_text())
    seal = json.loads((source/"SEALED_paid_source_before_untouched_QA.json").read_text())
    if not qa["complete"] or qa["scores"]["CB206"]["n"] != 16 or qa["scores"]["GT212"]["n"] != 16:
        raise AssertionError("Missing complete sealed v206/v212 QA")
    aliases = ("FP32_existing_135M", "CB206", "GT212")
    by_model = {name: {x["case_id"]: x for x in qa["scores"][name]["verbatim_answers_and_grades"]} for name in aliases}
    cases = {x["case_id"]: x for x in qa["cases"]}
    if any(set(d) != set(cases) for d in by_model.values()) or len(cases) != 16:
        raise AssertionError("Mismatched QA IDs")
    counters = Counter()
    rows = []
    for id, case in sorted(cases.items()):
        original, baseline, updated = (by_model[k][id] for k in aliases)
        a,b,c=(x["machine_status"] for x in (original,baseline,updated))
        if b=="machine_correct" and c!="machine_correct":
            counters["v206_correct_lost_in_v212"] += 1
        if b!="machine_correct" and c=="machine_correct":
            counters["v212_correct_gained_over_v206"] += 1
        if a=="machine_correct" and b!="machine_correct":
            counters["FP32_correct_lost_in_v206"] += 1
        if a=="machine_correct" and c!="machine_correct":
            counters["FP32_correct_lost_in_v212"] += 1
        if a!="machine_correct" and b=="machine_correct":
            counters["v206_correct_but_FP32_not_correct"] += 1
        if b=="ungraded" or c=="ungraded":
            counters["baseline_or_candidate_ungraded"] += 1
        rows.append({
            "case_id":id,"domain":case["domain"],
            "question":case["question"],"gold":case["gold"],
            "FP32":{k:original[k] for k in ("machine_status","candidate","raw")},
            "v206":{k:baseline[k] for k in ("machine_status","candidate","raw")},
            "v212":{k:updated[k] for k in ("machine_status","candidate","raw")},
            "v206_to_v212_status":b+" -> "+c
        })
    accepted=[{"step":s["step"],"operator":s["operator"],"candidate_5bit_groups":s["candidate_5bit_groups"],"train_loss_before":s["TRAIN_mix_before"],"train_loss_after":s["TRAIN_mix_after"]} for s in audit["steps"] if s["source_step_accepted"]]
    result={
        "case_count":len(rows),
        "v206_machine_correct":qa["scores"]["CB206"]["machine_confirmed"],
        "v212_machine_correct":qa["scores"]["GT212"]["machine_confirmed"],
        "FP32_machine_correct":qa["scores"]["FP32_existing_135M"]["machine_confirmed"],
        "transitions":dict(counters),
        "model_status_matrix":rows,
        "accepted_operators_on_TRAIN_only":accepted,
        "source_bpp":seal["complete_bpp"],
        "source_sha256":seal["sha256"],
        "PUBLIC_DEV_RETROSPECTIVE_ONLY_NOT_FOR_FUTURE_TRAIN_SELECTION":True,
        "NO_LIVE_CHANGES":True
    }
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("v212_artifact",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    report=summarize(args.v212_artifact)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("model_status_matrix","accepted_operators_on_TRAIN_only")},ensure_ascii=False),flush=True)
    for case in report["model_status_matrix"]:
        print(json.dumps({"case":case["case_id"],"domain":case["domain"],"question":case["question"][:90],
                          "gold":case["gold"],"FP32":case["FP32"]["machine_status"],"v206":case["v206"]["machine_status"],
                          "v212":case["v212"]["machine_status"],
                          "v206_output":case["v206"]["raw"][:130],"v212_output":case["v212"]["raw"][:130]},ensure_ascii=False),flush=True)
