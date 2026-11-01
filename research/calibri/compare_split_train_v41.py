#!/usr/bin/env python3
"""Cross-host integrity comparison for split TRAIN A/B, new SQuAD eval.

Source/physical SHA must match; negative generalization is a result, not
a CI error. No access to user data, production or evaluator decisions.
"""
import argparse,hashlib,json,math
from pathlib import Path

NAMES=("local_uncalibrated","local_trainA_calibrated",
       "joint_trainA_calibrated_UNGATED","joint_trainA_calibrated_GATED",
       "INT4_higher_rate_anchor")
EVAL=("original_teacher",)+NAMES
CORPORA=("WikiText2_development","NEW_SQuAD_validation")
FIXED=("experiment","source_model","source_revision","source_MLP_sha256",
       "real_source_MLP_parameter_count","datasets","TRAIN_A","TRAIN_B",
       "WikiText2_DEV_sha256","new_SQuAD_token_sha256","split_epoch")
def verify(label,data):
    errors=[]
    if (data.get("real_source_MLP_parameter_count")!=786432
        or len(data.get("source_MLP_sha256",{}))!=12
        or data.get("TRAIN_A",{}).get("start")!=0
        or data.get("TRAIN_A",{}).get("end_exclusive")!=2048
        or data.get("TRAIN_B",{}).get("start")!=2048
        or data.get("TRAIN_B",{}).get("end_exclusive")!=4096
        or data.get("TRAIN_A",{}).get("sha256")==
           data.get("TRAIN_B",{}).get("sha256")):
        errors.append(label+".invalid_real_source_or_train_split")
    split=data.get("split_epoch",{})
    if set(split)!={"TRAIN_A_selection_uses_A_only",
                   "TRAIN_B_checked_only_once_after_complete_TRAIN_A_search",
                   "new_SQuAD_loaded_after_binary_freeze"} or not all(split.values()):
        errors.append(label+".unfrozen_trainB_or_new_SQuAD")
    search=data.get("selection_trainA",{})
    if (search.get("number_of_module_matrices")!=12 or
        search.get("preregistered_max_rounds")!=2 or
        search.get("preregistered_max_probes_per_round")!=20 or
        search.get("actual_physical_binary_bytes")!=343840 or
        not search.get("no_development_or_test_quality_used_in_selection") or
        search.get("selected_joint_train_NLL",float("inf"))>
        search.get("original_baseline_joint_train_NLL",float("-inf"))+1e-8):
        errors.append(label+".invalid_trainA_joint_selection")
    floor=search.get("precision_row_bounds",{})
    if set(floor)!=set(data.get("source_MLP_sha256",{})):
        errors.append(label+".module_row_precision_bounds_missing")
    else:
        for n,v in floor.items():
            if not (v["floor_4bit"]<=v["selected_4bit"]<=v["ceiling_4bit"]):
                errors.append(label+"."+n+".floor_ceiling_violation")
    gate=data.get("internal_gate_trainB",{})
    if not (math.isfinite(gate.get("local_NLL",float("nan"))) and
            math.isfinite(gate.get("ungated_joint_NLL",float("nan"))) and
            gate.get("joint_improvement_required")==1e-6):
        errors.append(label+".unscored_trainB")
    else:
        must_accept=gate["ungated_joint_NLL"]<gate["local_NLL"]-1e-6
        if (gate.get("joint_accepted")!=must_accept or
            gate.get("decision")!=("accept_joint" if must_accept else "reject_joint_use_local")):
            errors.append(label+".wrong_one_shot_trainB_gate")
    archives=data.get("physical_MLP_bundles",{})
    if set(archives)!=set(NAMES):
        errors.append(label+".missing_physical_archives")
    else:
        for name,info in archives.items():
            size=info.get("real_disk_bytes",0)
            if (size!=343840 if name!="INT4_higher_rate_anchor" else size<=343840):
                errors.append(label+"."+name+".wrong_physical_bpw")
            if (len(info.get("real_sha256",""))!=64 or
                info.get("bpw_for_12_MLP_only")!=8*size/786432 or
                info.get("module_count")!=12):
                errors.append(label+"."+name+".invalid_physical_sha")
        if gate.get("decision") in ("accept_joint","reject_joint_use_local"):
            winner="joint_trainA_calibrated_UNGATED" if gate["joint_accepted"] else "local_trainA_calibrated"
            if archives.get(winner,{}).get("real_sha256") != archives.get("joint_trainA_calibrated_GATED",{}).get("real_sha256"):
                errors.append(label+".physical_gate_does_not_match_internal_holdout")
    metrics=data.get("full_model_decoded_PPL",{})
    if set(metrics)!=set(EVAL):
        errors.append(label+".missing_teacher_or_quantized_variant")
    else:
        for v,scores in metrics.items():
            if set(scores)!=set(CORPORA) or any(
                not math.isfinite(x.get("ppl",float("nan"))) or
                x.get("ppl",0)<=0 or x.get("tokens",0)!=(
                    1024//128*(128-1) if c=="WikiText2_development" else 2048//128*(128-1))
                for c,x in scores.items()):
                errors.append(label+"."+v+".bad_PPL_measurement")
    env=data.get("environment",{})
    if (env.get("cpu_capability")!="AVX2" or
        env.get("onednn_max_requested")!="AVX2" or
        env.get("threads")!=2 or not env.get("determinism")):
        errors.append(label+".nonmatching_cpu_dispatch")
    return errors

def summary(data):
    return {
        "real_disk_bytes":{n:v["real_disk_bytes"] for n,v in data["physical_MLP_bundles"].items()},
        "real_disk_sha":{n:v["real_sha256"] for n,v in data["physical_MLP_bundles"].items()},
        "TRAIN_A_NLL_local":data["selection_trainA"]["original_baseline_joint_train_NLL"],
        "TRAIN_A_NLL_joint":data["selection_trainA"]["selected_joint_train_NLL"],
        "TRAIN_B_NLL_local":data["internal_gate_trainB"]["local_NLL"],
        "TRAIN_B_NLL_joint":data["internal_gate_trainB"]["ungated_joint_NLL"],
        "one_shot_trainB_decision":data["internal_gate_trainB"]["decision"],
        "PPL":{n:{c:s["ppl"] for c,s in v.items()}
               for n,v in data["full_model_decoded_PPL"].items()},
        "deltas":data["quality_delta_against_equal_rate_local"],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True)
    p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    fail=verify("ubuntu22",a)+verify("ubuntu24",b)
    for key in FIXED:
        if a.get(key)!=b.get(key): fail.append(key+".source_not_identical_across_CPUs")
    x,y=summary(a),summary(b)
    if x["real_disk_bytes"]!=y["real_disk_bytes"] or x["real_disk_sha"]!=y["real_disk_sha"]:
        fail.append("crosshost_file_sha_mismatch")
    if x["one_shot_trainB_decision"]!=y["one_shot_trainB_decision"]:
        fail.append("crosshost_trainB_acceptance_decision_mismatch")
    eq=x["PPL"]==y["PPL"]
    result={
        "experiment":"CALIBRI_v41_split_train_selection_A_gate_B_new_SQuAD_cross_cpu",
        "integrity_pass":not fail,
        "physical_source_binaries_bitwise_equal":x["real_disk_sha"]==y["real_disk_sha"],
        "PPL_bitwise_identical_across_hosts":eq,
        "errors":fail,
        "outcomes":{"ubuntu22":x,"ubuntu24":y},
        "limits":"One 2048-token untouched-SQuAD exploratory evaluation; 12 MLP-only physical files, decoded dense PPL, no runtime gain proof.",
    }
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if fail:raise SystemExit(2)

if __name__=="__main__":main()
