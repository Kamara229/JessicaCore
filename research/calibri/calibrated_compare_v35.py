#!/usr/bin/env python3
"""Reproducibility gate: strict physical sizes and PPL evidence, negative results kept."""
from __future__ import annotations
import argparse,json
from pathlib import Path

INVARIANTS=(
    "experiment","model","model_revision","matrix_name",
    "matrix_source_weight_sha256","source_shape","datasets",
    "token_stream_sha256","tokens","physical_budget_bytes",
    "v34_baseline_bits_per_row","calibration_train_only",
    "higher_rate_INT4_bytes",
)
EXPECTED=("train_only_v34_rowwise_mixed3_4_control",
          "train_output_SSE_calibrated_same_row_bits")
CORPORA=("WikiText2_validation","TinyStories_validation")

def run():
    p=argparse.ArgumentParser()
    p.add_argument("--a",required=True);p.add_argument("--b",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(Path(args.a).read_text(encoding="utf-8"))
    b=json.loads(Path(args.b).read_text(encoding="utf-8"))
    issues=[k for k in INVARIANTS if a.get(k)!=b.get(k)]
    values={}
    for host,record in (("ubuntu_22_04",a),("ubuntu_24_04",b)):
        blob=record["physical_binary"]
        if set(blob)!=set(EXPECTED):
            issues.append(host+".missing_binary")
            continue
        for k in EXPECTED:
            if (blob[k]["actual_bytes"]>28672 or blob[k]["actual_bytes"]<=0
                    or abs(blob[k]["actual_bits_per_weight_including_metadata"]-
                           (8*blob[k]["actual_bytes"]/65536))>1e-10):
                issues.append(host+"."+k+".invalid_budget")
        if blob[EXPECTED[0]]["actual_bytes"]!=blob[EXPECTED[1]]["actual_bytes"]:
            issues.append(host+".calibrated_size_differs_from_baseline")
        s=record["calibration_train_only"]
        if (not s.get("bits_are_identical_per_row")
                or len(s["row_bitwidths"])!=128
                or sum(q==4 for q in s["row_bitwidths"])!=record["v34_baseline_bits_per_row"]["tiles_4bit"]
                or s["train_output_SSE_calibrated"]>s["train_output_SSE_baseline"]*(1+1e-7)):
            issues.append(host+".calibration_invariant_failed")
        vals=record["model_PPL_validation"]
        for key in ("source_fp32_original","v34_unmodified_3p5bpw_rowwise",
                    "v35_train_calibrated_3p5bpw_rowwise",
                    "INT4_OVER_BUDGET_high_rate_anchor"):
            if key not in vals or set(vals[key])!=set(CORPORA):
                issues.append(host+".missing_PPL_"+key)
        values[host]={
            "calibration_rows_changed":s["rows_changed"],
            "train_output_SSE_ratio":s["train_output_SSE_ratio"],
            "actual_bytes":{k:blob[k]["actual_bytes"] for k in EXPECTED},
            "binary_SHA256":{k:blob[k]["physical_SHA256"] for k in EXPECTED},
            "WikiText2_PPL":{k:v[CORPORA[0]]["ppl"] for k,v in vals.items()},
            "TinyStories_PPL":{k:v[CORPORA[1]]["ppl"] for k,v in vals.items()},
            "PPL_calibrated_minus_original_3p5bpw":{
                k:v["calibrated_minus_v34_baseline_PPL"]
                for k,v in record["quality_comparison"].items()},
        }
    if len(values)==2:
        if values["ubuntu_22_04"]["binary_SHA256"]!=values["ubuntu_24_04"]["binary_SHA256"]:
            issues.append("physical_serialized_sha_across_hosts_differs")
    identical_ppl=(len(values)==2 and
        values["ubuntu_22_04"]["WikiText2_PPL"]==values["ubuntu_24_04"]["WikiText2_PPL"] and
        values["ubuntu_22_04"]["TinyStories_PPL"]==values["ubuntu_24_04"]["TinyStories_PPL"])
    result={
        "experiment":"CALIBRI_v35_CPU_paired_pretrained_train_calibration",
        "source_and_physical_integrity_pass":not issues,
        "integrity_issues":issues,
        "physical_binary_SHA_equal_between_hosts":not any("sha" in k.lower() for k in issues),
        "PPL_exactly_equal_between_hosts":identical_ppl,
        "measurements":values,
        "limits":"One 128x512 projection, local train-only calibration, physical 3.5bpw rows; WikiText/TinyStories used previously in exploratory research; no full-model compression or packed inference throughput",
    }
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "integrity_pass":not issues,"PPL_equal":identical_ppl,
        "measurements":values,"issues":issues
    },ensure_ascii=False),flush=True)
    if issues:raise SystemExit(2)

if __name__=="__main__":
    run()
