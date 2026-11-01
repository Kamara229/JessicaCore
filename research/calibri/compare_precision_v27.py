#!/usr/bin/env python3
"""Compare pinned pretrained MoE logits and exact routed expert masks across hosts.

This is a measurement, not an unconditional CI pass on scientific precision.
Only source-invariant/probe-integrity failures make exit status nonzero.
"""
import argparse
import hashlib
import json
import pathlib
import numpy as np

CORPORA=("WikiText2_validation", "TinyStories_validation")
MODES=("bf16_mkldnn","fp32_no_mkldnn")
INVARIANTS=("experiment", "source_model", "model_revision",
            "dataset_revisions","source_token_sha256",
            "original_BF16_weight_sha256","torch_version",
            "transformers_version","torch_threads",
            "deterministic_algorithms",
            "router_jitter_forced_zero",
            "checkpoint_loading_missing_unexpected_mismatched")


def compare(a,b):
    bad_source=[key for key in INVARIANTS if a.get(key)!=b.get(key)]
    details={}
    for mode in MODES:
        details[mode]={}
        for corpus in CORPORA:
            left=a["modes"][mode][corpus]
            right=b["modes"][mode][corpus]
            if left["sampled_logit_positions"] != right["sampled_logit_positions"]:
                bad_source.append(f"{mode}.{corpus}.probe_positions")
                continue
            if (left["vocab"]!=right["vocab"] or
                left["tokens_evaluated"]!=right["tokens_evaluated"]):
                bad_source.append(f"{mode}.{corpus}.data_shape")
                continue
            x=np.asarray(left["sampled_token_logits"],dtype=np.float64)
            y=np.asarray(right["sampled_token_logits"],dtype=np.float64)
            mask1=np.asarray(left["exact_eight_expert_mask_per_token"],dtype=np.int8)
            mask2=np.asarray(right["exact_eight_expert_mask_per_token"],dtype=np.int8)
            id1=np.asarray(left["full_vocab_top1_token_id"],dtype=np.int64)
            id2=np.asarray(right["full_vocab_top1_token_id"],dtype=np.int64)
            norm1=np.asarray(left["per_token_full_logit_l2"],dtype=np.float64)
            norm2=np.asarray(right["per_token_full_logit_l2"],dtype=np.float64)
            if (x.shape!=y.shape or mask1.shape!=mask2.shape or
                id1.shape!=id2.shape or norm1.shape!=norm2.shape or
                x.ndim!=2 or mask1.ndim!=2 or mask1.shape[1]!=8 or
                x.shape[0]!=id1.size or mask1.shape[0]!=id1.size or
                not np.isfinite(x).all() or not np.isfinite(y).all()):
                bad_source.append(f"{mode}.{corpus}.invalid_or_nonfinite_data")
                continue
            delta=np.abs(x-y)
            router_equal=np.all(mask1==mask2,axis=1)
            details[mode][corpus]={
                "tokens":int(id1.size),
                "sampled_logit_values":int(x.size),
                "full_logits_bitwise_equal":left["full_logit_sha256"]==right["full_logit_sha256"],
                "router_masks_bitwise_equal":left["router_mask_sha256"]==right["router_mask_sha256"],
                "sampled_logits_max_abs_delta":float(np.max(delta)),
                "sampled_logits_rmse":float(np.sqrt(np.mean(delta**2))),
                "sampled_logits_max_relative_to_1_or_reference":float(
                    np.max(delta / np.maximum(1.0,np.abs(x)))),
                "full_logit_norm_max_abs_delta":float(np.max(np.abs(norm1-norm2))),
                "full_vocab_top1_token_agreement":float(np.mean(id1==id2)),
                "next_router_exact_per_token_agreement":float(np.mean(router_equal)),
                "next_router_tokens_changed":int(np.count_nonzero(~router_equal)),
                "run_22_04_seconds_diagnostic":left.get("elapsed_seconds_diagnostic"),
                "run_24_04_seconds_diagnostic":right.get("elapsed_seconds_diagnostic"),
            }
    report={
        "experiment":"CALIBRI_v27_crosshost_BF16_vs_FP32_diagnostic",
        "source_invariants_pass":not bad_source,
        "source_or_measurement_errors":bad_source,
        "cpu_22_04":a.get("cpu_brand"),
        "cpu_24_04":b.get("cpu_brand"),
        "cpu_capability_22_04":a.get("torch_cpu_capability"),
        "cpu_capability_24_04":b.get("torch_cpu_capability"),
        "results":details,
        "fp32_crosshost_exact_full_logits":
            all(details.get("fp32_no_mkldnn",{}).get(c,{}).get("full_logits_bitwise_equal",False) for c in CORPORA),
        "bf16_crosshost_exact_full_logits":
            all(details.get("bf16_mkldnn",{}).get(c,{}).get("full_logits_bitwise_equal",False) for c in CORPORA),
        "interpretation":"This is a diagnostic: FP32 arithmetic on BF16-source weights versus native BF16; no PPL, physical compression or native packed speed measured. Full logit hashes are exact; absolute deltas apply only to the 24 predeclared vocab probes per token.",
    }
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--first",required=True)
    p.add_argument("--second",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    a=json.loads(pathlib.Path(args.first).read_text(encoding="utf-8"))
    b=json.loads(pathlib.Path(args.second).read_text(encoding="utf-8"))
    report=compare(a,b)
    report["source_json_sha256"]={
        "ubuntu_22_04":hashlib.sha256(pathlib.Path(args.first).read_bytes()).hexdigest(),
        "ubuntu_24_04":hashlib.sha256(pathlib.Path(args.second).read_bytes()).hexdigest(),
    }
    dst=pathlib.Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    summary={m:{c:{
        "max_abs":v.get("sampled_logits_max_abs_delta"),
        "router_agreement":v.get("next_router_exact_per_token_agreement"),
        "full_hash_same":v.get("full_logits_bitwise_equal")}
        for c,v in vals.items()} for m,vals in report["results"].items()}
    print(json.dumps({"source_invariants_pass":report["source_invariants_pass"],
                      "differences":report["source_or_measurement_errors"],
                      "precision_comparison":summary},ensure_ascii=False),flush=True)
    if not report["source_invariants_pass"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
