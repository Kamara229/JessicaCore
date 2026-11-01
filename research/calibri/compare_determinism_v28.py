#!/usr/bin/env python3
"""Compare independently collected CALIBRI v28 BF16/MKLDNN on/off records."""
import argparse, json, pathlib
from determinism_factor_v28 import CORPORA, MODES, compare

INVARIANTS=(
    "experiment","source_model","model_revision","dataset_revisions",
    "source_token_sha256","original_BF16_weight_sha256",
    "torch_version","transformers_version","cpu_arch","torch_threads",
    "compute_dtype","mkldnn_enabled","router_jitter_forced_zero"
)

def run(args):
    a=json.loads(pathlib.Path(args.first).read_text(encoding="utf-8"))
    b=json.loads(pathlib.Path(args.second).read_text(encoding="utf-8"))
    mismatches=[k for k in INVARIANTS if a.get(k)!=b.get(k)]
    results={}
    for mode in MODES:
        results[mode]={}
        for corpus in CORPORA:
            left=a["modes"][mode][corpus]
            right=b["modes"][mode][corpus]
            if left["sampled_logit_positions"]!=right["sampled_logit_positions"] or left["vocab"]!=right["vocab"]:
                mismatches.append(f"{mode}.{corpus}.probe_geometry")
                continue
            results[mode][corpus]=compare(left,right)
    verdict={
        "experiment":"CALIBRI_v28_crosshost_determinism_flag_factor",
        "source_invariants_pass":not mismatches,
        "invariant_mismatches":mismatches,
        "cpu_ubuntu_22_04":a.get("cpu_brand"),
        "cpu_ubuntu_24_04":b.get("cpu_brand"),
        "crosshost":results,
        "within_host_22_04":a.get("within_host_pairwise"),
        "within_host_24_04":b.get("within_host_pairwise"),
        "interpretation":"Only deterministic-algorithms flag changed, BF16+MKLDNN+zero-jitter fixed. No PPL, physical bytes or speed claims.",
    }
    path=pathlib.Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(verdict,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    compact={mode:{corpus:{
        "full_logits_same":r.get("full_logits_exact_sha256_equal"),
        "max_abs_probe":r.get("sampled_logits_max_abs_delta"),
        "router_agreement":r.get("next_router_exact_per_token_agreement")
    } for corpus,r in group.items()} for mode,group in results.items()}
    print(json.dumps({"invariants":not mismatches,"crosshost":compact},ensure_ascii=False),flush=True)
    if mismatches:raise SystemExit(2)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--first",required=True)
    p.add_argument("--second",required=True)
    p.add_argument("--output",required=True)
    run(p.parse_args())
