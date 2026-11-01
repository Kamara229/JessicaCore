#!/usr/bin/env python3
"""Compare independently collected CALIBRI v28 BF16/MKLDNN on/off records."""
import argparse, json, pathlib
import numpy as np

CORPORA=("WikiText2_validation","TinyStories_validation")
MODES=("deterministic_on_1","deterministic_off","deterministic_on_2")

def compare(a,b):
    x=np.asarray(a["sampled_token_logits"],dtype=np.float64)
    y=np.asarray(b["sampled_token_logits"],dtype=np.float64)
    m1=np.asarray(a["exact_eight_expert_mask_per_token"],dtype=np.int8)
    m2=np.asarray(b["exact_eight_expert_mask_per_token"],dtype=np.int8)
    top1=np.asarray(a["full_vocab_top1_token_id"],dtype=np.int64)
    top2=np.asarray(b["full_vocab_top1_token_id"],dtype=np.int64)
    if (x.ndim!=2 or x.shape!=y.shape or not np.isfinite(x).all()
        or not np.isfinite(y).all() or m1.ndim!=2 or m1.shape!=m2.shape
        or m1.shape[1]!=8 or top1.shape!=top2.shape
        or x.shape[0]!=top1.size or m1.shape[0]!=top1.size):
        raise ValueError("incompatible, nonfinite or absent model diagnostic samples")
    delta=x-y
    return {
        "full_logits_exact_sha256_equal":a["full_logit_sha256"]==b["full_logit_sha256"],
        "router_masks_exact_sha256_equal":a["router_mask_sha256"]==b["router_mask_sha256"],
        "sampled_logits_max_abs_delta":float(np.max(np.abs(delta))),
        "sampled_logits_rmse":float(np.sqrt(np.mean(delta**2))),
        "full_vocab_top1_agreement":float(np.mean(top1==top2)),
        "next_router_exact_per_token_agreement":float(np.mean(np.all(m1==m2,axis=1))),
    }

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
