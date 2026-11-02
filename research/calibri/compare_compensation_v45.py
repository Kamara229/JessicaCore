#!/usr/bin/env python3
"""CALIBRI v45 independent cross-host evidence comparator.

Integrity and functional fidelity are hard gates; changes in PPL
or poor quality are recorded rather than silently filtered or failed.
"""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a",required=True)
    ap.add_argument("--b",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    a=json.loads(Path(args.a).read_text())
    b=json.loads(Path(args.b).read_text())
    errors=[]
    required=("ct33","int4","rank2","affine","hybrid")
    for key in ("model_revision","wiki_revision","train_A_tokens_SHA256",
                "train_B_disjoint_tokens_SHA256","source_MLP_SHA256",
                "eval_SHA256","eval_dataset_revisions","source_coefficients"):
        if a.get(key)!=b.get(key):
            errors.append("source/data mismatch: "+key)
    for tag in required:
        x=a["physical_files"][tag]
        y=b["physical_files"][tag]
        if x["real_bytes"]!=y["real_bytes"]:
            errors.append("physical byte mismatch: "+tag)
        if tag in ("ct33","int4") and x["SHA256"]!=y["SHA256"]:
            errors.append("original trained source physical SHA mismatch: "+tag)
        if tag not in ("ct33","int4"):
            for z in (x,y):
                if z["all_in_storage_bytes"]>=a["physical_files"]["int4"]["real_bytes"]:
                    errors.append("combined format exceeds INT4 bytes: "+tag)
    for host,d in (("ubuntu22",a),("ubuntu24",b)):
        bridge=d["streamed_runtime_fidelity"]
        if not bridge["mmap_path_never_materializes_full_W"] or bridge["max_decoded_W_scratch_bytes"]>32768:
            errors.append(host+" memory structural limit failed")
        for mode,delta in bridge["maximum_abs_delta_mmap_vs_dense_corrected"].items():
            if delta>2e-4:
                errors.append(host+" runtime residual fidelity failed "+mode)
        configs=d["full_model_PPL_after_dense_weight_restore_and_torch_hooks"]
        if set(configs)!={"teacher","base_CT33","pre_post_rank2",
                           "pre_post_affine","pre_post_hybrid","calibrated_INT4"}:
            errors.append(host+" missing quality control")
        if any(p["ppl"]<=0 for variant in configs.values() for p in variant.values()):
            errors.append(host+" invalid PPL")
    d_a=a["full_model_PPL_after_dense_weight_restore_and_torch_hooks"]
    d_b=b["full_model_PPL_after_dense_weight_restore_and_torch_hooks"]
    pp={}
    for method in d_a:
        pp[method]={}
        for corpus in d_a[method]:
            va=d_a[method][corpus]["ppl"]
            vb=d_b[method][corpus]["ppl"]
            pp[method][corpus]={"ubuntu22":va,"ubuntu24":vb,
                 "absolute_CPU_PPL_difference":abs(va-vb),
                 "host_mean_PPL":(va+vb)/2}
    outcomes={}
    for method in ("pre_post_rank2","pre_post_affine","pre_post_hybrid"):
        outcomes[method]={}
        for corpus in d_a[method]:
            a_delta=d_a[method][corpus]["ppl"]-d_a["base_CT33"][corpus]["ppl"]
            b_delta=d_b[method][corpus]["ppl"]-d_b["base_CT33"][corpus]["ppl"]
            outcomes[method][corpus]={
                "ubuntu22_delta_PPL_vs_CT33":a_delta,
                "ubuntu24_delta_PPL_vs_CT33":b_delta,
                "improved_both_hosts":a_delta<0 and b_delta<0,
                "worsened_either_host":a_delta>0 or b_delta>0}
    verdict={"integrity_and_functional_PASS":not errors,"errors":errors,
             "physical_bytes":{tag:a["physical_files"][tag]["real_bytes"]
                               for tag in required},
             "combined_bytes":{tag:a["physical_files"][tag].get("all_in_storage_bytes")
                               for tag in required if tag not in ("ct33","int4")},
             "correction_sidecar_crosshost_hash_equal":{
                tag:a["physical_files"][tag]["SHA256"]==b["physical_files"][tag]["SHA256"]
                for tag in ("rank2","affine","hybrid")},
             "PPL_by_CPU":pp,"pre_post_quality_vs_base":outcomes,
             "warnings":[
                 "Calibration sidecar SHA differences can arise from CPU SVD and FP16 rounding; compared sizes and quality without hiding differences.",
                 "Quality regression is a valid result and NOT a CI integrity failure.",
                 "Not a rate-matched comparison against 343840B baseline; report full sidecar bytes.",
                 "This is not end-to-end mmap full-model execution or Jessica integration."]}
    p=Path(args.output)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(verdict,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"PASS":not errors,"errors":errors,
                      "size":verdict["combined_bytes"],
                      "quality_directions":outcomes},ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
