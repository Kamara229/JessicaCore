#!/usr/bin/env python3
"""OscAR CLI: independent source FP32 checkpoint -> one CB59/CB61 file.

Accept local COMPLETE safetensors or an explicitly named pinned HF pretrained
checkpoint (trust_remote_code=False). The CLI does NOT claim output answers
are preserved before a real paired inference and heldout independent QA.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from oscar.encoder import encode_fp32,to_cb61,ALLOWED_MODES,MAX_BPP


def parse():
    p=argparse.ArgumentParser(description="Oscar independent FP32→CALIBRI paid source")
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fp32-safetensors",type=Path,
                   help="Complete, nonsharded FP32 source safetensors with tied embedding")
    g.add_argument("--hf-model",type=str,
                   help="Independent FP32 model identifier on Hugging Face")
    p.add_argument("--hf-revision",type=str,default=None,
                   help="Pinned HF commit SHA/revision for external source reproducibility")
    p.add_argument("--mode",choices=ALLOWED_MODES,default="cold")
    p.add_argument("--prior-cb59",type=Path,default=None,
                   help="Offline trained compatible prior, optional for cold")
    p.add_argument("--format",choices=("cb59","cb61"),default="cb59")
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--max-bpp",type=float,default=MAX_BPP)
    p.add_argument("--conservative-denominator",type=int,default=None,
                   help="Exclude zero/fixed inflated model-growth parameters")
    p.add_argument("--require-conservative-pass",action="store_true")
    return p.parse_args()


def run(args):
    if not 0 < args.max_bpp<=16:
        raise ValueError("Oscar must use positive explicit max bits/parameter")
    if args.prior_cb59 is None and args.mode!="cold":
        raise ValueError("Oscar warm model needs a real physically saved pretrained dictionary")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Oscar output directory must be empty (immutable research result)")
    if args.fp32_safetensors:
        state=load_file(str(args.fp32_safetensors),device="cpu")
        origin="EXPLICIT_COMPLETE_FP32_LOCAL_SAFETENSORS"
        revision=None
    else:
        if not args.hf_revision:
            raise ValueError("Oscar independent HF pretrained model requires pinned --hf-revision")
        model=AutoModelForCausalLM.from_pretrained(
            args.hf_model,revision=args.hf_revision,
            torch_dtype=torch.float32,
            trust_remote_code=False).cpu().eval()
        state=model.state_dict()
        origin=args.hf_model
        revision=args.hf_revision
    prior=Path(args.prior_cb59).read_bytes() if args.prior_cb59 else None
    enc=encode_fp32(state,mode=args.mode,prior_blob=prior)
    physical=enc.blob if args.format=="cb59" else to_cb61(enc.blob)
    weightfile=out/("OSCAR_FULL_FP32_CONVERTED_MODEL."+args.format)
    weightfile.write_bytes(physical)
    n=enc.report["original_unique_parameters"]
    denom=args.conservative_denominator if args.conservative_denominator else n
    if denom>n or denom<=0:
        raise ValueError("Conservative reference parameter denominator must be <= actual source")
    summary={
       "algorithm":"Oscar","version":0,
       "input_FP32_origin":origin,
       "independent_FP32_checkpoint_claim_not_verified_by_codec":True,
       "pin":revision,
       "actual_FP32_coding_mode":args.mode,
       "prior_offline_only":args.prior_cb59 is not None,
       "physical_weight_file":weightfile.name,
       "physical_weight_size":len(physical),
       "physical_weight_sha":hashlib.sha256(physical).hexdigest(),
       "original_full_unique_source_parameters":n,
       "conservative_effective_denominator":denom,
       "all_in_nominal_bpp":0.,
       "all_in_conservative_bpp":0.,
       "all_runtime_sidecars_INCLUDED":False,
       "requires_subsequent_fully_paid_GT_sidecar_and_BPP_audit":True,
       "under_budget_core_only":False,
       "under_budget_conservative_core_only":False,
       "actual_answer_quality_from_this_source_not_evaluated":True,
       "model_accepted_for_deployment":False,
       "encoder":enc.report,
    }
    report=out/"OSCAR_COMPILER_SOURCE_PROVENANCE_AND_PHYSICAL_BUDGET.json"
    prior_size=-1
    for _ in range(32):
        report.write_text(json.dumps(summary,indent=2,ensure_ascii=False,
                            sort_keys=True)+"\n",encoding="utf8")
        size=report.stat().st_size
        total=len(physical)+size
        summary["all_in_nominal_bpp"]=8*total/n
        summary["all_in_conservative_bpp"]=8*total/denom
        summary["under_budget_core_only"]=8*total/n<=args.max_bpp
        summary["under_budget_conservative_core_only"]=8*total/denom<=args.max_bpp
        if size==prior_size:break
        prior_size=size
    else: raise AssertionError("Oscar physical JSON manifest failed stable byte accounting")
    report.write_text(json.dumps(summary,indent=2,ensure_ascii=False,
                      sort_keys=True)+"\n",encoding="utf8")
    if (report.stat().st_size!=prior_size or
        len(physical)+report.stat().st_size !=total):
        raise AssertionError("Oscar physical source size and provenance manifest differ")
    if args.require_conservative_pass and not summary["under_budget_conservative_core_only"]:
        raise SystemExit("Oscar conservative physical bpp exceeded "+str(args.max_bpp))
    print(json.dumps({
      "stage":"OSCAR_INDEPENDENT_FP32_SOURCE_COMPILER_DONE",
      "physical_file":str(weightfile),
      "full_unique_FP32_params":n,
      "nominal_core_bpp_incl_manifest":summary["all_in_nominal_bpp"],
      "conservative_core_bpp_incl_manifest":summary["all_in_conservative_bpp"],
      "all_runtime_neural_GT_sidecars_counted":False,
      "quality_recovered":False,
      "accepted_for_deployment":False,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    run(parse())
