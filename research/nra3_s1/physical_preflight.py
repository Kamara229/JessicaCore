"""Physically pack UNTRAINED 6M NRA3 weights into genuine CB59 and audit bytes.

No gold/train/EVAL/checkpoints or Jessica. This does not certify native C
inference, codebook training, generated EOS, retention or Green Tunnel speed.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch

from research.nra3_s0.neural_area import NRA3NeuralArea
from research.nra3_s1.physical_cb59 import save_and_validate


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Do not overwrite an existing research result")
    torch.manual_seed(30159)
    torch.set_num_threads(1)
    model=NRA3NeuralArea().cpu().eval()
    # Full physical output must be the actual bytes from the existing Calibri
    # CB59 implementation: no hand-calculated 0.625bpw success claims.
    physical=args.out/"NRA3_UNTRAINED_6M_ACTUAL_CB59_K32V8.cb59"
    report=save_and_validate(model,physical)
    baseline_n=43_261_760
    # Only an *illustrative* parent 0.69 bpp scenario. No parent artifact
    # loaded and no Green Tunnel byte ownership proven in this stage.
    assumed_parent_bpp=0.69
    report["illustrative_parent_bpp_NOT_PHYSICAL"]=assumed_parent_bpp
    report["illustrative_combined_bpp_EXCLUDING_GT_MANIFEST"]=(
        (baseline_n*assumed_parent_bpp+len(physical.read_bytes())*8)/
        (baseline_n+model.parameter_count()))
    report["illustrative_signed_remaining_bytes_for_GT_and_manifest"]=(
        math.floor(.7*(baseline_n+model.parameter_count())/8
                   - baseline_n*assumed_parent_bpp/8
                   - physical.stat().st_size))
    report["all_inclusive_gt_and_Calibri_source_loaded"]=False
    report["all_inclusive_actual_rate_accepted"]=False
    report["native_CB59_CPU_operator_dispatch"]=False
    report["locked_before_any_full_EOS_or_616_J7"]=True
    report["no_prior_five_or_reserved_sixth_EVAL_data"]=True
    (args.out/"NRA3_S1_PHYSICAL_UNTRAINED_RESEARCH.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
