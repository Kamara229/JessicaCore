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
from research.nra3_s1.entropy_envelope import store_lossless_cb59, restore_lossless_cb59


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
    # S61-style lossless entropy coding of *exact* genuine CB59 physical bytes.
    # Recovered CB59 is bit-for-bit identical; no approximation beyond CB59.
    original=physical.read_bytes()
    entropy=store_lossless_cb59(original)
    recovered=restore_lossless_cb59(entropy)
    if recovered!=original:
        raise AssertionError("CB61 physical recovery mutated CB59")
    envelope=args.out/"NRA3_UNTRAINED_6M_EXACT_CB59_S61_ENTROPY.n3s61"
    if envelope.exists():
        raise FileExistsError("Research envelope cannot be overwritten")
    envelope.write_bytes(entropy)
    if restore_lossless_cb59(envelope.read_bytes())!=original:
        raise AssertionError("CB61 persisted envelope failed physical restore")
    report["lossless_CB61_physical_archive"]=envelope.name
    report["lossless_CB61_stored_bytes"]=len(entropy)
    report["lossless_CB61_stored_sha256"]=__import__("hashlib").sha256(entropy).hexdigest()
    report["lossless_CB61_restores_exact_CB59_sha256"]=report["sha256"]
    report["lossless_CB61_bytes_saved_vs_raw_CB59"]=len(original)-len(entropy)
    report["lossless_CB61_new_area_effective_bpp"]=8*len(entropy)/model.parameter_count()
    baseline_n=43_261_760
    assumed_parent_bpp=0.69  # illustrative only; NO real Calibri or GT loaded
    report["illustrative_parent_bpp_NOT_PHYSICAL"]=assumed_parent_bpp
    report["illustrative_raw_CB59_plus_parent_bpp_EXCLUDING_GT_MANIFEST"]=(
        (baseline_n*assumed_parent_bpp+len(original)*8)/
        (baseline_n+model.parameter_count()))
    report["illustrative_CB61_plus_parent_bpp_EXCLUDING_GT_MANIFEST"]=(
        (baseline_n*assumed_parent_bpp+len(entropy)*8)/
        (baseline_n+model.parameter_count()))
    report["illustrative_signed_remaining_bytes_for_GT_and_manifest"]=(
        math.floor(.7*(baseline_n+model.parameter_count())/8
                   - baseline_n*assumed_parent_bpp/8
                   - len(entropy)))
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
