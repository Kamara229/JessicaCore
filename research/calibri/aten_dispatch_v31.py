#!/usr/bin/env python3
"""CALIBRI v31: isolate PyTorch ATen CPU dispatch while oneDNN is capped AVX2.

Separate fresh processes change ONE variable: ATEN_CPU_CAPABILITY=default
or avx2. Source model, weights, BF16, zero router jitter, data, threads,
oneDNN cap, and deterministic option stay pinned. Dispatch metadata is NOT
CPU assembly verification and does not cover every ATen/BLAS operator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# CI MUST set ATEN_CPU_CAPABILITY *before* this or any other torch import.
requested = os.environ.get("ATEN_CPU_CAPABILITY")
if requested not in ("default", "avx2"):
    raise RuntimeError("ATEN_CPU_CAPABILITY must be default or avx2 on process startup")
if os.environ.get("DNNL_MAX_CPU_ISA") != "AVX2" or os.environ.get("ONEDNN_MAX_CPU_ISA") != "AVX2":
    raise RuntimeError("oneDNN AVX2 cap must be fixed before any Torch import")

import torch  # noqa: E402
import isa_cap_v29  # noqa: E402
import operator_trace_v30  # noqa: E402


def run(args):
    # This is PyTorch DispatchStub's selected CPU capability, not evidence of
    # effective vector assembly or every CPU/BLAS backend implementation.
    selected = str(torch._C._get_cpu_capability()).upper()
    expected = {"default": "DEFAULT", "avx2": "AVX2"}[requested]
    if selected != expected:
        raise AssertionError(f"ATen CPU capability was {selected!r} instead of {expected!r}")

    # The existing v30 profiler wraps the identical v29 real-pretrained
    # Switch BF16 forward and captures actual ATen operator names/calls.
    # Do not accidentally profile an independent synthetic matmul.
    isa_cap_v29.trace = operator_trace_v30.profiled_trace
    original = argparse.Namespace(
        cap="AVX2", output=args.output,
        tokens=args.tokens, encoder_len=args.encoder_len,
        decoder_len=args.decoder_len,
    )
    isa_cap_v29.run(original)
    dest = Path(args.output)
    doc = json.loads(dest.read_text(encoding="utf-8"))
    doc["experiment"] = "CALIBRI_v31_Switch_real_BF16_ATen_dispatch_factor"
    doc["torch_ATen_dispatch_requested"] = requested
    doc["torch_ATen_dispatch_selected_reported"] = selected
    doc["ATen_dispatch_observation_limits"] = (
        "torch._C._get_cpu_capability() reports CPU dispatch capability chosen "
        "by PyTorch (DispatchStub); not proof of the exact assembly instructions "
        "or optimized BLAS kernel used for each operator. Profiler labels "
        "identify ATen operator kinds, not ISA."
    )
    doc["experiment_control"] = (
        "Only ATEN_CPU_CAPABILITY changes across fresh processes; both use "
        "DNNL_MAX_CPU_ISA=AVX2, ONEDNN_MAX_CPU_ISA=AVX2, BF16, MKLDNN ON, "
        "torch deterministic=True, router jitter zero and pinned checkpoints."
    )
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "selected_ATen": selected, "requested_ATen": requested,
        "requested_oneDNN": doc["requested_dnnl_max_cpu_isa"],
        "cpu": doc["cpu_brand"],
        "logits": {key: val["full_logit_sha256"] for key,val in doc["outputs"].items()},
        "ATen_mm_calls": {key: next(
            (e["calls"] for e in val["cpu_profiler"]["events"]
             if e["operator"] == "aten::mm"), 0)
            for key,val in doc["outputs"].items()},
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--encoder-len", type=int, default=64)
    parser.add_argument("--decoder-len", type=int, default=32)
    run(parser.parse_args())
