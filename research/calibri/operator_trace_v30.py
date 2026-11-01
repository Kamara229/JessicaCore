#!/usr/bin/env python3
"""CALIBRI v30: PyTorch ATen CPU operator trace around an unchanged v29 forward.

The original v29 source checkpoint, token selection, BF16 computation and
oneDNN cap stay unchanged. Only a CPU profiler is added. The profiler reports
ATen OPERATOR names, NOT the CPU instruction set actually executed by ATen.
"""
from __future__ import annotations
import argparse
from torch.profiler import ProfilerActivity, profile, record_function
import isa_cap_v29


def profiled_trace(model, windows, router):
    with profile(
        activities=[ProfilerActivity.CPU], record_shapes=False,
        profile_memory=False, with_stack=False
    ) as prof:
        with record_function("CALIBRI_v30_pretrained_BF16_forward"):
            result = ORIGINAL_TRACE(model, windows, router)
    events = []
    for event in prof.key_averages():
        if event.key.startswith("aten::") or event.key.startswith("CALIBRI_v30"):
            events.append({
                "operator": event.key,
                "calls": int(event.count),
                "self_cpu_time_us_diagnostic": round(float(event.self_cpu_time_total), 3),
            })
    if not events or not any(x["operator"] in ("aten::mm", "aten::addmm", "aten::bmm") for x in events):
        raise AssertionError("real pretrained BF16 forward lacks observable ATen matmul operators")
    result["cpu_profiler"] = {
        "observability": "ATen operator names/call counts, NOT ATen instruction-level ISA",
        "operator_kinds_observed": len(events),
        "events": sorted(events, key=lambda e: e["operator"]),
        "cpu_profiler_is_a_diagnostic_not_throughput": True,
    }
    return result


ORIGINAL_TRACE = isa_cap_v29.trace


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cap", required=True, choices=["DEFAULT", "AVX2"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--encoder-len", type=int, default=64)
    parser.add_argument("--decoder-len", type=int, default=32)
    args=parser.parse_args()
    # Original trace still produces SAME numeric evidence as v29; profiler
    # only instruments the forward and stores operator call counts.
    isa_cap_v29.trace = profiled_trace
    isa_cap_v29.run(args)
