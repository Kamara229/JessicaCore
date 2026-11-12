"""S150: research-only, same-host 2x MPS + 1x CPU parallel neural Skill probe.

This is a capacity/throughput experiment, NOT S142 training or a quality gain.
Three independent spawned processes train actual S145 NeuralSkillBlock weights
against deterministic synthetic hidden-state targets while carrying a frozen
43.26M-parameter FP16 storage envelope in each process. This is deliberately
a conservative separate-parent-memory probe; S142 full forward/456+128 audit
must be performed before promoting any genuinely trained Skill.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

from calibri2.s145.neural_skill_blocks import LayerwiseNeuralSkillBank, NeuralSkillSpec

SOURCE_PARAMS = 43_261_760
WORKERS = (
    ("binding_gpu", "mps", (3, 9), 150001),
    ("language_gpu", "mps", (4, 10), 150002),
    ("context_cpu", "cpu", (2, 8), 150003),
)


def run_training(name: str, device_name: str, layers: tuple[int, int], seed: int,
                 *, steps: int, seq_len: int, ref_params: int) -> dict:
    """Genuine independent Skill autograd; synthetic inputs are NOT QA evidence."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS GPU unavailable; no CPU fallback permitted")
        # Each GPU worker is bounded to 32% of recommended working-set memory.
        torch.mps.set_per_process_memory_fraction(0.32)
    device = torch.device(device_name)
    # Parent-memory *envelope*: real S142 tensors are NEVER loaded in S150.
    # Actual S142 model forward and independent J7 retention are future steps.
    frozen_reference = torch.zeros(ref_params, device=device, dtype=torch.float16)
    assert not frozen_reference.requires_grad
    bank = LayerwiseNeuralSkillBank(320, 12)
    bank.add_skill(NeuralSkillSpec(name, layers, 8))
    bank.to(device)
    weights = bank.select_new_skill_for_training(name)
    opt = torch.optim.Adam(weights, lr=0.008)
    hidden = 0.55 * torch.randn(4, seq_len, 320, device=device)
    target = hidden + 0.11 * torch.tanh(hidden)
    first_loss = None
    last_loss = None
    begin = time.monotonic_ns()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        output = bank.forward_at_layer(layers[0], hidden)
        output = bank.forward_at_layer(layers[1], output)
        loss = F.mse_loss(output, target)
        if not torch.isfinite(loss).item():
            raise AssertionError("nonfinite synthetic training objective")
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    if device_name == "mps":
        torch.mps.synchronize()
    end = time.monotonic_ns()
    if not (math.isfinite(last_loss) and last_loss < first_loss):
        raise AssertionError(f"{name}: neural weights did not learn synthetic mapping")
    # Only these Skill weights updated. The frozen envelope never trains.
    if any(p.requires_grad for p in [frozen_reference]):
        raise AssertionError("frozen parent memory envelope became trainable")
    gpu_bytes = (
        int(torch.mps.current_allocated_memory()) if device_name == "mps" else 0
    )
    del frozen_reference
    return {
        "skill": name, "device": device_name, "layers": list(layers),
        "skill_trainable_parameters": bank.parameter_count(),
        "frozen_parent_envelope_params": ref_params,
        "steps": steps, "sequence_length": seq_len,
        "first_synthetic_loss": first_loss, "last_synthetic_loss": last_loss,
        "start_monotonic_ns": begin, "end_monotonic_ns": end,
        "elapsed_seconds": (end - begin) / 1e9,
        "gpu_current_allocated_bytes": gpu_bytes,
        "pid": os.getpid(), "genuine_autograd": True,
        "model_accepted": False,
    }


def worker(entry, release, events, steps, seq_len, ref_params):
    name, device, layers, seed = entry
    try:
        # Complete initialization BEFORE the shared release event.
        events.put({"kind": "ready", "skill": name, "device": device})
        if not release.wait(timeout=150):
            raise TimeoutError("parallel startup barrier timed out")
        outcome = run_training(name, device, layers, seed,
                               steps=steps, seq_len=seq_len,
                               ref_params=ref_params)
        events.put({"kind": "result", "result": outcome})
    except BaseException as exc:
        events.put({"kind": "error", "skill": name,
                    "error": repr(exc), "traceback": traceback.format_exc()})


def run_probe(*, steps=72, seq_len=192, ref_params=SOURCE_PARAMS):
    """Use spawn, never fork an initialized MPS device."""
    ctx = mp.get_context("spawn")
    events = ctx.Queue()
    release = ctx.Event()
    processes = [ctx.Process(target=worker,
                             args=(entry, release, events, steps, seq_len, ref_params),
                             name=entry[0]) for entry in WORKERS]
    messages = []
    try:
        for process in processes:
            process.start()
        ready = {}
        while len(ready) != len(WORKERS):
            msg = events.get(timeout=150)
            messages.append(msg)
            if msg["kind"] == "error":
                raise RuntimeError(msg["traceback"])
            if msg["kind"] != "ready":
                raise AssertionError("worker emitted result before common barrier")
            if msg["skill"] in ready:
                raise AssertionError("duplicate worker readiness")
            ready[msg["skill"]] = msg
        if len(ready) != 3 or {x["device"] for x in ready.values()} != {"mps", "cpu"}:
            raise AssertionError("three intended workers not ready")
        release.set()
        outcomes = []
        while len(outcomes) != len(WORKERS):
            msg = events.get(timeout=240)
            messages.append(msg)
            if msg["kind"] == "error":
                raise RuntimeError(msg["traceback"])
            if msg["kind"] != "result":
                raise AssertionError(f"unexpected worker event: {msg}")
            outcomes.append(msg["result"])
        for process in processes:
            process.join(timeout=30)
            if process.exitcode != 0:
                raise RuntimeError(f"worker {process.name} exit={process.exitcode}")
        by_name = {row["skill"]: row for row in outcomes}
        if set(by_name) != {x[0] for x in WORKERS}:
            raise AssertionError("unexpected or duplicate skill outcome")
        if not (sum(r["device"] == "mps" for r in outcomes) == 2 and
                sum(r["device"] == "cpu" for r in outcomes) == 1):
            raise AssertionError("not exactly 2 GPU+1 CPU parallel processes")
        latest_start = max(r["start_monotonic_ns"] for r in outcomes)
        earliest_end = min(r["end_monotonic_ns"] for r in outcomes)
        overlap_seconds = max(0, (earliest_end - latest_start)/1e9)
        if overlap_seconds <= 0:
            raise AssertionError("three-way simultaneous training not demonstrated")
        report = {
            "stage": "S150_SINGLE_MACOS_HOST_THREE_INDEPENDENT_SKILLS",
            "all_3_training_intervals_overlap": True,
            "common_overlapping_seconds": overlap_seconds,
            "concurrent_gpu_workers": 2, "concurrent_cpu_workers": 1,
            "same_runner_processes": True,
            "workers": outcomes,
            "hardware_capacity_probe_only": True,
            "synthetic_training_data": True,
            "original_S142_weights_loaded": False,
            "original_456_and_S122_128_QA_replayed": False,
            "no_independent_heldout_QA_run": True,
            "J7_quality_and_model_acceptance": "not_tested",
            "model_accepted": False,
            "no_LIVE_Jessica_Render_Supabase_APK_changes": True,
        }
        return report
    finally:
        release.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=72)
    p.add_argument("--seq-len", type=int, default=192)
    p.add_argument("--report", type=Path, default=Path("results/s150/three_way_hardware_probe.json"))
    args = p.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_probe(steps=args.steps, seq_len=args.seq_len)
    except BaseException as exc:
        args.report.write_text(json.dumps({
            "stage": "S150_SINGLE_MACOS_HOST_THREE_INDEPENDENT_SKILLS",
            "hardware_capacity_probe_only": True,
            "model_accepted": False, "failure": repr(exc),
            "traceback": traceback.format_exc()
        }, indent=2))
        raise
    args.report.write_text(json.dumps(report, indent=2))
    print("S150_THREE_WAY_SAME_HOST", json.dumps({
        "overlap_seconds": report["common_overlapping_seconds"],
        "workers": [
            {"skill": x["skill"], "device": x["device"],
             "elapsed": round(x["elapsed_seconds"], 3),
             "loss_before": x["first_synthetic_loss"],
             "loss_after": x["last_synthetic_loss"]}
            for x in report["workers"]
        ],
        "model_accepted": False,
    }), flush=True)


if __name__ == "__main__":
    main()
