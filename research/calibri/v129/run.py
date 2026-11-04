"""Full-model entry point. TRAIN, autonomous validation and timing are separate commands."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v94.generation import generate_one
from v108.generation import build_chat
from v117.protocol import TOTAL_UNIQUE
from v118.archive import Paid
from v122.policy import SharedGreenPolicy
from v127.protocol import SHARED_GREEN_SHA
from v128.protocol import read_v119
from v128.scoring import matches
from .archive import Source
from .fit import fit
from .operators import installed
from .protocol import VALID, OPEN


def load_green(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHARED_GREEN_SHA or path.stat().st_size > 10_000_000:
        raise ValueError("unexpected shared Green weights")
    policy = SharedGreenPolicy().eval()
    policy.load_state_dict(load_file(str(path)), strict=True)
    return policy.requires_grad_(False)


@torch.inference_mode()
def assess(model, tok, bridge, policy):
    rows = []
    for question, gold in (*VALID, *((q, None) for q in OPEN)):
        if bridge is not None:
            bridge.control(policy, question)
        item = generate_one(model, tok, build_chat(tok, question), stop_at_eos=True)
        ok = None if gold is None else matches(item["text"], gold, question) and item["eos_generated"]
        rows.append({"question": question, "gold": gold, "text": item["text"], "ids": item["ids"],
                     "eos": item["eos_generated"], "correct_with_eos": ok})
    return {"correct_with_eos": sum(r["correct_with_eos"] is True for r in rows),
            "gold_questions": len(VALID), "rows": rows}


@torch.inference_mode()
def bench(model, tok, source, policy):
    modes = ("bare_CB59", "relocated_green")
    timings = []
    for i, question in enumerate((VALID[0][0], OPEN[0])):
        prompt = build_chat(tok, question)
        for repeat in range(-1, 4):
            pair = {}
            for mode in (modes if (repeat+i) % 2 == 0 else modes[::-1]):
                ctx = installed(model, source, True) if mode == modes[1] else nullcontext(None)
                with ctx as bridge:
                    start = perf_counter()
                    if bridge is not None:
                        bridge.control(policy, question)
                    gen = generate_one(model, tok, prompt)
                    pair[mode] = {"seconds": perf_counter()-start, "tokens": len(gen["ids"])}
            if repeat >= 0:
                timings.append(pair)
    rates = {k: sum(p[k]["tokens"] for p in timings)/sum(p[k]["seconds"] for p in timings) for k in modes}
    return {"pairs": timings, "tps_including_green_and_prefill": rates,
            "relative_to_CB59": rates[modes[1]] / rates[modes[0]]}


def run(mode, host, green, out):
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    policy = load_green(green)
    source = read_v119(host)
    raw = source.raw
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    with TemporaryDirectory(dir=out) as folder:
        tmp = Path(folder)
        base = tmp / "source.cb59"
        base.write_bytes(source.base)
        if mode == "FP32":
            return assess(original_teacher(), tok, None, None)
        model, meta, _ = make_model("baseline", SimpleNamespace(original=base, folder=tmp))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("dense source in packed runtime")
        moved = mode.startswith("relocated") or mode in ("fit_relocated", "speed")
        if mode.startswith("fit_"):
            return {"sha256": fit(model, tok, policy, source, out, moved, tmp)}
        if moved or mode == "control_green":
            file = out / ("relocated.cb129" if moved else "control.cb118")
            raw = file.read_bytes()
            lock = out / ("fit_relocated.json" if moved else "fit_control.json")
            if json.loads(lock.read_text())["sha256"] != hashlib.sha256(raw).hexdigest():
                raise ValueError("model differs from frozen TRAIN output")
            source = Source(raw) if moved else Paid(raw)
        if mode == "speed":
            return bench(model, tok, source, policy)
        ctx = (nullcontext(None) if mode == "CB59" else
               installed(model, source, moved, tie_input=mode != "relocated_no_input"))
        with ctx as bridge:
            using_green = mode.endswith("green") or mode == "relocated_no_input"
            result = assess(model, tok, bridge, policy if using_green else None)
        archived = source.base if mode == "CB59" else raw
        result.update({"host_sha256": hashlib.sha256(archived).hexdigest(), "core_bytes": len(archived),
                       "core_bpp": 8*len(archived)/TOTAL_UNIQUE, "active_green_bytes": green.stat().st_size if using_green else 0,
                       "green_sha256": SHARED_GREEN_SHA if using_green else None})
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fit_control", "fit_relocated", "FP32", "CB59", "v119_green",
                   "v119_neutral", "control_green", "relocated_green", "relocated_neutral", "relocated_no_input", "speed"))
    for arg in ("host", "green", "out"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    result = run(args.mode, args.host, args.green, args.out)
    (args.out / (args.mode + ".json")).write_text(json.dumps(result, indent=2) + "\n")
