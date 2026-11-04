"""Autonomous frozen archive EOS and output-head failure localization.

This project remains isolated from Jessica LIVE. Nothing is trained, changed,
or committed from within evaluation; controller is only driven by prompts.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v117.protocol import ORIGINAL_CB59_SHA, TOTAL_UNIQUE
from v128.protocol import read_v119
from v129.run import load_green
from v129.operators import installed
from .protocol import CHOSEN, MODES, examples
from .probe import first_logits, summarize_logits, controller_probe


def execute(host, green, out):
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    source = read_v119(host)
    if hashlib.sha256(source.base).hexdigest() != ORIGINAL_CB59_SHA:
        raise ValueError("expected unchanged physical CB59")
    controller = load_green(green)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    tasks = tuple(examples())
    data = {"experiment": "v131 native first-token EOS forensics",
            "host_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
            "green_sha256": hashlib.sha256(green.read_bytes()).hexdigest(),
            "host_bytes": host.stat().st_size,
            "host_bpp_excluding_green": 8 * host.stat().st_size / TOTAL_UNIQUE,
            "green_bytes": green.stat().st_size,
            "status": "DIAGNOSTIC_ONLY - no trained weight changes",
            "no_test_v89": True,
            "no_live_jessica": True,
            "arms": {}, "controller": []}
    with TemporaryDirectory(dir=out) as folder:
        base = Path(folder) / "core.cb59"
        base.write_bytes(source.base)
        model, meta, _ = make_model("baseline",
              SimpleNamespace(original=base, folder=Path(folder)))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("unexpected dense original in packed kernel")
        for key in MODES:
            data["arms"][key] = []
        for name, side, question, opposite, gold, foil in tasks:
            # The only possible answer-aware work is summarizing frozen logits
            # after the model has computed them on the real question.
            logits, _, size = first_logits(model, tok, question)
            data["arms"]["CB59"].append({"pair": name, "side": side,
                **summarize_logits(logits, tok, gold, foil, None, size)})
        with installed(model, source, moved=False) as bridge:
            real_head = model.lm_head
            if real_head is not bridge.head:
                raise AssertionError("head bridge inconsistency")
            for name, side, question, opposite, gold, foil in tasks:
                data["controller"].append({"pair": name, "side": side,
                    **controller_probe(controller, question, opposite, tasks[0][2])})
                for mode in MODES[1:]:
                    if mode == "v119_green_current":
                        bridge.control(controller, question)
                    elif mode == "v119_green_swapped":
                        bridge.control(controller, opposite)
                    elif mode == "v119_green_constant":
                        bridge.control(controller, tasks[0][2])
                    else:
                        bridge.control(None, question)
                    if mode == "head_only":
                        for layer in bridge.blocks.values():
                            layer.gain = 0.0
                    if mode == "body_only":
                        # Temporarily use original output operator without any
                        # weight modification. Restore wrapper unconditionally.
                        model.lm_head = real_head.original
                    try:
                        logits, parts, size = first_logits(model, tok, question)
                    finally:
                        model.lm_head = real_head
                    data["arms"][mode].append({"pair": name, "side": side,
                        **summarize_logits(logits, tok, gold, foil, parts, size)})
    stats = {}
    for name, rows in data["arms"].items():
        stats[name] = {
            "first_eos_argmax": sum(x["argmax_id"] == x["eos_id"] for x in rows),
            "eos_top8": sum(x["eos_rank"] <= 8 for x in rows),
            "mean_eos_prob": sum(x["eos_probability"] for x in rows) / len(rows),
            "mean_eos_margin": sum(x["eos_vs_best_non_eos_margin"] for x in rows) / len(rows),
            "first_gold_token_top8": sum(min(v["rank"] for v in
                x["answer_first_token_probe_NOT_full_answer_accuracy"]["expected"]) <= 8 for x in rows),
        }
    stats["controller"] = {
        "max_swapped_coefficients_l2": max(x["current_vs_swapped_l2"] for x in data["controller"]),
        "min_swapped_coefficients_l2": min(x["current_vs_swapped_l2"] for x in data["controller"]),
    }
    data["summary"] = stats
    destination = out / "v131_first_token_eos_forensics.json"
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "v131_EOS_DIAGNOSTICS_ONLY", "summary": stats,
                      "report": str(destination)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("host", type=Path)
    p.add_argument("green", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    execute(args.host, args.green, args.out)
