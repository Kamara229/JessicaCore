"""Read-only frozen-host paired audit; NO fit, NO model edits, NO official TEST.

All predictions are genuine autoregressive CPU generations. Gold answers
are read by the score function only AFTER generation. Frozen Green receives
only the current question, a counterfactual question or a fixed question.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v94.generation import generate_one
from v108.generation import build_chat
from v117.protocol import ORIGINAL_CB59_SHA, TOTAL_UNIQUE
from v128.protocol import read_v119
from v129.operators import installed
from v129.run import load_green
from .protocol import PAIRS, tasks
from .scoring import evidence, summarize
from .analysis import compare


def predict(model, tokenizer, bridge, policy, mode):
    result = []
    for pair, side, question, gold, foil, opposite_question in tasks():
        if bridge is not None:
            if mode == "green_current":
                bridge.control(policy, question)
            elif mode == "green_swapped":
                bridge.control(policy, opposite_question)
            elif mode == "green_constant":
                bridge.control(policy, PAIRS[0].left)
            else:
                bridge.control(None, question)
        # No reference answer enters model, tokenizer prompt or controller.
        with torch.inference_mode():
            generation = generate_one(model, tokenizer, build_chat(tokenizer, question),
                                      stop_at_eos=True)
        result.append({
            "pair": pair.name, "family": pair.family, "side": side,
            "question": question, "reference_for_post_generation_grading_only": gold,
            "foil_for_post_generation_grading_only": foil,
            "generated_text": generation["text"],
            "generated_token_ids": generation["ids"],
            "observation": evidence(generation["text"], question, gold, foil,
                                    generation["eos_generated"]),
        })
    return {"summary": summarize(result), "rows": result}


def audit(host, green_path, out):
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    source = read_v119(host)  # pins full learned v119 archive and its hash
    if hashlib.sha256(source.base).hexdigest() != ORIGINAL_CB59_SHA:
        raise ValueError("CB59 source is not pinned")
    policy = load_green(green_path)  # pins 8.2MB frozen controller
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    report = {
        "experiment": "v130 counterfactual conditioned-information pilot",
        "frozen_host_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
        "frozen_green_sha256": hashlib.sha256(green_path.read_bytes()).hexdigest(),
        "core_bytes": len(host.read_bytes()),
        "core_bpp_excluding_green": 8 * len(host.read_bytes()) / TOTAL_UNIQUE,
        "controller_bytes_excluding_core": green_path.stat().st_size,
        "frozen_no_training": True,
        "no_oracle_in_generation": True,
        "no_new_weights": True,
        "independent_v89_test_untouched": True,
        "int6_not_evaluated_in_this_pilot": True,
        "claims": "Automated answer-token visibility is NOT proof of semantic correctness.",
        "arms": {},
    }

    # Hold the FP32 original teacher as an upper-bound context-following check.
    teacher = original_teacher()
    try:
        report["arms"]["FP32"] = predict(teacher, tokenizer, None, None, "plain")
    finally:
        del teacher

    with TemporaryDirectory(dir=out) as temporary:
        base = Path(temporary) / "core.cb59"
        base.write_bytes(source.base)
        from types import SimpleNamespace
        model, meta, _ = make_model("baseline", SimpleNamespace(original=base, folder=Path(temporary)))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise AssertionError("densely decoded host cannot be called direct native")
        report["arms"]["CB59"] = predict(model, tokenizer, None, None, "plain")
        with installed(model, source, moved=False) as bridge:
            for label, mode in (
                ("v119_neutral", "neutral"),
                ("v119_green_current", "green_current"),
                ("v119_green_swapped", "green_swapped"),
                ("v119_green_constant", "green_constant"),
            ):
                report["arms"][label] = predict(model, tokenizer, bridge, policy, mode)
    report["posthoc_FP32_and_Green_controls"] = compare(report["arms"])
    path = out / "v130_paired_information_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return path, {k: v["summary"] for k, v in report["arms"].items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", type=Path)
    parser.add_argument("green", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    filename, summary = audit(args.host, args.green, args.output)
    print(json.dumps({"stage": "v130_frozen_paired_pilot", "report": str(filename),
                      "scores": summary}, ensure_ascii=False))
