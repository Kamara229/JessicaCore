"""Frozen full-generation replication of v132's first five-layer ablation.

This is not a teacher-guided correction and cannot use the 16 reference answers
before decoding. It never trains, changes bytes, touches official v89 TEST,
or modifies any Jessica backend service.
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
from v84.sources import original_teacher
from v94.generation import generate_one
from v108.generation import build_chat
from v117.protocol import ORIGINAL_CB59_SHA, TOTAL_UNIQUE
from v128.protocol import read_v119
from v129.operators import installed
from v129.run import load_green
from v132.protocol import all_examples
from .arms import ARMS, set_arm, assert_band0_gains
from .scoring import evaluate_case, report


def predict(model, tokenizer, bridge, policy, arm, cases):
    rows = []
    for pair, side, question, opposite, gold, foil in cases:
        if bridge is not None:
            gains = set_arm(bridge, policy, question, opposite, arm)
            assert_band0_gains(gains, arm)
        with torch.inference_mode():
            generation = generate_one(model, tokenizer, build_chat(tokenizer,question),
                                      stop_at_eos=True)
        rows.append(evaluate_case(tokenizer,pair,side,question,gold,foil,generation))
    return rows


def run(host, green, out):
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    source = read_v119(host)
    if hashlib.sha256(source.base).hexdigest() != ORIGINAL_CB59_SHA:
        raise ValueError("physical CB59 fingerprint changed")
    controller = load_green(green)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    cases = tuple(all_examples())
    if len(cases) != 16: raise AssertionError("missing counterfactual questions")
    data = {
        "experiment": "v133 frozen full-generation drop-band0 counterfactual",
        "host_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
        "green_sha256": hashlib.sha256(green.read_bytes()).hexdigest(),
        "physical_host_bytes": host.stat().st_size,
        "physical_host_bpp_excluding_Green": 8*host.stat().st_size/TOTAL_UNIQUE,
        "green_bytes_not_in_host_bpp": green.stat().st_size,
        "first_layer_band_dropped": [0,1,2,3,4],
        "source": "v132 fixed predeclared band0, never chosen from v133 responses",
        "no_training": True,"native_compressed_host": True,
        "no_leaked_gold_answers": True,"locked_v89_TEST_untouched": True,
        "no_Jessica_test_or_LIVE_changes": True,
        "arms": {},
    }
    teacher = original_teacher()
    try:
        data["arms"]["FP32"] = predict(teacher,tok,None,None,"FP32",cases)
    finally:
        del teacher
    with TemporaryDirectory(dir=out) as directory:
        base = Path(directory) / "core.cb59"
        base.write_bytes(source.base)
        model, meta, _ = make_model("baseline",
            SimpleNamespace(original=base,folder=Path(directory)))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise AssertionError("expected native compressed host")
        data["arms"]["CB59"] = predict(model,tok,None,None,"CB59",cases)
        with installed(model,source,moved=False) as bridge:
            for arm in ARMS[2:]:
                data["arms"][arm] = predict(model,tok,bridge,controller,arm,cases)
    data["score"] = report(data["arms"])
    dest = out / "v133_band0_autoregressive_replay.json"
    dest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v133_FROZEN_GENERATION",
        "scores":data["score"]["automated_visibility_ONLY"],
        "comparisons":data["score"]["paired_controls"],
        "report":str(dest)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("host",type=Path)
    p.add_argument("green",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.host,args.green,args.out)
