"""Frozen v119 body/head interactions on the identical native CB59 archive.

No teacher, new trained weights, forward oracle, locked TEST, production change,
or Green adjustment in this experiment. Native EOS-margin diagnosis only.
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
from v129.operators import installed
from v131.probe import first_logits, summarize_logits
from .protocol import all_examples, conditions, verify_roles, band_of, BANDS
from .analysis import summarize


def run(host, out):
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    packed = read_v119(host)
    if hashlib.sha256(packed.base).hexdigest() != ORIGINAL_CB59_SHA:
        raise ValueError("baseline CB59 archive mismatched")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    cases = all_examples()
    data = {"status": "diagnostic, weights frozen and byte-identical",
            "model": MODEL, "source_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
            "full_source_bytes": host.stat().st_size,
            "core_bpp": 8 * host.stat().st_size / TOTAL_UNIQUE,
            "shared_green_weight_bytes_added": 0, "official_TEST_unopened": True,
            "bands": [{"first": start, "end_exclusive": end} for start, end in BANDS],
            "arms": {}}
    with TemporaryDirectory(dir=out) as folder:
        folder = Path(folder)
        raw = folder / "original.cb59"
        raw.write_bytes(packed.base)
        model, metadata, _ = make_model("baseline",
            SimpleNamespace(original=raw, folder=folder))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("original trainable dense weight materialized")
        with installed(model, packed, moved=False) as bridge:
            verify_roles(bridge)
            for name, active in conditions():
                observations = []
                for pair, side, question, opposite, gold, foil in cases:
                    # All contributions from the core exist in the frozen
                    # archive. Only runtime gains are set to zero or one.
                    for role, layer in bridge.blocks.items():
                        layer.gain = 1.0 if active(band_of(role)) else 0.0
                    with torch.inference_mode():
                        logits, parts, prompt_tokens = first_logits(model, tokenizer, question)
                    fields = summarize_logits(logits, tokenizer, gold, foil, parts, prompt_tokens)
                    probes = fields["answer_first_token_probe_NOT_full_answer_accuracy"]
                    observations.append({
                        "pair": pair, "side": side,
                        "question": question, "gold_FOR_ANALYSIS_ONLY": gold,
                        "foil_FOR_ANALYSIS_ONLY": foil,
                        "eos_first": fields["argmax_id"] == fields["eos_id"],
                        "eos_margin": fields["eos_vs_best_non_eos_margin"],
                        "eos_probability": fields["eos_probability"],
                        "best_expected_rank": min(x["rank"] for x in probes["expected"]),
                        "best_foil_rank": min(x["rank"] for x in probes["swapped"]),
                        "first_token_id": fields["argmax_id"],
                        "first_token_text": fields["argmax_text"],
                        "head_terms": fields["head_terms"],
                    })
                data["arms"][name] = observations
    data["summary"] = summarize(data["arms"])
    dest = out / "v132_body_head_interaction.json"
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": "v132_PAIRED_BODY_HEAD_ABLATION",
                      "summary": data["summary"], "report": str(dest)}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("host", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    run(args.host, args.out)
