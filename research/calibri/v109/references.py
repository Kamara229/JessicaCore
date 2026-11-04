"""Rerun the frozen v102 and v104 weights with their own native operators."""
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
from v104 import runtime
from v104.physical import unpack
from v108.spec import V102_SOURCE_SHA256
from .evaluate import assess

PINS = {"v102": V102_SOURCE_SHA256,
        "v104": "311eabebeab0307611167e12d6c62ab2c011df1ac9a2ba7c340627cfb3e3d41f"}


def measure(name, source, out):
    blob = source.read_bytes()
    if hashlib.sha256(blob).hexdigest() != PINS[name]:
        raise ValueError(f"unexpected frozen {name} archive")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    with TemporaryDirectory(dir=out) as tmp:
        folder = Path(tmp)
        if name == "v104":
            embedded, banks = unpack(blob)
            source = folder / "nested.cb59"
            source.write_bytes(embedded)
        model, _, _ = make_model("baseline", SimpleNamespace(original=source, folder=folder))
        if name == "v104":
            model, _ = runtime.bind_native_banks(model, banks, Path(runtime.__file__).with_name("native104.c"),
                                                 folder / "banked.so")
        result = assess(model, tok)
    result.update({"model_revision": MODEL_SHA, "mode": name, "source_sha256": PINS[name]})
    (out / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=tuple(PINS))
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    measure(args.mode, args.archive, args.out)
