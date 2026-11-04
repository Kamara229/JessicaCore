"""Quality control at a higher physical rate using the existing v58 plain INT6 codec."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from shared_checkpoint_v58 import cold_load_shared, pack_shared
from v84.sources import original_teacher
from .evaluate import assess


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = original_teacher()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    with TemporaryDirectory(dir=out) as tmp:
        path = Path(tmp) / "control.cb58"
        path.write_bytes(pack_shared(model.state_dict())[0])
        provenance = cold_load_shared(model, path)
    result = assess(model, tok)
    result.update({"mode": "INT6", "model_revision": MODEL_SHA, "physical": provenance,
                   "dense_decoded_quality_control": True})
    (out / "INT6.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
