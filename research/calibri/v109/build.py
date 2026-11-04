"""Offline encoder facade. Only the resulting physical archive is used for inference."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from binary_geometry_v64 import EXPECTED_UNIQUE, MODEL, MODEL_SHA
from v84.sources import original_teacher
from .encoder import rewrite
from .sources import calibration


def build(source_path, out):
    out.mkdir(parents=True, exist_ok=True)
    teacher = original_teacher()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    bases, tokens_sha = calibration(teacher, tok)
    candidate, by_matrix = rewrite(source_path.read_bytes(), teacher, bases)
    report = {"model_revision": MODEL_SHA, "calibration_token_sha256": tokens_sha,
              "sha256": hashlib.sha256(candidate).hexdigest(), "archive_bytes": len(candidate),
              "bits_per_unique_weight": 8 * len(candidate) / EXPECTED_UNIQUE,
              "changed_codes": sum(m["changed_codes"] for m in by_matrix.values()),
              "matrices": by_matrix}
    (out / "v109.cb59").write_bytes(candidate)
    (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Encoded {len(by_matrix)} matrices; candidate {report['sha256']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    build(args.original, args.out)
