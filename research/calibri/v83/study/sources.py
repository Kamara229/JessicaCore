"""Pinned real source provenance and original TRAIN-only input windows."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from binary_geometry_v64 import EXPECTED_UNIQUE, MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import SOURCE_BYTES, WINDOW

TRAIN_DOMAINS = ("wiki_fit", "news_fit")
WINDOWS_PER_DOMAIN = 2


def original_packed_source(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) != SOURCE_BYTES:
        raise AssertionError("incomplete pretrained original CB59 source")
    archive = CB59(raw)
    original = archive.meta["original_unique_parameter_count"]
    if original != EXPECTED_UNIQUE:
        raise AssertionError("wrong pretrained unique original parameter count")
    return {
        "original_real_pretrained_model": MODEL,
        "pretrained_revision": MODEL_SHA,
        "original_unique_parameters": int(original),
        "original_CB59_source_bytes": len(raw),
        "original_CB59_source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_CB59_physical_bpw": 8 * len(raw) / original,
        "fully_encoded_original_weight_source_including_headers": True,
    }


def train_windows():
    # Import only at execution time so numerical unit tests need no datasets.
    from paid_lowrank_v76 import get_tokens
    for domain in TRAIN_DOMAINS:
        ids, source = get_tokens(domain)
        if domain not in TRAIN_DOMAINS or source.get("split") != "train":
            raise AssertionError("TEST or VALID corpus prohibited in v83")
        if ids.numel() < WINDOWS_PER_DOMAIN * WINDOW:
            raise AssertionError("insufficient pinned genuine TRAIN tokens")
        windows = []
        for j in range(WINDOWS_PER_DOMAIN):
            one = ids[j * WINDOW:(j + 1) * WINDOW].clone()
            windows.append({
                "tokens": one,
                "sha256": hashlib.sha256(one.numpy().tobytes()).hexdigest(),
                "source_window_offset": j * WINDOW,
            })
        yield domain, source, windows
