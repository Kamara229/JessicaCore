"""Explicit UNPAID dense interventions, preserving original tied parameters."""
import hashlib
import numpy as np
import torch
from direct_codebook_v59 import CB59
from v173.recover import exact_dense_matrix
from .protocol import SOURCE_BYTES, SOURCE_SHA, FAMILIES, family, selected


def extract_original(path):
    # The original is the leading CB59 in the previously verified CB161 artifact.
    # Its independent exact SHA and CB59 internal CRC/SHA are verified below.
    blob = path.read_bytes()[:SOURCE_BYTES]
    if len(blob) != SOURCE_BYTES or hashlib.sha256(blob).hexdigest() != SOURCE_SHA:
        raise ValueError("Not the immutable original CB59 source")
    return CB59(blob)


@torch.no_grad()
def snapshots(model, archive):
    state = model.state_dict()
    if set(state) != set(archive.meta["full_original_state_keys"]):
        raise ValueError("Model/source state keys differ")
    originals, decoded, counts = {}, {}, dict.fromkeys(FAMILIES, 0)
    for name, desc in archive.meta["tensor_descriptors"].items():
        if desc["dtype"] == "CB59_K32V8":
            counts[family(name)] += 1
            originals[name] = state[name].detach().clone()
            decoded[name] = torch.from_numpy(exact_dense_matrix(archive.handles[name]))
            if originals[name].shape != decoded[name].shape:
                raise ValueError("Source matrix shape differs")
        else:
            raw = torch.from_numpy(np.array(archive.raw[name], copy=True))
            if not torch.equal(state[name], raw):
                raise ValueError("FP32/source uncompressed vector mismatch: " + name)
    if counts != {"embedding_head": 1, "attention": 120, "mlp": 90}:
        raise ValueError("Missing source family: " + str(counts))
    assert_tie(model)
    return originals, decoded, counts


def assert_tie(model):
    if model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise ValueError("Original tied embedding/head broken")


@torch.no_grad()
def apply_mask(model, originals, decoded, mask):
    state = model.state_dict()
    quantized = []
    for path, source in originals.items():
        compressed = selected(mask, path)
        value = decoded[path] if compressed else source
        state[path].copy_(value)
        if not torch.equal(state[path], value):
            raise AssertionError("Intervention not exactly applied")
        if compressed:
            quantized.append(path)
    assert_tie(model)
    return quantized
