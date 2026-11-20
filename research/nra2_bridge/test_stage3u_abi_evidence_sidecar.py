"""Synthetic-only tests for S3U ABI sidecar; no model, weights, labels or gold."""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch

from research.nra2_bridge.stage3u_abi_evidence_sidecar import write_abi_evidence

COMMITMENT = "034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3"


@dataclass
class Mention:
    token_positions: list
    node: int
    char_start: int
    char_end: int


def fixture():
    n, t = 64, 8
    token_mask = torch.ones(n, t, dtype=torch.bool)
    token_lengths = torch.full((n,), t, dtype=torch.long)
    query = torch.zeros(n, t, dtype=torch.bool)
    query[:, 3:7] = True
    anchor = torch.zeros(n, 2, t, dtype=torch.bool)
    anchor[:, 0, 3:5] = True
    anchor[:, 1, 5:7] = True
    order = torch.tensor([[1, 2]] * n, dtype=torch.long)
    aligned = [
        {"tokens": t, "mentions": [
            Mention([0], 9, 0, 1),  # Graph mention: outside query
            Mention([3, 4], 1, 30, 34),
            Mention([5, 6], 2, 35, 39),
        ]}
        for _ in range(n)
    ]
    provenance = {
        "s3t_zip_sha256": "a" * 64,
        "s3t_inner_json_sha256": "b" * 64,
        "s3t_commitment_sha256": COMMITMENT,
        "s3u_token_fp16_sha256": "c" * 64,
        "tokenizer_revision": "pinned-test-tokenizer",
        "s3o_mask_builder_revision": "pinned-test-mask",
    }
    return dict(
        qmask=SimpleNamespace(query_mask=query, anchor_mask=anchor, node_order=order),
        aligned=aligned, token_mask=token_mask, token_lengths=token_lengths,
        prompt_sha256=[f"{i:064x}" for i in range(n)],
        provenance=provenance, j7_verified=True, prereg_verified=True,
    )


def test_roundtrip_synthetic_only():
    args = fixture()
    with TemporaryDirectory() as directory:
        manifest = write_abi_evidence(Path(directory), **args)
        assert manifest["output_gate"] == 0
        assert manifest["candidate_accepted"] is False
        assert manifest["candidate_scored"] is False
        assert (Path(directory) / "NRA2_S3U_QUERY_ABI_EVIDENCE.safetensors").is_file()


@pytest.mark.parametrize("mutation", [
    "j7", "prereg", "anchor_overlap", "missing_anchor", "extra_query_mention",
    "wrong_order", "bad_padding", "duplicate_prompt", "bad_commitment",
    "malformed_zip_sha", "cross_boundary_mention", "duplicate_native_tokens", "reversed_native_tokens",
])
def test_fail_closed_synthetic_only(mutation):
    args = fixture()
    if mutation == "j7":
        args["j7_verified"] = False
    elif mutation == "prereg":
        args["prereg_verified"] = False
    elif mutation == "anchor_overlap":
        args["qmask"].anchor_mask[0, 1, 4] = True
    elif mutation == "missing_anchor":
        args["qmask"].anchor_mask[0, 1, 5:7] = False
    elif mutation == "extra_query_mention":
        args["aligned"][0]["mentions"].append(Mention([3], 3, 40, 41))
    elif mutation == "cross_boundary_mention":
        args["aligned"][0]["mentions"].append(Mention([2, 3], 3, 40, 41))
    elif mutation == "duplicate_native_tokens":
        args["aligned"][0]["mentions"][1].token_positions = [3, 3, 4]
    elif mutation == "reversed_native_tokens":
        args["aligned"][0]["mentions"][1].token_positions = [4, 3]
    elif mutation == "wrong_order":
        args["qmask"].node_order[0] = torch.tensor([2, 1])
    elif mutation == "bad_padding":
        args["token_mask"][0, 7] = False
    elif mutation == "duplicate_prompt":
        args["prompt_sha256"][1] = args["prompt_sha256"][0]
    elif mutation == "bad_commitment":
        args["provenance"]["s3t_commitment_sha256"] = "0" * 64
    elif mutation == "malformed_zip_sha":
        args["provenance"]["s3t_zip_sha256"] = "not-a-physical-sha"
    with TemporaryDirectory() as directory:
        with pytest.raises((AssertionError, ValueError)):
            write_abi_evidence(Path(directory), **args)
        assert not (Path(directory) / "NRA2_S3U_QUERY_ABI_MANIFEST.json").exists()
