"""S3BY: fail-closed fit-only loader boundary; no training, scoring, EVAL, or weights writes.

This module must be called by any future optimizer *at each minibatch*.
It deliberately cannot read EVAL and refuses arbitrary sample objects.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from research.nra2_bridge.stage3bx_blocked_family_topology_protocol import (
    split_data, validate_splits, PRE_S3BR_PARENT_SHA, REJECTED_S3BT_SHA,
    OLD_S3BR_SHA, require,
)


class FitOnlyGate:
    def __init__(self):
        groups = split_data()
        validate_splits(groups)
        self.allowed = frozenset(r.prompt_sha256 for r in groups["fit"])
        self.denied = frozenset(
            r.prompt_sha256 for name in ("grammar_guard", "topology_guard", "double_guard")
            for r in groups[name]
        )
        require(len(self.allowed) == 192 and len(self.denied) == 320
                and not (self.allowed & self.denied), "invalid fit/guard partition")
        self.parent_verified = False

    def verify_parent(self, physical_fp16_path: str | Path):
        p = Path(physical_fp16_path)
        require(p.is_file() and not p.is_symlink(), "parent FP16 missing or symlink")
        digest = hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        require(digest != REJECTED_S3BT_SHA, "rejected S3BT parent")
        require(digest == PRE_S3BR_PARENT_SHA, "S3BN physical FP16 SHA mismatch")
        self.parent_verified = True
        return digest

    def minibatch(self, rows):
        require(self.parent_verified, "parent not physically verified")
        require(isinstance(rows, (list, tuple)) and len(rows) > 0,
                "empty or unsupported minibatch")
        digests = []
        for row in rows:
            digest = getattr(row, "prompt_sha256", None)
            require(isinstance(digest, str) and len(digest) == 64,
                    "missing prompt SHA")
            require(digest in self.allowed and digest not in self.denied,
                    "guard, EVAL or unknown sample attempted optimizer entry")
            require(getattr(row, "split", None) is not None,
                    "sample missing original TRAIN split")
            digests.append(digest)
        require(len(digests) == len(set(digests)), "duplicate within minibatch")
        return tuple(rows)


def self_test():
    gate = FitOnlyGate()
    groups = split_data()
    # The parent must be verified before any training batch is admitted.
    try:
        gate.minibatch(groups["fit"][:2])
    except AssertionError:
        pass
    else:
        raise AssertionError("unverified parent admitted")
    # Exercise batch gate without pretending physical FP16 verification.
    gate.parent_verified = True  # TEST ONLY; never set in training code.
    assert len(gate.minibatch(groups["fit"][:2])) == 2
    for key in ("grammar_guard", "topology_guard", "double_guard"):
        try:
            gate.minibatch([groups["fit"][0], groups[key][0]])
        except AssertionError:
            pass
        else:
            raise AssertionError("guard admitted: " + key)
    try:
        gate.minibatch([groups["fit"][0], groups["fit"][0]])
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate admitted")
    print("S3BY loader negative controls PASS; no weights or labels loaded; output_gate=0")


if __name__ == "__main__":
    self_test()
