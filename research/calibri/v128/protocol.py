"""Reuse original predeclared twenty-four diagnostic queries; no fit."""
from v120.protocol import VALID_A,VALID_B
from hashlib import sha256
TOP_K=(1,8,32,128,512)
V119_SHA256="ac01fe4e2319520aa2b845b519ddde67d2a67f3a40d098440c1cbc9193d993a0"
CASES=VALID_A+VALID_B
assert len(CASES)==24 and len({q for q,ref in CASES})==24


def read_v119(path):
    from v118.archive import Paid
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != V119_SHA256:
        raise ValueError("expected complete frozen v119 including all learned corrections")
    return Paid(raw)
