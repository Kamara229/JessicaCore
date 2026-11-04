"""Reuse original predeclared twenty-four diagnostic queries; no fit."""
from v120.protocol import VALID_A,VALID_B
TOP_K=(1,8,32,128,512)
CASES=VALID_A+VALID_B
assert len(CASES)==24 and len({q for q,ref in CASES})==24
