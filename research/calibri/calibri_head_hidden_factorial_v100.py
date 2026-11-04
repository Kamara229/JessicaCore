#!/usr/bin/env python3
"""Frozen original CB59, 2x2 original-FP32 residual/head diagnostic oracle."""
from pathlib import Path
import argparse
from v100.pipeline import run
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--locked",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run(a.original,a.locked,a.out)
