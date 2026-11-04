#!/usr/bin/env python3
"""Evaluate original v106 text signals without changing frozen historical models."""
from pathlib import Path
import argparse
from v107.crossrun import run
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--v103",type=Path,required=True)
    p.add_argument("--v104",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.v103,a.v104,a.out)
