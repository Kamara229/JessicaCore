#!/usr/bin/env python3
"""Run v97 read-only source-verified compressed neural fault attribution."""
import argparse
from pathlib import Path
from v97.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--locked",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run(a.original,a.locked,a.out)
