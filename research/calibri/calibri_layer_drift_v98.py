#!/usr/bin/env python3
"""CALIBRI v98 four fixed same-token-prefix FP32-vs-native CB59 probes."""
from pathlib import Path
import argparse
from v98.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--locked",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run(a.original,a.locked,a.out)
