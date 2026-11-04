#!/usr/bin/env python3
"""CALIBRI v99 frozen-source exact FP32 hidden-boundary oracle, diagnostic only."""
import argparse
from pathlib import Path
from v99.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--locked",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.original,args.locked,args.out)
