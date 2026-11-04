#!/usr/bin/env python3
"""Immutable real CB59 head source encode/decode parity vs original FP32."""
from pathlib import Path
import argparse
from v101.pipeline import run
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--locked",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.original,args.locked,args.out)
