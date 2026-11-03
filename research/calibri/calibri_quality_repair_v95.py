#!/usr/bin/env python3
"""CALIBRI behavioral quality guided whole-block search, same physical bytes."""
import argparse
from pathlib import Path
from v95.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--v92",required=True,type=Path)
    p.add_argument("--v92-report",required=True,type=Path)
    p.add_argument("--locked-baseline",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    run(args.original,args.v92,args.v92_report,args.locked_baseline,args.output)
