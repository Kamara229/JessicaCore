#!/usr/bin/env python3
"""Pinned pretrained chat and labeled ground truth, physically unchanged v102."""
from pathlib import Path
import argparse
from v103.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--candidate",required=True,type=Path)
    p.add_argument("--v102-report",required=True,type=Path)
    p.add_argument("--locked",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    run(args.original,args.candidate,args.v102_report,args.locked,args.out)
