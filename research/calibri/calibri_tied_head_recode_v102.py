#!/usr/bin/env python3
"""CALIBRI v102 real same-byte whole pretrained tied-head source rewrite trial."""
from pathlib import Path
import argparse
from v102.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--locked",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    run(args.original,args.locked,args.out)
