#!/usr/bin/env python3
"""One-stage nonproduction original pretrained TRAIN diagnosis."""
import argparse
from pathlib import Path
from v90.pipeline import run

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--original",required=True,type=Path)
    ap.add_argument("--cb88",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    run(args.original,args.cb88,args.output)

if __name__=="__main__":
    main()
