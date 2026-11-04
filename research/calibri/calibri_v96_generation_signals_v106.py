#!/usr/bin/env python3
"""Inspect ORIGINAL immutable pretrained v96 generated texts without model reruns."""
from pathlib import Path
import argparse
from v106.audit import run

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    run(a.source,a.out)
