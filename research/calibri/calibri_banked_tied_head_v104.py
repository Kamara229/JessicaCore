#!/usr/bin/env python3
"""Frozen original trained entire model, physically paid 8-bank C output head."""
import argparse
from pathlib import Path
from v104.pipeline import run

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    ap.add_argument("--locked",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    run(a.source,a.locked,a.out)
