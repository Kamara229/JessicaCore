#!/usr/bin/env python3
"""Forensic real-answer and text-generation audit, frozen original sources."""
import argparse
from pathlib import Path
from v96.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ("original","v92","v92_report","locked_baseline","out"):
        p.add_argument("--"+arg.replace("_","-"),required=True,type=Path)
    a=p.parse_args()
    run(a.original,a.v92,a.v92_report,a.locked_baseline,a.out)
