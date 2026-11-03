#!/usr/bin/env python3
"""Real frozen physical compressed source fidelity by behavior, not teacher likeness."""
import argparse
from pathlib import Path
from v94.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--prior-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.original,a.candidate,a.prior_report,a.out)
