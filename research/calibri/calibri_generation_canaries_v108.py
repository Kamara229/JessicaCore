#!/usr/bin/env python3
"""New chat-template original C-native output audit, no test through Jessica."""
from pathlib import Path
import argparse
from v108.pipeline import run
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--candidate",required=True,type=Path)
    p.add_argument("--v102-report",required=True,type=Path)
    p.add_argument("--locked",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run(a.original,a.candidate,a.v102_report,a.locked,a.out)
