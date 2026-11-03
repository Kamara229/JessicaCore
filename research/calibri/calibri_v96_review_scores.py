#!/usr/bin/env python3
"""Reread immutable v96 output, fixing false-negative interpretation only."""
import argparse
from pathlib import Path
from v96.answer_review import run
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();run(a.source,a.out)
