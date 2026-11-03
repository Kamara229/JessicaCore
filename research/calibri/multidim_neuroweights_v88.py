#!/usr/bin/env python3
"""Thin CLI to physically paid 4D original-pretrained neuro-weights research."""
from pathlib import Path
import argparse
from v88 import run
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--frozen-cb85",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.original,a.frozen_cb85,a.output)
if __name__=="__main__":
    main()
