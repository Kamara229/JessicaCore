#!/usr/bin/env python3
"""Tiny CLI facade. No additional trained weight data lives in this script."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--v84-rewritten",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("results/v85"))
    a=parser.parse_args()
    from v85 import run
    run(a.original,a.v84_rewritten,a.output)


if __name__=="__main__":
    main()
