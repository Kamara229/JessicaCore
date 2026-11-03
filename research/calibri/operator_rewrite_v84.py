#!/usr/bin/env python3
"""Tiny CLI facade: all source, training and validation in v84/*.py."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/v84"))
    args = p.parse_args()
    from v84 import run
    run(args.original,args.out_dir)


if __name__ == "__main__":
    main()
