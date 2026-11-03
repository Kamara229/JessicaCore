#!/usr/bin/env python3
"""Thin CLI facade for v83: use nested v83.study and v83.measure modules."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True,
                   help="complete original pretrained CB59 physical source")
    p.add_argument("--output", type=Path, default=Path("results/v83_layer_atlas_TRAIN.json"))
    args = p.parse_args()
    from v83 import run
    run(args.original, args.output)


if __name__ == "__main__":
    main()
