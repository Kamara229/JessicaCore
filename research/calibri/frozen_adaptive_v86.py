#!/usr/bin/env python3
"""Only v86 CLI facade. Independent TEST with frozen original CB59 and CB85."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--adaptive",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("results/v86"))
    args=p.parse_args()
    from v86 import run
    run(args.original,args.adaptive,args.output)


if __name__=="__main__":
    main()
