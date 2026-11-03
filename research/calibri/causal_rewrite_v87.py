#!/usr/bin/env python3
"""Thin CLI facade for research-only CALIBRI v87."""
import argparse
from pathlib import Path
from v87 import run
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--adaptive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.original,a.adaptive,a.output)
if __name__=="__main__":
    main()
