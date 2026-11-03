#!/usr/bin/env python3
"""Thin CALIBRI v89 public CLI; no experiment selection on official TEST."""
import argparse
from pathlib import Path
from v89 import run
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--cb85",required=True,type=Path)
    p.add_argument("--cb88",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    run(a.original,a.cb85,a.cb88,a.output)
if __name__=="__main__":
    main()
