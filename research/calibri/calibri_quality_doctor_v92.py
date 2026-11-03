#!/usr/bin/env python3
"""Automated research diagnosis and bounded full-block rewrite trial, never live."""
import argparse
from pathlib import Path
from v92.pipeline import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--cb88",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.report,a.original,a.cb88,a.output)

if __name__=="__main__":main()
