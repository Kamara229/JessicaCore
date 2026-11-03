#!/usr/bin/env python3
"""Review earlier CALIBRI v92 source by observable answers and generation only."""
import argparse
from pathlib import Path
from v93.pipeline import run

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--v92",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.v92,args.output)

if __name__=="__main__":main()
