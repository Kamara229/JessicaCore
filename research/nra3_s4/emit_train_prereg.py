"""Publish immutable Russian UD TRAIN-only SHA evidence, no training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from research.nra3_s4.russian_ud_train_only import build_train,commitment


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("S4 refuses to overwrite an existing corpus commitment")
    report=commitment(build_train())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
