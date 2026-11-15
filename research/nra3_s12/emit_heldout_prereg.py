"""Output S12 full-annotation heldout SHA manifest BEFORE any model run.

Deliberately no tokenizer, no original 43M forward, no CB59, no selector
inference: this is a preregistration only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from research.nra3_s12.heldout_ru_eval16 import (
  sealed_eval16,build_eval16,canonical_eval16
)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    a.add_argument("--include-annotations",action="store_true")
    args=a.parse_args()
    if args.out.exists():
        raise FileExistsError("S12 EVAL source prereg cannot overwrite")
    r=sealed_eval16()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
                        encoding="utf-8")
    if args.include_annotations:
        p=args.out.parent/"NRA3_S12_LOCKED_RUSSIAN_EVAL16_SOURCE_ANNOTATIONS.json"
        if p.exists():
            raise FileExistsError("S12 EVAL annotations cannot overwrite")
        p.write_text(json.dumps(
          json.loads(canonical_eval16(build_eval16()).decode("utf-8")),
          sort_keys=True,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(r,sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
