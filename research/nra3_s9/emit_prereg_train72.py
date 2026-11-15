"""S9 new 72 (TRAIN only) immutable UD annotation preregistration.

Archives only manifest+SHA, not real Calibri input hidden or independent
EVAL text. No original source modified and output gate remains zero.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from research.nra3_s9.new_ru_train72 import (
    build_s9_train72,seal_train72,canonical_train72,
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--include-train-annotations",action="store_true")
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("S9 refuses overwrite of a sealed TRAIN72 source")
    report=seal_train72()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,
                                  ensure_ascii=False,indent=2)+"\n",
                        encoding="utf-8")
    if args.include_train_annotations:
        gold=args.out.with_name("NRA3_S9_RU_UD_TRAIN72_ANNOTATIONS_ONLY.json")
        if gold.exists():
            raise FileExistsError("S9 never overwrites an annotation file")
        json_data=json.loads(canonical_train72(build_s9_train72()).decode("utf-8"))
        gold.write_text(json.dumps(json_data,sort_keys=True,
                                   ensure_ascii=False,indent=2)+"\n",
                        encoding="utf-8")
    print(json.dumps(report,sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
