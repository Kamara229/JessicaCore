"""Emit SOURCE-only complete 96 TRAIN Russian morphology/UD/role gold SHA."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from research.nra3_s18.ru96_multiverb_train_source import (
    build_s18_new_train96,canonical_s18_new96_train,seal_s18_new96_train
)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--out",type=Path,required=True)
    args=a.parse_args()
    if args.out.exists():
        raise FileExistsError("S18 TRAIN96 source SHA prereg cannot overwrite")
    rows=build_s18_new_train96()
    raw=canonical_s18_new96_train(rows)
    manifest=seal_s18_new96_train()
    if hashlib.sha256(raw).hexdigest()!=manifest[
        "S18_complete_original_96_new_TRAIN_full_annotation_SHA256"]:
        raise ValueError("S18 source changed between validation and commit")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(manifest,indent=2,sort_keys=True,
                   ensure_ascii=False)+"\n",encoding="utf-8")
    full=args.out.parent/"NRA3_S18_NEW96_MULTIVERB_300_DIRECTED_TRAIN_GOLD.json"
    if full.exists():
        raise FileExistsError("S18 original96 TRAIN gold cannot overwrite")
    full.write_text(json.dumps(json.loads(raw),indent=2,
                 sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(manifest,sort_keys=True,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
