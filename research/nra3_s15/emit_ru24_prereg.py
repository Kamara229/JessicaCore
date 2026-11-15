"""Emit post-S14 frozen 24 new Russian EVAL source SHA WITHOUT model calls."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from research.nra3_s15.new_ru_eval24_prereg import (
    new_russian_eval24,canonical_new_eval24,sealed_new_eval24
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("S15 refuses overwrite of one-time model-independent prereg")
    source=canonical_new_eval24(new_russian_eval24())
    record=sealed_new_eval24()
    if hashlib.sha256(source).hexdigest()!=record[
        "complete_UD_semantic_role_coreference_event_gold_sha256"]:
        raise ValueError("S15 canonical EVAL24 changed during source-only seal")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,indent=2,sort_keys=True,
                                   ensure_ascii=False)+"\n",encoding="utf-8")
    labels=args.out.parent/"NRA3_S15_NEW_RU24_SOURCE_ONLY_LOCKED_GOLD.json"
    if labels.exists():
        raise FileExistsError("S15 EVAL24 annotation source can never overwrite")
    labels.write_text(json.dumps(json.loads(source.decode("utf-8")),
                                 indent=2,sort_keys=True,ensure_ascii=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(record,ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
