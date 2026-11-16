"""Emit source-only complete new EVAL24 SHA without any model forward."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from research.nra3_s17.new_ru24_directed_eval_prereg import (
    build_new_eval24,canonical_eval24,preregister_new_eval24
)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("S17 one-time EVAL source manifest cannot overwrite")
    r=preregister_new_eval24()
    source=canonical_eval24(build_new_eval24())
    if hashlib.sha256(source).hexdigest()!=r[
        "S17_new_Russian_EVAL_complete_UD_and_directed_link_SHA256"]:
        raise ValueError("S17 annotated data mutated while sealing")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,sort_keys=True,
                                  ensure_ascii=False,indent=2)+"\n",
                     encoding="utf-8")
    gold=a.out.parent/"NRA3_S17_NEW_RUSSIAN24_DIRECTED_PREDICATE_ARGUMENT_GOLD_SOURCE_ONLY.json"
    if gold.exists():
        raise FileExistsError("S17 no source overwrite allowed")
    gold.write_text(json.dumps(json.loads(source),sort_keys=True,
                         ensure_ascii=False,indent=2)+"\n",
                    encoding="utf-8")
    print(json.dumps(r,sort_keys=True,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
