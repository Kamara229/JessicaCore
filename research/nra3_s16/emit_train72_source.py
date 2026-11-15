"""Pre-register S16 directed graph labels on existing TRAIN72 only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from research.nra3_s16.direct_train72_edges import (
    build_directed_train72,canonical_directed_train72,sealed_directed_train72
)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("S16 directed TRAIN-only manifest is append-only")
    source=canonical_directed_train72(build_directed_train72())
    m=sealed_directed_train72()
    assert hashlib.sha256(source).hexdigest()==m[
        "S16_complete_predicate_to_argument_gold_source_sha256"]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(m,sort_keys=True,
                       ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    path=a.out.parent/"NRA3_S16_DIRECTED_AGENT_ACTION_OBJECT_TRAIN72_FULL_GOLD.json"
    if path.exists():
        raise FileExistsError("S16 cannot overwrite TRAIN-only physical labels")
    path.write_text(json.dumps(json.loads(source),sort_keys=True,
                    ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(m,sort_keys=True,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
