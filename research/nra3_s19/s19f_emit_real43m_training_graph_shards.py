"""S19f materialize frozen original43M gold-blind word338 separately from graph gold.

Research-only: physically split neural features and TRAIN graph supervision
as distinct safetensors files, each SHA-locked, never train or score a model.
Consumes **only** S19e REAL original CPU hidden320 and sealed S19 source9.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from research.nra3_s19.s19f_real43m_train9_graph_targets import (
    join_genuine_original43m_train9, train9_source_manifest,
)


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-hidden",required=True,type=Path)
    p.add_argument("--source-report",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("S19f cannot overwrite previous physical source")
    joined=join_genuine_original43m_train9(a.source_hidden,a.source_report)
    a.out.mkdir(parents=True,exist_ok=False)
    inp=a.out/"S19F_TRUE43M_GT_NEW9_RAWONLY_WORD338.safetensors"
    tgt=a.out/"S19F_HUMAN_NEW9_GRAPH_GOLD_TARGETS_ONLY.safetensors"
    features={
        "raw_text_frozen_word338":joined.inputs.raw_word_hidden320_and_raw18,
        "valid_word_mask":joined.inputs.valid_word_mask,
    }
    gold={
        "event_status_TRAIN_ONLY":joined.targets.event_status,
        "predicate_argument_role_TRAIN_ONLY":joined.targets.predicate_argument_role,
        "argument_polarity_TRAIN_ONLY":joined.targets.argument_polarity,
        "coreference_TRAIN_ONLY":joined.targets.coreference,
        "event_relation_TRAIN_ONLY":joined.targets.event_links,
    }
    # All human annotations physically segregated from the ONLY two
    # gold-blind model-feature tensors and are never passed as model inputs.
    assert all(torch.isfinite(t.float()).all() for t in features.values())
    save_file(features,str(inp))
    save_file(gold,str(tgt))
    for path,values in ((inp,features),(tgt,gold)):
        loaded=load_file(str(path),device="cpu")
        if set(loaded)!=set(values) or any(
                not torch.equal(loaded[k],v) for k,v in values.items()):
            raise AssertionError("S19f exact SHA physical tensor reload mismatch")
    if set(features) & set(gold):
        raise AssertionError("S19f human target tensors leaked into model input")
    manifest=train9_source_manifest(joined)
    manifest.update({
        "gold_blind_only_feature_file":inp.name,
        "gold_blind_only_feature_sha256":sha(inp),
        "source_human_targets_filename":tgt.name,
        "source_human_targets_sha256":sha(tgt),
        "inputs_contains_exactly_no_labels":list(sorted(features)),
        "targets_kept_in_separate_physical_file":list(sorted(gold)),
        "new9_realCPU_full_tensor_roundtrip":True,
        "optimizer_steps":0,
    })
    report=a.out/"S19F_TRUE43M_GT_NEW9_SEPARATE_TRAIN_GRAPH_TARGETS_MANIFEST.json"
    report.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,sort_keys=True)+"\n",
                      encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
