"""S22c: physically separate true original43M+GT 256 raw input from TRAIN gold.

No training/model forward, no independent EVAL or protected J7 replay here.
Original J7/hidden SHA fully validated by S22b parent artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file,load_file

from research.nra3_s20.s22c_real43m_new256_graph_target_join import (
    join_original43m_new256,EXPECTED_COUNTS,
)
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    splits,manifest,
)


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-hidden",type=Path,required=True)
    p.add_argument("--source-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("S22c will never overwrite existing research source")
    joined=join_original43m_new256(a.source_hidden,a.source_report)
    a.out.mkdir(parents=True,exist_ok=False)
    inputs=a.out/"S22C_REAL43M_GT_NEW256_RAWONLY_WORD338.safetensors"
    gold=a.out/"S22C_256_SYNTHETIC_EVENT_GRAPH_TARGETS_TRAIN_ONLY.safetensors"
    raw={
        "raw_text_frozen_word338":joined.inputs.raw_word_hidden320_and_raw18,
        "valid_word_mask":joined.inputs.valid_word_mask,
    }
    targets={
        "event_status_TRAIN_ONLY":joined.gold.event_status,
        "predicate_argument_role_TRAIN_ONLY":joined.gold.predicate_argument_role,
        "argument_polarity_TRAIN_ONLY":joined.gold.argument_polarity,
        "coreference_TRAIN_ONLY":joined.gold.coreference,
        "event_relation_TRAIN_ONLY":joined.gold.event_links,
    }
    if set(raw)&set(targets):
        raise ValueError("S22c gold must never appear in model input")
    for tensor in raw.values():
        if not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError("S22c nonfinite actual raw input")
    save_file(raw,str(inputs))
    save_file(targets,str(gold))
    for file,src in ((inputs,raw),(gold,targets)):
        loaded=load_file(str(file),device="cpu")
        if set(loaded)!=set(src) or any(
            not torch.equal(loaded[k],value) for k,value in src.items()):
            raise ValueError("S22c physical shard save/load changed")
    counts=tuple(int(x) for x in (
        (targets["event_status_TRAIN_ONLY"]>0).sum(),
        (targets["predicate_argument_role_TRAIN_ONLY"]>0).sum(),
        (targets["argument_polarity_TRAIN_ONLY"]==1).sum(),
        (targets["coreference_TRAIN_ONLY"]==1).sum(),
        (targets["event_relation_TRAIN_ONLY"]>0).sum(),
    ))
    if counts!=EXPECTED_COUNTS:
        raise ValueError("S22c human synthetic gold counts drift")
    m={
        "stage":"NRA3_S22C_REAL_ORIGINAL43M_GT_NEW256_SEPARATE_GRAPH_TRAIN_SHARDS",
        "S22b_true_real_CPU_capture_run":"PINNED_IN_WORKFLOW",
        "S22_original_protected_616_J7_parent_unchanged":True,
        "S22_sha_locked_synthetic_TRAIN256_graph_gold":joined.graph_sha,
        "S22b_original43M_GT_CPU_hidden320_FP16_sha":joined.original_cpu_hidden_sha,
        "256_curated_synthetic_train_texts":256,
        "all_real_frozen_Calibri_word_vectors":int(joined.inputs.valid_word_mask.sum()),
        "TRAIN_only_events":counts[0],
        "TRAIN_only_directed_role_links":counts[1],
        "TRAIN_only_rejected_recipient_links":counts[2],
        "TRAIN_only_coreference_edges":counts[3],
        "TRAIN_only_event_relations":counts[4],
        "strict_predeclared_lexical_and_syntax_TRAIN_folds":len(splits()),
        "genuine_original_CPU_hidden_plus_plain_raw18_ONLY_as_model_inputs":True,
        "human_graph_targets_NEVER_passed_to_original_or_adapter_forward":True,
        "feature_keys":list(sorted(raw)),
        "human_gold_keys_in_separate_file":list(sorted(targets)),
        "raw338_word_features_filename":inputs.name,
        "raw338_word_features_SHA256":sha(inputs),
        "TRAIN_only_gold_filename":gold.name,
        "TRAIN_only_gold_SHA256":sha(gold),
        "no_new_optimizer_no_model_scoring_or_quality_claim":True,
        "all_256_from_template_generated_TRAIN_not_independent_natural_Russian":True,
        "no_consumed_EVAL32_EVAL128_S17c_S15c_S12c":True,
        "new_active_candidate_616_J7_CPU_bpp_NOT_CERTIFIED":True,
        "original_LIVE_Render_Supabase_Jessica_APK_unchanged":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
    report=a.out/"S22C_REAL43M_GT_NEW256_RAWONLY_AND_TRAIN_GOLD_SEPARATE_REPORT.json"
    report.write_text(json.dumps(m,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                      encoding="utf-8")
    print("S22C_REAL43M_GT_TRAIN256_GOLD_BLIND_INPUT_SHA",m["raw338_word_features_SHA256"],flush=True)
    print("S22C_HUMAN_SYNTHETIC_TRAIN256_GOLD_SHA",m["TRAIN_only_gold_SHA256"],flush=True)
    print("S22C_REAL_CPU_NO_OPTIMIZER_FIVE_GRAPH_TARGET_COUNTS",counts,flush=True)


if __name__=="__main__":
    main()
