"""Locate causal family effects after v173 ruled out tested native decode errors.

Eight complete-model dense diagnostic arms, 96 direct questions per arm.
Every combination of compressed embedding/head, attention, and MLP is tested.
FP32-containing arms have NO compact storage or native throughput claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from v84.sources import original_teacher
from v169.sources import all_cases
from .protocol import (MASKS, FAMILIES, SOURCE_BYTES, SOURCE_SHA, TOTAL_UNIQUE,
                       MAX_NEW_TOKENS, arm_name, freeze_cases)
from .intervene import extract_original, snapshots, apply_mask
from .answers import generate, annotate, summary


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def run(source, out):
    torch.set_num_threads(2)
    torch.manual_seed(174)
    out.mkdir(parents=True, exist_ok=True)
    cases = freeze_cases(all_cases())
    frozen = {
        "stage": "v174_full_factorial_source_family_localization",
        "model": MODEL, "model_revision": MODEL_SHA,
        "source_sha256": SOURCE_SHA, "source_bytes": SOURCE_BYTES,
        "immutable_original_bpp": 8 * SOURCE_BYTES / TOTAL_UNIQUE,
        "families": FAMILIES, "masks": MASKS, "max_new_tokens": MAX_NEW_TOKENS,
        "cases": cases, "expected_generations": len(cases) * len(MASKS),
        "ALL_ARMS_ARE_DENSE_DIAGNOSTICS_NOT_NEW_COMPACT_CANDIDATES": True,
        "no_KL_NLL_PPL_quality_gate": True, "no_training": True,
        "no_private_v89_TEST": True, "no_production_changes": True,
        "unparsed_is_ungraded_not_automatically_wrong": True,
        "historical_development_suite_not_an_independent_final_test": True,
    }
    save(out / "protocol_before_models.json", frozen)
    archive = extract_original(source)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    model = original_teacher()
    originals, decoded, family_counts = snapshots(model, archive)
    evidence = {"protocol": frozen, "family_matrix_counts": family_counts,
                "by_arm": {}, "complete": False}
    prompt_hashes = None
    for mask in MASKS:
        name = arm_name(mask)
        quantized = apply_mask(model, originals, decoded, mask)
        rows = []
        # No cache or activation is shared between independent cases or arms.
        for case in cases:
            answer = generate(model, tokenizer, case["prompt"])
            rows.append(annotate(case, answer))
            if len(rows) % 16 == 0:
                print(json.dumps({"arm": name, "completed_cases": len(rows)}), flush=True)
                save(out / (name + "_partial.json"), rows)
        hashes = [r["prompt_sha"] for r in rows]
        if prompt_hashes is None:
            prompt_hashes = hashes
        elif prompt_hashes != hashes:
            raise ValueError("Mismatched prompts across interventions")
        result = {"mask": mask, "quantized_matrix_paths": quantized,
                  "quantized_unique_parameters": sum(originals[p].numel() for p in quantized),
                  "UNPAID_FP32_original_parameters": sum(v.numel() for p, v in originals.items()
                                                          if p not in quantized),
                  "summary": summary(rows),
                  "by_domain": {d: summary([r for r in rows if r["domain"] == d])
                                for d in sorted({r["domain"] for r in rows})},
                  "answers": rows}
        evidence["by_arm"][name] = result
        save(out / "factorial_answers.json", evidence)
        print(json.dumps({"arm": name, "summary": result["summary"]}), flush=True)
    evidence["complete"] = True
    evidence["source_unchanged"] = hashlib.sha256(archive.blob).hexdigest() == SOURCE_SHA
    if sum(len(a["answers"]) for a in evidence["by_arm"].values()) != 768:
        raise AssertionError("Incomplete full factorial")
    save(out / "factorial_answers.json", evidence)
    print(json.dumps({"stage": frozen["stage"], "complete": True,
                      "generations": 768, "source_unchanged": evidence["source_unchanged"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.source, args.out)
