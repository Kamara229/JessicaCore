#!/usr/bin/env python3
"""Compare independent Switch MoE runs without tolerances or model downloads.

Scientific gate: identical checkpoint and dataset token hashes do not imply
identical numerical output, and within-process replay is not cross-run proof.
Never change model weights or production files. Only JSON evidence is consumed.
"""
import argparse
import hashlib
import json
from pathlib import Path


PHASES = ("original_eval", "zero_router_jitter_eval", "int5_eval", "restored_eval")
CORPORA = ("WikiText2_validation", "TinyStories_validation")
FIELDS = ("logits_sha256", "router_mask_sha256")


def compare(left, right):
    mismatches = []
    invariants = (
        "experiment", "source_model", "model_revision",
        "dataset_revisions", "evaluation_token_sha256",
        "model_library", "torch_version", "cpu_arch", "torch_threads",
        "torch_mkldnn_enabled", "router_layer_count",
        "source_key_weight_SHA256", "loading_info",
    )
    def check(label, a, b):
        if a != b:
            mismatches.append(label)

    for key in invariants:
        check(key, left.get(key), right.get(key))
    for name, result in (("run_1", left), ("run_2", right)):
        check(name + ".all_weights_restored", result.get("all_8_expert_pair_weights_restored_exactly"), True)
        check(name + ".original_vs_restored", result.get("original_vs_restored_logits_bitwise_identical_with_zero_jitter"), True)
        info = result.get("loading_info", {})
        for field in ("missing_key_count", "unexpected_key_count", "mismatched_key_count"):
            check(f"{name}.loading_info.{field}", info.get(field), 0)
        for phase in PHASES:
            for corpus in CORPORA:
                group = result.get("traces", {}).get(phase, {}).get(corpus, {})
                for field in ("all_logits_bitwise_identical", "all_router_masks_bitwise_identical"):
                    check(f"{name}.{phase}.{corpus}.{field}", group.get(field), True)
                runs = group.get("runs", [])
                if len(runs) != 3:
                    mismatches.append(f"{name}.{phase}.{corpus}.repeat_count")
    for phase in PHASES:
        for corpus in CORPORA:
            try:
                a = left["traces"][phase][corpus]["runs"][0]
                b = right["traces"][phase][corpus]["runs"][0]
            except (KeyError, IndexError, TypeError):
                mismatches.append(f"{phase}.{corpus}.missing_trace")
                continue
            for field in FIELDS:
                check(f"{phase}.{corpus}.{field}", a.get(field), b.get(field))
            check(f"{phase}.{corpus}.first_sequence_logits_checksum",
                  a.get("first_sequence_logits_checksum"),
                  b.get("first_sequence_logits_checksum"))
    return sorted(set(mismatches))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first", required=True)
    p.add_argument("--second", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    first, second = Path(args.first), Path(args.second)
    left = json.loads(first.read_text(encoding="utf-8"))
    right = json.loads(second.read_text(encoding="utf-8"))
    problems = compare(left, right)
    report = {
        "experiment": "CALIBRI_v26_independent_CPU_runner_MoE_replay",
        "pass": not problems,
        "run_count": 2,
        "identical_logit_and_router_hashes_across_runner_images": not problems,
        "diff_fields": problems,
        "source_json_sha256": {
            "ubuntu_2204": hashlib.sha256(first.read_bytes()).hexdigest(),
            "ubuntu_2404": hashlib.sha256(second.read_bytes()).hexdigest(),
        },
        "interpretation": "Exact replay across two independent fresh hosted runners"
            if not problems else "Between-run or between-image mismatch: prior tiny MoE KL gains are NOT reliable",
        "scope": "No quality/size/throughput breakthrough, PPL or QMoE reproduction claimed.",
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "diff_fields": problems}, ensure_ascii=False))
    # Negative results remain uploaded and visible. CI fails CLOSED on mismatches.
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
