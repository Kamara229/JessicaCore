"""S3BF research-only: physically pin S3BA/S3BE and preregister blind EVAL128.

This program NEVER imports or builds evaluation prompts, uses a model, runs
tokenization, scores DEV/EVAL, or updates weights. Native 43M/Green Tunnel/NRA,
Jessica LIVE, Render, Supabase, APK, main and QA/TEST stay untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

S3BA_SHA = "4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
S3BE_SHA = "9b6f0c455492c1fd508c38476a3505900898ef2ae08907329a3953751a93e513"
S3AQ_PREREG_SHA = "c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839"
S3BE_CI = 43593460123
S3BA_CI = 43590032315
GROUPS = {
    "S3M_TRAIN128": 128, "S3X_TRAIN128": 128, "S3AQ_TRAIN256": 256,
    "S3AU_TRAIN512": 512, "S3AY_TRAIN512": 512, "S3BC_TRAIN512": 512,
}

def require(condition: bool, reason: str) -> None:
    if not condition:
        raise AssertionError("S3BF fail-closed: " + reason)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_safetensors_raw(path: Path) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Pure-stdlib independent physical tensor byte comparison; no torch/model."""
    raw = path.read_bytes()
    require(len(raw) > 16, "FP16 physical file too short")
    hlen = struct.unpack("<Q", raw[:8])[0]
    require(0 < hlen < 2_000_000 and hlen + 8 < len(raw), "invalid safetensors header")
    head = json.loads(raw[8:8+hlen])
    base = 8 + hlen
    tensors = {}
    intervals = []
    for key, entry in head.items():
        if key == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        dims = tuple(entry["shape"])
        dtype = entry["dtype"]
        require(dtype == "F16", "FP16 ABI drift: " + key)  # scalar shape [] is valid
        require(all(type(d) is int and d >= 0 for d in dims), "invalid shape")
        count = 1
        for d in dims:
            count *= d
        require(0 <= start <= end <= len(raw)-base and end-start == 2*count,
                "invalid physical tensor offsets: " + key)
        intervals.append((start, end))
        tensors[key] = (dtype, dims, raw[base+start:base+end])
    require(len(tensors) >= 10, "missing weights")
    require(sorted(intervals)[0][0] == 0, "data offset not zero")
    previous = 0
    for start, end in sorted(intervals):
        require(start == previous, "tensor gaps or overlaps")
        previous = end
    require(previous == len(raw)-base, "trailing unverified bytes")
    return tensors

def metric_total(report: dict, name: str) -> int:
    groups = report[name]
    require(set(groups) == set(GROUPS), "unexpected TRAIN group")
    total = 0
    for group, n in GROUPS.items():
        all_scores = groups[group]["all"]
        require(all_scores["count"] == n, "TRAIN count mismatch " + group)
        correct = all_scores["both_correct"]
        require(0 <= correct <= n, "TRAIN accuracy invalid " + group)
        require(all_scores["source_correct"] >= correct
                and all_scores["target_correct"] >= correct,
                "TRAIN role metrics inconsistent " + group)
        total += correct
    return total

def main(a_dir: Path, b_dir: Path, out: Path) -> None:
    parent = a_dir/"NRA2_S3BA_S3AW_INIT_ROLE1536_FP16.safetensors"
    candidate = b_dir/"NRA2_S3BE_S3BA_INIT_ROLE2048_FP16.safetensors"
    audit_path = b_dir/"NRA2_S3BE_S3BA_INIT_TRAIN2048_NO_DEV_ROLE_AUDIT.json"
    for file in (parent, candidate, audit_path):
        require(file.is_file(), "missing pinned artifact file " + file.name)
    require(sha(parent) == S3BA_SHA, "S3BA physical parent SHA")
    require(sha(candidate) == S3BE_SHA, "S3BE physical candidate SHA")
    audit = json.loads(audit_path.read_text())
    require(audit["stage"] == "NRA2_S3BE_S3BA_INIT_2048_TRAIN_ONLY_ROLE_HEAD_NO_DEV_SCORE",
            "S3BE stage drift")
    require(audit["S3BA_parent_fp16_sha256"] == S3BA_SHA
            and audit["new_S3BE_physical_fp16_sha256"] == S3BE_SHA
            and audit["S3BA_pinned_parent_run"] == S3BA_CI,
            "model provenance mismatch")
    require(audit["train_groups"] == GROUPS and audit["train_count"] == 2048,
            "TRAIN2048 source drift")
    require(audit["optimizer_steps"] == 160 and audit["seed"] == 2026111498,
            "locked optimizer drift")
    require(audit["TRAIN_loss_after"] < audit["TRAIN_loss_before"],
            "reported TRAIN loss did not decrease")
    require(audit["original_616_S194_J7_digests_verified_from_six_true_CPU_sources"]
            and audit["original_graph_physical_fp16_bitwise_equal"]
            and audit["original_frozen_43M_normalizer_bitwise_equal"]
            and audit["no_DEV128_loaded_into_optimizer_or_scored"]
            and audit["S3AQ_future_EVAL128_never_CPU_prefilled_scored_or_trained"]
            and audit["no_heldout_model_inference_this_stage"]
            and audit["only_query_role_head_trained"]
            and audit["original_graph_43M_green_tunnel_NRA_frozen"]
            and audit["output_gate"] == 0.0
            and audit["candidate_accepted"] is False,
            "research-only protected gate failed")
    pa = load_safetensors_raw(parent)
    ca = load_safetensors_raw(candidate)
    require(set(pa) == set(ca), "FP16 parameter names mismatch")
    changed_query = []
    frozen_count = 0
    for key in sorted(pa):
        if key.startswith("bridge.query."):
            require(pa[key][:2] == ca[key][:2], "query ABI drift " + key)
            if pa[key][2] != ca[key][2]:
                changed_query.append(key)
        else:
            require(pa[key] == ca[key], "unauthorized changed frozen tensor: " + key)
            frozen_count += 1
    require(bool(changed_query) and frozen_count > 0,
            "only physical query-role weights may change")
    before = metric_total(audit, "TRAIN_only_parent_role_metrics")
    after = metric_total(audit, "TRAIN_only_S3BE_role_metrics")
    require(before == 1848 and after == 2043, "S3BE known TRAIN-only score drift")
    require(audit["TRAIN_only_parent_role_metrics"]["S3BC_TRAIN512"]["all"]["both_correct"] == 314
            and audit["TRAIN_only_S3BE_role_metrics"]["S3BC_TRAIN512"]["all"]["both_correct"] == 508,
            "source-target contrastive TRAIN score drift")

    # Immutable protocol; note: NO future EVAL rows, hashes or labels opened here.
    protocol = {
        "stage": "NRA2_S3BF_PRECOMMIT_ONE_TIME_BLIND_S3AQ_EVAL128_NO_SCORE",
        "status": "PREREG_ONLY_NO_EVAL_CPU_CAPTURE_NO_SCORING_NO_MODEL_UPDATE",
        "parent_run": S3BA_CI, "candidate_run": S3BE_CI,
        "parent_fp16_sha256": S3BA_SHA, "candidate_fp16_sha256": S3BE_SHA,
        "original_S3AQ_all_splits_prereg_commitment_sha256": S3AQ_PREREG_SHA,
        "blind_split_identifier": "S3AQ_FUTURE_INDEPENDENT_EVAL128",
        "blind_n": 128, "blind_unseen_topologies": 32,
        "blind_four_unseen_grammars_each": 32,
        "blind_destination_first_n": 64, "blind_source_first_n": 64,
        "prereg_roles_only_primary": "Both query source and target identified correctly on the same 128 items",
        "primary_pair": ["S3BA_FROZEN_FP16", "S3BE_FROZEN_FP16"],
        "paired_decision": "Compare same item ID, both-correct delta and exact two-sided McNemar p<=0.05; no resampling or threshold tuning",
        "secondary_fixed": [
            "source-only 128", "target-only 128", "both-role by four grammars",
            "both-role destination-first 64 vs source-first 64",
            "frozen graph reachability via predicted roles vs immutable BFS gold",
            "paired exact newly-right vs newly-wrong counts",
        ],
        "next_separate_stage_only_after_this_precommit": [
            "Authentic unmodified original frozen 43M CPU tokens, same blinded EVAL inputs for both heads",
            "Use independently parsed literal query mentions, never gold source/target to bind predicted roles",
            "Hash raw EVAL inputs and original J7=616 before inference, record run and every checkpoint SHA",
            "One-time evaluation of both locked FP16 heads without gradient, calibration, manual rephrasing or checkpoint selection",
        ],
        "if_test_fails": "Retire this blind set from future independent evaluation: do not retrain or re-score based on it",
        "cannot_accept_without": [
            "Separate physical full-EOS quality evidence",
            "Separate all-in <=0.7 bpp measurement",
            "Paired original CPU speed >=90% baseline measurement",
            "Original protected 616 J7 no loss and no production change",
        ],
        "forbidden": [
            "Access S3AQ EVAL128 prompts, tokens, truth labels or hashes at prereg stage",
            "Score or optimize on previously repeatedly viewed S3AQ DEV128",
            "Change source model, graph, Green Tunnel, NRA, QA/TEST, main, Jessica LIVE, Render, Supabase, APK",
        ],
        "output_gate": 0.0, "candidate_accepted": False,
    }
    payload = json.dumps(protocol, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()
    out.mkdir(parents=True, exist_ok=False)
    (out/"NRA2_S3BF_BLIND_EVAL128_PROTOCOL.json").write_bytes(payload+b"\n")
    result = {
        "stage": "NRA2_S3BF_READONLY_PHYSICAL_AUDIT_AND_BLIND_PREREG",
        "parent_sha256": S3BA_SHA, "candidate_sha256": S3BE_SHA,
        "frozen_tensor_count_bitwise_identical": frozen_count,
        "changed_query_tensors_only": changed_query,
        "TRAIN_only_parent_both_correct": before,
        "TRAIN_only_candidate_both_correct": after,
        "TRAIN_only_net_gain_not_generalization": after-before,
        "TRAIN_only_S3BC_parent_both_correct": 314,
        "TRAIN_only_S3BC_candidate_both_correct": 508,
        "no_blind_or_DEV_model_scoring": True,
        "future_EVAL128_unopened": True,
        "protocol_sha256": hashlib.sha256(payload).hexdigest(),
        "output_gate": 0.0, "candidate_accepted": False,
    }
    (out/"NRA2_S3BF_PHYSICAL_AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, sort_keys=True), flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: stage3bf_blind_eval128_prereg.py S3BA_DIR S3BE_DIR OUTPUT_DIR")
    main(*(Path(v) for v in sys.argv[1:]))
