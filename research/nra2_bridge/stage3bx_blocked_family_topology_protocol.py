"""S3BX: sealed TRAIN-only factorial split after S3BT failed external transfer.

Do NOT treat these diagnostics as independent evaluation: rejected S3BT already
trained on the entire S3BR TRAIN512. Any subsequent model must start from the
physically pinned PRE-S3BR S3BN checkpoint, NEVER from S3BT. The four
independent EVAL128s are consumed and absolutely excluded from model
selection, prompts, tokenization, calibration, or reruns.

The next learner can use only split 'fit'. The other three disjoint groups
probe unseen construction families, unseen old-TRAIN graph topologies, and
both simultaneously. This step only SHA-seals a protocol and trains/scores
nothing; no original Calibri CPU, original 616 J7 or LIVE is invoked.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from research.nra2_bridge.stage3br_relation_language_train512_prereg import (
    build_rows, commitment as s3br_commitment, CONSTRUCTIONS, SPLIT as S3BR_SPLIT,
)

STAGE = "NRA2_S3BX_BLOCKED_GRAMMAR_AND_TOPOLOGY_TRAIN_ONLY_PROTOCOL"
OLD_S3BR_SHA = "e941bd45f947b7fc16f0cce4307ed9ea69c0b0e8347da7c0754717430ff0262f"
PRE_S3BR_PARENT_SHA = "ea8a81665370cba22c4e8bc75fbe82314089244d4b32278ee62c2c30d88ae811"
REJECTED_S3BT_SHA = "a700b59d8f759d816fa151c1d21d340fe1f26772ee91e9788f6fe98a123c4262"
CONSUMED_EVAL_RUNS = (43640363396, 43644153544, 43648038071, 43651233135)
SPLITS = ("fit", "grammar_guard", "topology_guard", "double_guard")
EXPECTED = {"fit": 192, "grammar_guard": 192, "topology_guard": 64, "double_guard": 64}


def require(ok, msg):
    if not ok:
        raise AssertionError("S3BX FAIL CLOSED: " + msg)


def split_data():
    rows = build_rows()  # Only prior S3BR TRAIN512. Never load any EVAL records.
    require(s3br_commitment(rows)["commitment_sha256"] == OLD_S3BR_SHA,
            "S3BR pinned TRAIN512 changed")
    require(len(CONSTRUCTIONS) == 4 and len(rows) == 512, "grammar ABI drift")
    by_graph = defaultdict(list)
    for row in rows:
        require(row.split == S3BR_SPLIT and 0 <= row.query_style < 8,
                "only original TRAIN512 is permitted")
        by_graph[row.graph_id].append(row)
    require(len(by_graph) == 64 and all(len(v) == 8 for v in by_graph.values()),
            "requires exactly 64 TRAIN-only graph blocks")

    result = {split: [] for split in SPLITS}
    for index, graph_id in enumerate(sorted(by_graph)):
        unseen_topology = index % 4 == 0  # 16 / 64, balanced deterministic holdout
        for row in sorted(by_graph[graph_id], key=lambda v: v.query_style):
            unseen_family = row.query_style // 2 >= 2  # 2 / 4 full grammar families
            if not unseen_topology and not unseen_family:
                name = "fit"
            elif not unseen_topology:
                name = "grammar_guard"
            elif not unseen_family:
                name = "topology_guard"
            else:
                name = "double_guard"
            result[name].append(row)
    validate_splits(result)
    return result


def validate_splits(groups):
    require(set(groups) == set(SPLITS), "split labels changed")
    all_rows = [r for key in SPLITS for r in groups[key]]
    require({k: len(v) for k,v in groups.items()} == EXPECTED
            and len({r.prompt_sha256 for r in all_rows}) == 512,
            "row counts, unique prompts or exact partition changed")
    graph_sets = {}
    hashes = {}
    for key in SPLITS:
        part = groups[key]
        hs = {r.prompt_sha256 for r in part}
        hashes[key] = hs
        graph_sets[key] = {r.topology_key for r in part}
        require(len(hs) == len(part), "duplicate prompt inside " + key)
        require(sum(bool(r.reachable) for r in part) == len(part)//2,
                "label imbalance in " + key)
        require(sum(r.query_style % 2 == 0 for r in part) == len(part)//2,
                "source/destination-first imbalance in " + key)
        families = {r.query_style // 2 for r in part}
        require(families == ({0,1} if key in ("fit", "topology_guard") else {2,3}),
                "grammar-family blocked split drift " + key)
        require(len(part) == 4 * len(graph_sets[key]),
                "requires paired voices per grammar family and graph")
    for i,k1 in enumerate(SPLITS):
        for k2 in SPLITS[i+1:]:
            require(not (hashes[k1] & hashes[k2]), "train/pseudo-holdout prompt leakage")
    require(graph_sets["fit"] == graph_sets["grammar_guard"]
            and graph_sets["topology_guard"] == graph_sets["double_guard"]
            and len(graph_sets["fit"]) == 48
            and len(graph_sets["topology_guard"]) == 16
            and not graph_sets["fit"] & graph_sets["topology_guard"],
            "graph topology fold crossed the guard")
    return True


def sha_json(obj):
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def protocol(groups):
    validate_splits(groups)
    p = {
        "stage": STAGE,
        "original_S3BR_TRAIN512_SHA256": OLD_S3BR_SHA,
        "safe_initialization_only_PRE_S3BR_S3BN_FP16_SHA256": PRE_S3BR_PARENT_SHA,
        "rejected_do_not_initialize_S3BT_FP16_SHA256": REJECTED_S3BT_SHA,
        "all_four_independent_EVAL128_already_CONSUMED_run_ids": list(CONSUMED_EVAL_RUNS),
        "these_512_were_used_by_rejected_S3BT_not_independent_EVAL": True,
        "blocked_grammar_family_index": {"fit": [0,1], "guard": [2,3]},
        "grammar_family_names": list(CONSTRUCTIONS),
        "blocked_graph_rule": "sorted(original S3BR graph_id) index modulo 4 == 0 is guard",
        "subset_counts_expected": EXPECTED,
        "subsets": {
            key: {
                "n": len(groups[key]),
                "unique_graphs": len({r.topology_key for r in groups[key]}),
                "yes": sum(bool(r.reachable) for r in groups[key]),
                "no": sum(not r.reachable for r in groups[key]),
                "source_first": sum(r.query_style % 2 == 0 for r in groups[key]),
                "destination_first": sum(r.query_style % 2 == 1 for r in groups[key]),
                "prompt_hashes_sha256": sha_json([r.prompt_sha256 for r in groups[key]]),
                "graph_ids_sha256": sha_json(sorted({r.graph_id for r in groups[key]})),
            } for key in SPLITS
        },
        "future_training_must_only_load_fit_partition": True,
        "guard_partitions_not_allowed_in_gradient_or_alpha_or_checkpoint_choice": True,
        "diagnostic_guards_are_NOT_blind_generalization_or_promotion_evidence": True,
        "fresh_fifth_external_EVAL_to_be_preregistered_only_after_model_lock": True,
        "no_old_EVAL_content_loaded_retested_or_retuned": True,
        "no_model_weights_CPU_features_training_or_scoring_this_step": True,
        "frozen_original_Calibri_GreenTunnel_NRA_616_J7": True,
        "LIVE_Render_Supabase_APK_main_QA_TEST_untouched": True,
        "candidate_accepted": False, "output_gate": 0.0,
    }
    p["commitment_sha256"] = sha_json(p)
    return p


def main():
    require(len(sys.argv) == 2, "single NEW output directory required")
    output = Path(sys.argv[1])
    require(not output.exists(), "never overwrite a sealed preregistration")
    p = protocol(split_data())
    output.mkdir(parents=True, exist_ok=False)
    (output / "NRA2_S3BX_BLOCKED_TRAIN_ONLY_PROTOCOL.json").write_text(
        json.dumps(p, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"stage": p["stage"], "split_counts": EXPECTED,
                      "commitment_sha256": p["commitment_sha256"],
                      "candidate_accepted": False, "output_gate": 0.0},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
