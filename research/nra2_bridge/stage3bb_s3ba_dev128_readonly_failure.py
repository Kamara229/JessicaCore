"""S3BB: read-only diagnostic of already-scored S3AQ DEV128, S3AW vs S3BA.

No optimizer or training. Never prefill, infer or score future EVAL128.
Source is immutable S3AR true 43M CPU with original 616 J7 protected.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows, commitment, DEV, STYLES,
)
from research.nra2_bridge.stage3as_s3z_init_train_dev_roles import (
    S3AR_SHA, S3AQ_COMMIT_SHA, bind, gold_roles, require,
)
from research.nra2_bridge.stage3ax_s3aw_dev128_readonly_failure import (
    role_details, group_stats,
)
from research.nra2_bridge.stage3k_train_real_entity_binding import (
    exact_original_tokenizer, sha_file,
)

S3AW_RUN = 43583928298
S3BA_RUN = 43590032315
S3AZ_RUN = 43587917560
S3AW_SHA = "8541eea24d5074e29b6a4acd8b611d2d29d608dacf2dfe909e8df05eb0fb2130"
S3BA_SHA = "4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
S3BA_AUDIT = "NRA2_S3BA_S3AW_INIT_TRAIN1536_DEV128_ROLE_AUDIT.json"

def main():
    if len(sys.argv) != 5:
        raise SystemExit("S3BB S3AR_CPU384_DIR S3AW_LOCKED_DIR S3BA_LOCKED_DIR EMPTY_OUT")
    ar_dir, aw_dir, ba_dir, out = map(Path, sys.argv[1:])
    require(not out.exists() or not any(out.iterdir()), "S3BB archive cannot overwrite")
    ar = json.loads((ar_dir / "NRA2_S3AR_TRAIN256_DEV128_REAL_CPU_J7_AUDIT.json").read_text())
    ba = json.loads((ba_dir / S3BA_AUDIT).read_text())
    ar_t = ar_dir / "NRA2_S3AR_REAL_CPU_TRAIN256_DEV128_ALL_TOKENS_FP16.safetensors"
    ar_y = ar_dir / "NRA2_S3AR_TRAIN256_DEV128_STRUCTURAL_LABELS.safetensors"
    aw_w = aw_dir / "NRA2_S3AW_S3AS_INIT_ROLE1024_FP16.safetensors"
    ba_w = ba_dir / "NRA2_S3BA_S3AW_INIT_ROLE1536_FP16.safetensors"

    require(ar["S3AQ_registered_commitment_sha256"] == S3AQ_COMMIT_SHA
            and ar["all_616_token_digest_matches_S194"] is True
            and ar["S3AQ_future_EVAL128_never_prefilled_scored_or_trained"] is True
            and ar["full_token_fp16_sha256"] == S3AR_SHA == sha_file(ar_t)
            and ar["physical_graph_labels_sha256"] == sha_file(ar_y)
            and ar["graph_prefill_count"] == 384
            and ar["train_count"] == 256 and ar["dev_count"] == 128,
            "S3AR original 43M real CPU/616 J7 or split drift")
    require(ba["stage"] == "NRA2_S3BA_S3AW_INIT_ROLE1536_TRAIN_S3AY512_DEV128_EXPLORATORY"
            and ba["S3AW_locked_parent_run"] == S3AW_RUN
            and ba["S3AZ_native_CPU_run"] == S3AZ_RUN
            and ba["S3AW_parent_fp16_sha256"] == S3AW_SHA == sha_file(aw_w)
            and ba["new_S3BA_physical_fp16_sha256"] == S3BA_SHA == sha_file(ba_w)
            and ba["train_count"] == 1536 and ba["DEV_count"] == 128
            and ba["DEV128_scored_only_after_FP16_SHA_commit"] is True
            and ba["DEV128_excluded_from_gradient_optimizer"] is True
            and ba["DEV128_is_already_scored_exploratory_not_independent"] is True
            and ba["S3AQ_future_EVAL128_not_prefilled_scored_or_trained"] is True
            and ba["S3AO_already_scored_EVAL128_NOT_reused_for_training_or_selection"] is True
            and ba["full_S194_original_616_J7_digests_from_S3AV_protected"] is True
            and ba["original_graph_physical_fp16_bitwise_equal"] is True
            and ba["physical_fp16_reload_equal"] is True
            and ba["output_gate"] == 0.0 and ba["candidate_accepted"] is False,
            "S3AW/S3BA FP16, native 616 J7, future EVAL or fixed DEV protocol drift")

    rows = build_rows()
    require(commitment(rows)["commitment_sha256"] == S3AQ_COMMIT_SHA,
            "preregistered parent dataset drift")
    dev = [r for r in rows if r.split == DEV]
    require(len(dev) == 128 and len({r.graph_id for r in dev}) == 32
            and [r.prompt_sha256 for r in dev]
            == ar["split_groups"][DEV]["prompt_sha256"],
            "exact already-scored DEV128 prompts and ordering drift")
    toks, labels = load_file(str(ar_t), device="cpu"), load_file(str(ar_y), device="cpu")
    dt = toks["token_hidden320"][256:384]
    dm = toks["token_mask"][256:384]
    dl = toks["token_lengths"][256:384]
    src, dst = gold_roles(labels, 256, 384)
    query, _ = bind(exact_original_tokenizer(), [r.question for r in dev], dt, dm, dl)
    require(bool(torch.equal(query.node_order.sort(1).values,
                            torch.stack([src, dst], dim=-1).sort(1).values)),
            "S3BB DEV128 true role alignment drift")
    previous = role_details(aw_w, dt, dm, query, src, dst, dev)
    candidate = role_details(ba_w, dt, dm, query, src, dst, dev)
    p = sum(x["correct"] for x in previous)
    q = sum(x["correct"] for x in candidate)
    require(p == 89 and q == 94
            and p == ba["DEV128_paired_S3AW_baseline"]["all"]["both_correct"]
            and q == ba["DEV128_paired_S3BA_candidate"]["all"]["both_correct"],
            "S3BA scores not independently reproduced from physical FP16")
    flips = {"prior_correct_candidate_wrong": 0,
             "prior_wrong_candidate_correct": 0,
             "both_correct": 0,
             "both_wrong": 0}
    records = []
    for old, new in zip(previous, candidate):
        require(old["graph_id"] == new["graph_id"]
                and old["style"] == new["style"], "paired exact DEV index drift")
        flag = ("both_correct" if old["correct"] and new["correct"] else
                "both_wrong" if not old["correct"] and not new["correct"] else
                "prior_correct_candidate_wrong" if old["correct"] else
                "prior_wrong_candidate_correct")
        flips[flag] += 1
        records.append({"style": new["style"], "family": new["family"],
                        "graph_id": new["graph_id"],
                        "old_correct": old["correct"], "new_correct": new["correct"],
                        "flip": flag, "destination_first": new["destination_first"],
                        "positive": new["positive"]})
    require(sum(flips.values()) == 128
            and flips["prior_wrong_candidate_correct"]
            - flips["prior_correct_candidate_wrong"] == 5,
            "paired DEV score changes do not reconcile with 89→94")
    by_graph = defaultdict(list)
    for r in candidate:
        by_graph[r["graph_id"]].append(r)
    quartet = {str(i): sum(sum(v["correct"] for v in group) == i
                           for group in by_graph.values()) for i in range(5)}
    report = {
        "stage": "NRA2_S3BB_S3BA_POSTSCORE_DEV128_ROLE_ERROR_DIAG_READONLY",
        "S3AW_successful_run": S3AW_RUN,
        "S3BA_successful_run": S3BA_RUN,
        "S3AQ_preregistered_commitment_sha256": S3AQ_COMMIT_SHA,
        "S3AW_physical_fp16_sha256": S3AW_SHA,
        "S3BA_physical_fp16_sha256": S3BA_SHA,
        "DEV128_prior_correct": p, "DEV128_candidate_correct": q,
        "DEV128_paired_flips": flips,
        "DEV128_S3AW_by_style": group_stats(previous, "style"),
        "DEV128_S3BA_by_style": group_stats(candidate, "style"),
        "DEV128_S3BA_by_family": group_stats(candidate, "family"),
        "DEV128_S3BA_by_truth": group_stats(candidate, "positive"),
        "DEV128_S3BA_correct_per_graph_quartet": quartet,
        "DEV128_style_templates": list(STYLES[DEV]),
        "DEV128_instance_flips_without_prompt_text": records,
        "DEV128_already_scored_exploratory_not_independent": True,
        "DEV128_not_used_for_optimizer_or_candidate_selection": True,
        "S3AR_original_CPU_frozen_616_J7": True,
        "S3AQ_future_EVAL128_not_prefilled_scored_or_trained": True,
        "output_gate": 0.0, "candidate_accepted": False,
        "new_weights_or_optimizer_steps": 0,
        "LIVE_Render_Supabase_APK_main_QA_TEST_unchanged": True,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "NRA2_S3BB_S3BA_DEV128_ROLE_ERROR_DIAGNOSTIC.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "stage", "DEV128_prior_correct", "DEV128_candidate_correct",
        "DEV128_paired_flips", "DEV128_S3AW_by_style",
        "DEV128_S3BA_by_style", "DEV128_S3BA_correct_per_graph_quartet")},
        sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
