"""S3BC: NEW contrastive query-role TRAIN512 precommit, data only, no model CPU.

Only 64 S3AQ TRAIN graph topologies; 8 novel styles, balanced truth, direction,
and literal mention order. Never expose previously scored DEV or future EVAL
prompts to training, scoring, tokenizer or original CPU hidden-state capture.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows as s3aq_rows, commitment as s3aq_commitment, TRAIN, DEV, EVAL,
    STYLES as S3AQ_STYLES,
)
from research.nra2_bridge.stage3au_train512_paraphrase_prereg import (
    build_rows as s3au_rows, STYLES as S3AU_STYLES,
)
from research.nra2_bridge.stage3ay_structural_train512_prereg import (
    build_rows as s3ay_rows, STYLES as S3AY_STYLES,
)
from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    GraphPrompt, QUERY_STYLES as S3M_STYLES, HEADERS,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import (
    query_char_offsets, QUERY_STYLES as S3T_STYLES,
)
from research.nra2_bridge.stage3x_reversal_train_prereg import NEW_STYLES as S3X_STYLES
from research.nra2_bridge.stage3aa_independent_newgrammar128_prereg import QUERY_STYLES as S3AA_STYLES
from research.nra2_bridge.stage3ae_train256_future_eval128_prereg import STYLES as S3AE_STYLES
from research.nra2_bridge.stage3ak_newtrain256_futureeval128_prereg import STYLES as S3AK_STYLES
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle

SPLIT = "S3BC_NEW_ROLE_CONTRASTIVE_TRAIN512"
S3AQ_SHA = "c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839"
S3BA_FP16_SHA = "4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
S3BA_SUCCESS_RUN = 43590032315
S3BB_READONLY_SUCCESS_RUN = 43590817240
DESTINATION_FIRST = (1, 3, 5, 7)
SOURCE_FIRST = (0, 2, 4, 6)
# Paired role-order variants: marker imperative; origin-subject; message-to-goal;
# conditional directional traversal. The query has exactly two literal mentions.
STYLES = (
    "Set departure marker {src} and arrival marker {dst}; does following the directed edges connect the markers in that order?",
    "Set arrival marker {dst} and departure marker {src}; does following the directed edges connect the markers in that order?",
    "A search launches at {src} and follows outgoing arrows only; will its search frontier ever contain {dst}?",
    "Would {dst} ever enter the frontier of a search launched at {src} that follows outgoing arrows only?",
    "Can {src} pass a token through successive forward arcs into {dst}?",
    "Can {dst} receive a token passed through successive forward arcs from {src}?",
    "If we label {src} as the beginning, can we follow forward arcs until we encounter {dst}?",
    "If we label {dst} as the endpoint, can we start from {src} and follow forward arcs to reach it?",
)
CONSTRUCTIONS = ("marker_imperative", "outgoing_search", "token_transfer", "conditional_direction")


def _all_prior_styles():
    return (
        set(S3M_STYLES) | set(S3T_STYLES) | set(S3X_STYLES)
        | set(S3AA_STYLES) | set(S3AU_STYLES) | set(S3AY_STYLES)
        | {t for v in S3AQ_STYLES.values() for t in v}
        | {t for v in S3AE_STYLES.values() for t in v}
        | {t for v in S3AK_STYLES.values() for t in v}
    )


def build_rows():
    parent = s3aq_rows()
    if s3aq_commitment(parent)["commitment_sha256"] != S3AQ_SHA:
        raise AssertionError("S3BC S3AQ sealed protocol drift")
    if len(STYLES) != 8 or len(set(STYLES)) != 8 or set(STYLES) & _all_prior_styles():
        raise AssertionError("S3BC query templates must be eight novel independent phrasings")
    old_train = [r for r in parent if r.split == TRAIN]
    # Heldout material is used strictly as hashed exclusion metadata,
    # never for model input, prompting, scoring, graph selection or fitting.
    forbidden_sha = {r.prompt_sha256 for r in parent}
    forbidden_sha.update(r.prompt_sha256 for r in s3au_rows())
    forbidden_sha.update(r.prompt_sha256 for r in s3ay_rows())
    by_graph = defaultdict(list)
    for r in old_train:
        by_graph[r.graph_id].append(r)
    if len(by_graph) != 64 or any(len(v) != 4 for v in by_graph.values()):
        raise AssertionError("S3BC requires 64 source TRAIN topologies x four old queries")
    out = []
    for graph_id, group in by_graph.items():
        group.sort(key=lambda r: r.query_style)
        if [r.query_style for r in group] != [0, 1, 2, 3]:
            raise AssertionError("S3BC parent query style order drift")
        if (group[0].src, group[0].dst) != (group[1].dst, group[1].src):
            raise AssertionError("S3BC parent opposite query roles drift")
        for style, template in enumerate(STYLES):
            source = group[0] if style % 2 == 0 else group[1]
            earlier = group[style % 4]
            s, e = query_char_offsets(earlier.question)["clause"]
            question = (earlier.question[:s] +
                        template.format(src=f"R{source.src}", dst=f"R{source.dst}") +
                        earlier.question[e:])
            row = GraphPrompt(
                graph_id=f"{SPLIT}-{graph_id}",
                split=SPLIT, family=earlier.family, active_nodes=earlier.active_nodes,
                edges=earlier.edges, src=source.src, dst=source.dst,
                reachable=oracle(earlier.edges, source.src, source.dst),
                query_style=style, question=question,
            )
            if row.prompt_sha256 in forbidden_sha:
                raise AssertionError("S3BC cannot reuse any original TRAIN/DEV/EVAL prompt hash")
            out.append(row)
    validate(out)
    return out


def validate(rows):
    if len(rows) != 512 or len({r.prompt_sha256 for r in rows}) != 512:
        raise AssertionError("S3BC must have precisely 512 unique TRAIN-only prompts")
    all_graphs = defaultdict(list)
    for r in rows:
        if r.split != SPLIT or not 0 <= r.query_style < 8:
            raise AssertionError("S3BC unauthorized split or template")
        order = query_char_offsets(r.question)["text_order_nodes"]
        expected = [r.dst, r.src] if r.query_style in DESTINATION_FIRST else [r.src, r.dst]
        if order != expected or r.reachable != oracle(r.edges, r.src, r.dst):
            raise AssertionError("S3BC literal roles or forward-edge truth drift")
        all_graphs[r.graph_id].append(r)
    if len(all_graphs) != 64 or len({r.topology_key for r in rows}) != 64:
        raise AssertionError("S3BC topology count drift")
    for graph in all_graphs.values():
        graph.sort(key=lambda r: r.query_style)
        if [r.query_style for r in graph] != list(range(8)):
            raise AssertionError("S3BC graph missing paired construction")
        if sum(r.reachable for r in graph) != 4:
            raise AssertionError("S3BC graph must be four yes / four no")
        if len({r.topology_key for r in graph}) != 1:
            raise AssertionError("S3BC graph edge reuse drift")
        if any((graph[i].src, graph[i].dst) !=
               (graph[i+2].src, graph[i+2].dst) for i in range(6)):
            raise AssertionError("S3BC paired orientation consistency drift")
        pairs = {(r.src, r.dst) for r in graph}
        if len(pairs) != 2 or any((b, a) not in pairs for a, b in pairs):
            raise AssertionError("S3BC every query must have opposite direction")
    for style in range(8):
        subset = [r for r in rows if r.query_style == style]
        if len(subset) != 64 or sum(r.reachable for r in subset) != 32:
            raise AssertionError("S3BC per-grammar yes/no leakage")
    if sum(r.reachable for r in rows) != 256:
        raise AssertionError("S3BC TRAIN512 class imbalance")
    return True


def commitment(rows):
    validate(rows)
    prior = s3aq_rows()
    if s3aq_commitment(prior)["commitment_sha256"] != S3AQ_SHA:
        raise AssertionError("S3BC original split hash mismatch")
    tr = [r for r in prior if r.split == TRAIN]
    held = [r for r in prior if r.split in (DEV, EVAL)]
    old = s3au_rows() + s3ay_rows()
    if ({r.topology_key for r in rows} != {r.topology_key for r in tr}
        or {r.topology_key for r in rows} & {r.topology_key for r in held}
        or {r.prompt_sha256 for r in rows} & {r.prompt_sha256 for r in held + tr + old}):
        raise AssertionError("S3BC TRAIN-only topology or old prompt leakage")
    stats = {str(i): {
        "count": sum(r.query_style == i for r in rows),
        "yes": sum(r.reachable and r.query_style == i for r in rows),
        "destination_first": i in DESTINATION_FIRST,
        "construction": CONSTRUCTIONS[i//2],
    } for i in range(8)}
    data = {
        "stage": "NRA2_S3BC_CONTRASTIVE_ROLE_TRAIN512_PREREG_ONLY_NO_MODEL",
        "S3AQ_parent_commitment_sha256": S3AQ_SHA,
        "S3BA_pinned_role_fp16_sha256": S3BA_FP16_SHA,
        "S3BA_successful_CI": S3BA_SUCCESS_RUN,
        "S3BB_readonly_successful_CI": S3BB_READONLY_SUCCESS_RUN,
        "TRAIN512_count": 512, "TRAIN_graph_count": 64,
        "source_first_count": 256, "destination_first_count": 256,
        "yes_count": 256, "no_count": 256,
        "old_DEV128_already_scored_and_excluded_from_training": True,
        "future_EVAL128_not_prefilled_scored_or_trained": True,
        "original_S194_616_J7_digests_required_before_any_future_training": True,
        "S3AQ_TRAIN_only_graph_topologies_no_DEV_or_EVAL": True,
        "new_styles_distinct_from_prior_styles": True,
        "new_prompts_sha_disjoint_from_all_S3AQ_and_S3AU_S3AY": True,
        "new_8_query_styles": list(STYLES),
        "per_style_balanced": stats,
        "prompt_sha256_in_order": [r.prompt_sha256 for r in rows],
        "graph_topology_sha256_in_order": [
            hashlib.sha256(json.dumps(sorted(r.edges)).encode()).hexdigest()
            for r in rows
        ],
        "oracle_yes_no_in_order": [bool(r.reachable) for r in rows],
        "no_new_CPU_prefill_or_model_weights_or_training_or_scores": True,
        "output_gate": 0.0, "candidate_accepted": False,
        "LIVE_Render_Supabase_APK_main_QA_TEST_unchanged": True,
    }
    data["commitment_sha256"] = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return data


def main():
    if len(sys.argv) != 2:
        raise SystemExit("S3BC requires an empty research-only output directory")
    out = Path(sys.argv[1])
    if out.exists():
        raise FileExistsError("S3BC immutable preregistration refuses existing path")
    report = commitment(build_rows())
    out.mkdir(parents=True, exist_ok=False)
    (out / "NRA2_S3BC_CONTRASTIVE_ROLE_TRAIN512_PREREG.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("prompt_sha256_in_order", "graph_topology_sha256_in_order",
                                   "oracle_yes_no_in_order")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
