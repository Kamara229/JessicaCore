"""NRA2 S3CG: SIXTH untouched independent128 graph+grammar preregistration.

S3CF safe FP16 was physically SHA-locked by successful genuine CPU TRAIN2240
#43678041407 before this data-only preregistration. The five preceding EVAL128
sets were consumed and are permanently forbidden for scoring or training.
This script only computes prior graph HASH EXCLUSIONS and commits new EVAL data.
Do not capture Calibri CPU tokens, train, score, tune, select or deploy here.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys

from research.nra2_bridge.stage3cb_fifth_future_unseen128_prereg import (
    ancestral_exclusions as older_graph_rows,
    build_rows as fifth_spent_rows, commitment as fifth_commit,
    STYLES as FIFTH_STYLES,
)
from research.nra2_bridge.stage3bo_third_future_independent128_prereg import STYLES as THIRD_STYLES
from research.nra2_bridge.stage3bu_fourth_future_unseen128_prereg import STYLES as FOURTH_STYLES
from research.nra2_bridge.stage3br_relation_language_train512_prereg import STYLES as NEW_TRAIN_STYLES
from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import STYLES as EARLY_STYLES
from research.nra2_bridge.stage3bj_future_independent128_prereg import STYLES as SECOND_STYLES
from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    GraphPrompt, FAMILIES, HEADERS, make_graph, graph_has_cycle,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle

SPLIT = "S3CG_SIXTH_INDEPENDENT_UNTOUCHED_ROLE_EVAL128"
SEED = 20261115127
DESTINATION_FIRST = (0, 2, 4, 6)
SOURCE_FIRST = (1, 3, 5, 7)
LENGTHS = (5, 6, 5, 6)
S3CB_FIFTH_SHA = "a23dd34eac45dfa17d00798c2e3837c6c837c490a9ba50e809d73d3c3374cd5b"
S3CF_SAFE_FP16_SHA = "3c0cb51ea9cdffba89666c043223044805e6d52397daa0ae6ca473f845efc312"
S3CF_TRAIN_SUCCESS_RUN = 43678041407
S3CD_FIFTH_CONSUMED_RUN = 43677179638
S3CE_TRAIN_OBJECTIVE_SUCCESS_RUN = 43677718791
STYLES = (
    "For destination {dst}, does the set of its directed predecessors contain origin {src}?",
    "Suppose movement begins at {src} and follows arrow direction: can it eventually stop at {dst}?",
    "Regarding terminal node {dst}, is an incoming directed walk possible when it begins at {src}?",
    "If a trace launches from {src}, could it ever visit {dst} without reversing any arrow?",
    "For endpoint {dst}, can a chain of permitted forward transitions originate at {src}?",
    "Can exploration of outgoing successors from {src} eventually discover node {dst}?",
    "Among all directed walks terminating at {dst}, is there one that begins at {src}?",
    "Is there a valid route starting at {src} and ending at {dst} when edges are forward-only?",
)


def require(ok, msg):
    if not bool(ok):
        raise AssertionError("S3CG SIXTH EVAL hash-only prereg fail-closed: " + msg)


@lru_cache(maxsize=1)
def ancestral_exclusions():
    # Previous EVAL generators are used ONLY to establish exact exclusion hashes.
    old = older_graph_rows()
    fifth = fifth_spent_rows()
    require(fifth_commit(fifth)["commitment_sha256"] == S3CB_FIFTH_SHA,
            "consumed fifth prereg SHA changed")
    return old + fifth


def build_rows():
    previous = ancestral_exclusions()
    used = {tuple(sorted(r.edges)) for r in previous}
    old_sha = {r.prompt_sha256 for r in previous}
    prior = (set(FIFTH_STYLES) | set(FOURTH_STYLES) | set(THIRD_STYLES) |
             set(SECOND_STYLES) | set(NEW_TRAIN_STYLES) |
             {s for x in EARLY_STYLES.values() for s in x})
    require(len(STYLES) == 8 and len(set(STYLES)) == 8 and
            not (set(STYLES) & prior),
            "SIXTH new grammar must not equal any of five preceding EVAL/known TRAIN styles")
    rng = random.Random(SEED)
    rows = []
    for family in FAMILIES:
        for j in range(4):
            n = LENGTHS[j]
            for _ in range(30000):
                order = tuple(rng.sample(range(6), n))
                edges = make_graph(order, family)
                key = tuple(sorted(edges))
                forward = [(a, b) for a in order for b in order
                           if a != b and oracle(edges, a, b) and not oracle(edges, b, a)]
                if key not in used and forward:
                    used.add(key)
                    break
            else:
                raise AssertionError("unable to find SIXTH independent new topology")
            src0, dst0 = rng.choice(forward)
            for style, template in enumerate(STYLES):
                yes = (style % 2) == (j % 2)
                src, dst = (src0, dst0) if yes else (dst0, src0)
                require(oracle(edges, src, dst) == yes, "SIXTH directional graph truth")
                edge_text = [f"R{a} -> R{b}" for a, b in edges]
                rng.shuffle(edge_text)
                q = HEADERS[(j // 2 + style) % len(HEADERS)].format(
                    edges="; ".join(edge_text),
                    query=template.format(src=f"R{src}", dst=f"R{dst}"),
                )
                row = GraphPrompt(
                    graph_id=f"{SPLIT}-{family}-{j:02}",
                    split=SPLIT, family=family, active_nodes=n,
                    edges=edges, src=src, dst=dst,
                    reachable=yes, query_style=style, question=q,
                )
                require(row.prompt_sha256 not in old_sha, "SIXTH question overlaps spent EVAL")
                rows.append(row)
    validate(rows)
    return rows


def validate(rows):
    require(len(rows) == 128 and len({r.prompt_sha256 for r in rows}) == 128,
            "sixth count, uniqueness")
    past = ancestral_exclusions()
    used = {tuple(sorted(r.edges)) for r in past}
    old_sha = {r.prompt_sha256 for r in past}
    require(not ({r.topology_key for r in rows} & used)
            and not ({r.prompt_sha256 for r in rows} & old_sha),
            "sixth graph or prompt overlaps 5 prior independent EVALs, old TRAIN or DEV")
    groups = defaultdict(list)
    for r in rows:
        require(r.split == SPLIT and r.family in FAMILIES
                and oracle(r.edges, r.src, r.dst) == r.reachable
                and graph_has_cycle(r.edges) == (r.family == "cycle_sink"),
                "sixth graph family or directional oracle truth")
        observed = query_char_offsets(r.question)["text_order_nodes"]
        require(observed == ([r.dst, r.src] if r.query_style in DESTINATION_FIRST
                             else [r.src, r.dst]),
                "sixth literal token role mention order mismatch")
        groups[r.graph_id].append(r)
    require(len(groups) == 16 and len({r.topology_key for r in rows}) == 16,
            "sixth must hold 16 novel topologies")
    for versions in groups.values():
        versions = sorted(versions, key=lambda r: r.query_style)
        require([r.query_style for r in versions] == list(range(8))
                and sum(r.reachable for r in versions) == 4
                and len({(r.src,r.dst) for r in versions}) == 2,
                "sixth grammar, class or semantic direction imbalance")
        for k in (0,2,4,6):
            require((versions[k].src, versions[k].dst) ==
                    (versions[k+1].dst, versions[k+1].src),
                    "sixth directional reversed role quartet missing")
    require(sum(r.reachable for r in rows) == 64
            and sum(r.query_style in DESTINATION_FIRST for r in rows) == 64,
            "sixth total class or text order skew")
    for family in FAMILIES:
        group = [r for r in rows if r.family == family]
        require(len(group) == 32 and sum(r.reachable for r in group) == 16,
                "sixth family imbalance")
    for style in range(8):
        group = [r for r in rows if r.query_style == style]
        require(len(group) == 16 and sum(r.reachable for r in group) == 8,
                "sixth grammar style must not predict answer")
    return True


def commitment(rows):
    validate(rows)
    obj = {
        "stage": "NRA2_S3CG_SIXTH_UNTOUCHED_EVAL128_PREREG_NO_CPU_OR_MODEL",
        "split": SPLIT, "seed": SEED,
        "S3CB_prior_fifth_consumed_commitment_sha256": S3CB_FIFTH_SHA,
        "S3CD_prior_fifth_consumed_run": S3CD_FIFTH_CONSUMED_RUN,
        "S3CE_prior_TRAIN_only_96_role_order_pairs_CI": S3CE_TRAIN_OBJECTIVE_SUCCESS_RUN,
        "S3CF_safe_frozen_before_sixth_EVAL_sha256": S3CF_SAFE_FP16_SHA,
        "S3CF_successful_TRAIN_only_run": S3CF_TRAIN_SUCCESS_RUN,
        "prior_first_independent_EVAL128_consumed_in": 43640363396,
        "prior_second_independent_EVAL128_consumed_in": 43644153544,
        "prior_third_independent_EVAL128_consumed_in": 43648038071,
        "prior_fourth_independent_EVAL128_consumed_in": 43651233135,
        "prior_fifth_independent_EVAL128_consumed_in": S3CD_FIFTH_CONSUMED_RUN,
        "all_five_previous_independent_EVAL128_permanently_consumed": True,
        "S3CF_candidate_physically_SHA_locked_before_sixth_EVAL_prereg": True,
        "sixth_independent_graphs": 16,
        "sixth_independent_new_grammar": 8,
        "future_EVAL128_count": 128,
        "yes_count": 64, "no_count": 64,
        "source_first_count": 64, "destination_first_count": 64,
        "styles": list(STYLES),
        "sha_prompts_in_order": [r.prompt_sha256 for r in rows],
        "sha_graphs_in_order": [hashlib.sha256(
            json.dumps(sorted(r.edges)).encode()).hexdigest()
            for r in rows[::8]],
        "sha_yes_no": hashlib.sha256(
            bytes(int(r.reachable) for r in rows)).hexdigest(),
        "all_prior_EVALs_excluded_from_optimizer_and_candidate_selection": True,
        "no_model_weights_or_Calibri_CPU_or_scores_or_training_this_stage": True,
        "original_616_S194_J7_preserved_by_S3CF": True,
        "no_regular_Jessica_chat_tests": True,
        "candidate_accepted": False, "output_gate": 0.0,
        "no_LIVE_Render_Supabase_APK_main_QA_TEST_changes": True,
    }
    obj["commitment_sha256"] = hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return obj


def main():
    require(len(sys.argv) == 2, "usage S3CG NEW_EMPTY_DIR")
    out = Path(sys.argv[1])
    require(not out.exists(), "sixth independent EVAL prereg must not overwrite")
    obj = commitment(build_rows())
    out.mkdir(parents=True, exist_ok=False)
    (out / "NRA2_S3CG_SIXTH_UNTOUCHED128_PREREG.json").write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k:v for k,v in obj.items()
                      if k not in ("styles","sha_prompts_in_order","sha_graphs_in_order")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
