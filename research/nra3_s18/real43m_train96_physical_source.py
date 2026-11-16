"""NRA3 S18c0: immutable 96 real Calibri43M+GT CPU TRAIN wordpiece features.

Download exclusively S18b physically verified genuine post-GT frozen original
CPU FP16 features. Strict source SHA + original protected 616 J7 + physical
token ID digest + wordpiece maps. TRAIN-internal grouped split definitions
cannot reuse or score S17c/S15c/S12c/EVAL128. Never train/score/evaluate here.
No model weights are written. Output gate stays zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord, CausalAlignment,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    TrainSentence, CONSTRUCTIONS, S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
    build_s18_new_train96, seal_s18_new96_train, direct_predicate_argument_gold,
)

S18B_TRUE_CPU_RUN = 43910479658
S18B_SOURCE_SHA = "abde63c34534d90e00a6ac61808d2453ce0cfc966c83d45377c1fb620999cca9"
EXPECTED_ROWS = 96
EXPECTED_ROLES = {"AGENT": 96, "PATIENT": 96, "RECIPIENT": 96, "SOURCE": 12}


def require(condition, message):
    if not bool(condition):
        raise ValueError("S18c0 original 43M+GT CPU NEW96 TRAIN-only: " + message)


@dataclass(frozen=True)
class Original43MTrain96:
    row: TrainSentence
    hidden320: torch.Tensor
    aligned: CausalAlignment


def checked_real_train96(feature_path: Path, report_path: Path):
    """No scores or gold in neural input: bind native hidden only to TRAIN rows."""
    feature_path, report_path = Path(feature_path), Path(report_path)
    require(feature_path.is_file() and report_path.is_file(),
            "original physical S18b feature+report absent")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    require(S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256 == S18B_SOURCE_SHA,
            "original immutable source SHA drift")
    want = {
        "stage": "NRA3_S18B_TRUE_ORIGINAL43M_GT_LAYER8_CPU_NEW96_TRAIN_ONLY",
        "S18_NEW96_TRAIN_COMPLETE_SOURCE_SHA256": S18B_SOURCE_SHA,
        "S18_SOURCE_ONLY_CI_RUN_ID": 43904457245,
        "S18b0_original_tokenizer_contract_CI_RUN_ID": 43908545721,
        "S18_new96_train_gold_links_only_in_source": 300,
        "S18_research_only_TRAIN_not_independent_EVAL": True,
        "S18_no_old_consumed_S17c_S15c_S12c_or_EVAL128_sources": True,
        "S18_2309FP16_S16b_adapter_not_loaded": True,
        "S194_ORIGINAL_616_J7_BASELINE_PARENT_RUN_ID": 43345267876,
        "all_616_native_CPU_generated_token_digests_unchanged": True,
        "original_456_full_EOS": 456,
        "original_new128_full_EOS": 128,
        "original_new32_full_EOS": 0,
        "actual_43M_CPU_12_transformer_layers": True,
        "observed_layer_index_zero_based": 7,
        "GT_residuals_before_readport_in_same_active_S93_scope": True,
        "NRA3_output_gate": 0,
        "NRA3_candidate_accepted": False,
        "activation_allowed": False,
        "new96_TRAIN_rows": EXPECTED_ROWS,
        "actual_genuine_43M_CPU_new96_TRAIN_prefills": EXPECTED_ROWS,
        "real_tokenizer_chat_offsets_verified": True,
        "true_real_Calibri_token_alignment_all_96": True,
        "S18_no_optimizer_or_model_selection": True,
        "S18_gold_labels_not_passed_to_original43M": True,
        "no_training_original43M_GT_NRA_or_S16b": True,
        "NRA3_grammar_or_semantic_accuracy_tested": False,
        "new_S18_candidate_616_active_J7_EOS_measured": False,
        "full_all_in_GT_NRA_bpp_le07_certified": False,
        "whole_system_CPU_speed_ge90pct_CB59_measured": False,
        "no_Render_Supabase_APK_LIVE_or_Jessica_chat": True,
    }
    require(all(report.get(k) == v for k, v in want.items()),
            "original physical source/616 J7 or no-activation contract failed")
    require(report.get("actual_fp16_feature_file") == feature_path.name
            and report.get("actual_fp16_feature_sha256") == digest,
            "true physical FP16 feature SHA/report mismatch")
    require(report.get("S194_ORIGINAL_CHECKPOINT_SHA256")
            == "b1e58fde43bc9d4593f6db78f1c9b55d556255f4dcac7c96bce09302b68b34c2",
            "original S194 checkpoint identity changed")
    tensors = load_file(str(feature_path), device="cpu")
    require(set(tensors) == {"token_hidden320", "input_ids", "token_mask", "token_lengths"},
            "unknown data/labels embedded in physical neural features")
    h, ids, mask, lens = (tensors[k] for k in (
        "token_hidden320", "input_ids", "token_mask", "token_lengths"))
    require(
        h.ndim == 3 and h.shape[0] == EXPECTED_ROWS and h.shape[-1] == 320
        and h.dtype == torch.float16 and ids.dtype == torch.int32
        and mask.dtype == torch.bool and h.shape[:2] == ids.shape == mask.shape
        and lens.shape == (EXPECTED_ROWS,) and len(h.shape) == 3
        and bool(torch.isfinite(h).all()) and
        list(h.shape) == report.get("actual_hidden320_shape") and
        lens.tolist() == report.get("true_train96_token_counts"),
        "actual original 43M source hidden320 shape/dtype/length/padding mismatch")
    rows = build_s18_new_train96()
    require(len(rows) == 96 and seal_s18_new96_train()[
        "S18_complete_original_96_new_TRAIN_full_annotation_SHA256"
    ] == S18B_SOURCE_SHA and len(report.get("wordpiece_alignment", [])) == 96,
            "immutable S18 new96 source or wordpiece map count changed")
    positives = [link[2] for row in rows for link in direct_predicate_argument_gold(row)]
    require(len(positives) == 300 and {
        k: positives.count(k) for k in EXPECTED_ROLES
    } == EXPECTED_ROLES, "original immutable TRAIN96 directed gold role counts changed")
    original_ids = []
    samples = []
    for i, row in enumerate(rows):
        n = int(lens[i])
        require(1 < n <= ids.shape[1]
                and bool(torch.equal(mask[i], torch.arange(ids.shape[1]) < n))
                and bool((ids[i, n:] == -1).all())
                and bool((h[i, n:] == 0).all())
                and bool((ids[i, :n] >= 0).all()),
                "actual original tokenizer IDs or wordpiece padding corrupted")
        tokens = tuple(int(x) for x in ids[i, :n].tolist())
        original_ids.append(list(tokens))
        m = report["wordpiece_alignment"][i]
        require(m.get("text_sha256") == row.text_sha256()
                and m.get("token_count") == n
                and len(m.get("word_last_token_positions", [])) == len(row.words)
                and len(m.get("word_token_positions", [])) == len(row.words),
                "original 43M wordpiece-to-gold TRAIN alignment/source identity drift")
        aligned_words, used = [], set()
        for j, (last, positions) in enumerate(zip(
                m["word_last_token_positions"], m["word_token_positions"]), 1):
            require(isinstance(positions, list) and bool(positions)
                    and type(last) is int and last == max(positions)
                    and all(type(p) is int and 0 <= p < n and p not in used
                            for p in positions),
                    "original wordpiece crosses words or has invalid token offsets")
            used.update(positions)
            aligned_words.append(AlignedWord(j, tuple(positions), last))
        require(all(aligned_words[j].last_token < aligned_words[j + 1].last_token
                    for j in range(len(aligned_words) - 1)),
                "original token order not strictly increasing")
        aligned = CausalAlignment(
            row.text_sha256(), n, tokens, tuple(aligned_words), n - 1, False)
        samples.append(Original43MTrain96(
            row, h[i, :n].float().contiguous(), aligned))
    actual_token_sha = hashlib.sha256(json.dumps(
        original_ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(actual_token_sha == report.get("actual_token_ids_lengths_sha256"),
            "real Calibri token ID payload SHA differs from physical S18b")
    return tuple(samples), report


def grouped_train96_split_contract(rows=None):
    """TRAIN-internal only, fully disjoint heldout lexeme families and syntax."""
    rows = build_s18_new_train96() if rows is None else tuple(rows)
    canonical = build_s18_new_train96()
    require(len(rows) == len(canonical) == 96
            and all(a == b for a, b in zip(rows, canonical)),
            "grouped CV requires exactly original immutable S18 TRAIN96")
    require(len(CONSTRUCTIONS) == 8, "unexpected syntactic groups")
    folds = []
    for k in range(4):
        held_groups = tuple(i for i in range(12) if i % 4 == k)
        train = tuple(i for i, r in enumerate(rows)
                      if r.lexical_family not in held_groups)
        held = tuple(i for i, r in enumerate(rows)
                     if r.lexical_family in held_groups)
        require(len(train) == 72 and len(held) == 24
                and {rows[i].lexical_family for i in train}.isdisjoint(
                    {rows[i].lexical_family for i in held}),
                "lexical family heldout leak")
        folds.append(("lexical", f"lex_{k}", held_groups, train, held))
    for k, construction in enumerate(CONSTRUCTIONS):
        train = tuple(i for i, r in enumerate(rows)
                      if r.construction != construction)
        held = tuple(i for i, r in enumerate(rows)
                     if r.construction == construction)
        require(len(train) == 84 and len(held) == 12
                and all(rows[i].construction != construction for i in train)
                and all(rows[i].construction == construction for i in held),
                "syntactic construction heldout leak")
        folds.append(("syntax", f"syntax_{k}", (construction,), train, held))
    require(len(folds) == 12 and
            sorted(i for f in folds[:4] for i in f[4]) == list(range(96)) and
            sorted(i for f in folds[4:] for i in f[4]) == list(range(96)) and
            all(set(f[3]).isdisjoint(f[4]) and
                sorted((*f[3], *f[4])) == list(range(96)) for f in folds),
            "unbalanced TRAIN internal group CV or duplicate row")
    return tuple(folds)
