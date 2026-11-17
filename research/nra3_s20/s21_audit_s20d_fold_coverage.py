"""NRA3 S21: read-only audit of S20d's actual completed TRAIN-only grouped-fold report.

No model forward, new optimizer, regenerated performance, independent EVAL,
or decoder integration. Gold is inspected ONLY to count TRAIN fold support;
never passed to a model. This identifies structurally unlearnable held-out
relation types and missing per-class rejection metrics before S22 collection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from research.nra3_s19.s19f_real43m_train9_graph_targets import EVENT_RELATION
from research.nra3_s20.s20_new96_synthetic_graph_train_source import (
    PATTERNS, FROZEN_S20_TRAIN96_SHA256, build_s20_synthetic_train96,
    train_only_manifest, train_only_split_contract,
)

PINNED_S20D_FP16_SHA256 = "1c2564d488722228b15cd6db0c1486eb5f1ae4678b6826275c880e765a4f467d"
PINNED_S20D_CI = 44180643254
REAL43M_PROTECTED_CAPTURE_CI = 44179026651


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError("NRA3 S21 fail-closed: " + message)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def micro_f1(counts: dict) -> float:
    tp, fp, fn = (int(counts[k]) for k in ("TP", "FP", "FN"))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def gold_support(rows: tuple, ids: tuple) -> tuple[dict, int]:
    relation = Counter(link.relation for i in ids for link in rows[i].event_relations)
    rejected = sum(a.polarity == "REJECTED" for i in ids for a in rows[i].arguments)
    return {name: relation[name] for name in EVENT_RELATION if name != "NONE"}, rejected


def audit(report: dict, model_sha: str) -> dict:
    require(report.get("stage") ==
            "NRA3_S20D_REAL43M_GT_NEW96_SYNTHETIC_TRAIN_ONLY_GROUPED20_GRAPH",
            "not an actual S20d grouped20 diagnostics report")
    require(report.get("source_96_synthetic_graph_SHA256") ==
            FROZEN_S20_TRAIN96_SHA256 ==
            train_only_manifest()["source_sha256"], "TRAIN96 source SHA drift")
    require(report.get("physical_FP16_candidate_SHA256") ==
            model_sha == PINNED_S20D_FP16_SHA256, "S20d paid FP16 SHA drift")
    require(report.get("full_physical_FP16_parameters") == 12381 and
            report.get("frozen_parent_FP16_parameters") == 9332 and
            report.get("trainable_graph_FP16_parameters") == 3049 and
            report.get("optimizer_scope") == "ONLY_SEMANTIC_HEAD_3049_FP16",
            "physical grammar/graph scope drift")
    require(report.get("true_frozen_original43M_GT_from_source_S20b") is True and
            report.get("protected_original_616_J7_digest_unchanged_at_source_capture_ONLY") is True and
            report.get("no_EVAL32_EVAL128_or_prior_spent_EVAL") is True and
            report.get("NRA2_live_event_graph_decoder_bridge_not_connected") is True and
            report.get("output_gate") == 0 and
            report.get("candidate_accepted") is False and
            report.get("activation_allowed") is False,
            "read-only original protected J7/EVAL/decoder gate provenance drift")

    rows = build_s20_synthetic_train96()
    folds = train_only_split_contract()
    diagnostics = report.get("all20_fixed_TRAIN_internal_folds")
    require(isinstance(diagnostics, list) and len(diagnostics) == len(folds) == 20,
            "20 actual completed S20d grouped folds required")
    output = []
    for i, ((axis, group, train_ids, held_ids), observed) in enumerate(zip(folds, diagnostics)):
        require(observed.get("fold_number") == i and
                observed.get("axis") == axis and observed.get("held_group_index") == group and
                observed.get("TRAIN_side_rows") == len(train_ids) and
                observed.get("held_TRAIN_rows") == len(held_ids) and
                observed.get("held_rows_never_in_gradient_checkpoint_or_model_selection") is True and
                observed.get("original_true43M_GT_grammar_context_parent_frozen") is True,
                "actual S20d fold mismatch or held TRAIN leakage")
        train, tr_rejected = gold_support(rows, train_ids)
        held, held_rejected = gold_support(rows, held_ids)
        reported_train = observed.get("TRAIN_relation_label_support")
        require(isinstance(reported_train, dict) and
                all(reported_train.get(label) == train[label]
                    for label in EVENT_RELATION if label != "NONE") and
                observed.get("TRAIN_rejected_recipient_support") == tr_rejected and
                observed.get("held_TRAIN_rejected_recipient_support") == held_rejected,
                "actual recorded TRAIN label support differs from source labels")
        m = observed["held_TRAIN_metrics"]["event_relation"]
        require(m["TRAIN_INTERNAL_held_positive_support"] == sum(held.values()),
                "S20d held relation support mismatch")
        score = micro_f1(m)
        require(abs(score - m.get("TRAIN_INTERNAL_grouped_micro_F1_NOT_independent",
                                 score)) < 1e-8, "held arithmetic drift")
        no_exposure = [label for label in held if held[label] and train[label] == 0]
        output.append({
            "axis": axis,
            "group_index": group,
            "construction": PATTERNS[group] if axis == "CONSTRUCTION" else None,
            "held_synthetic_TRAIN_rows": len(held_ids),
            "TRAIN_relation_label_support": train,
            "held_TRAIN_relation_label_support": held,
            "unseen_relation_labels_in_TRAIN_side": no_exposure,
            "TRAIN_rejected_role_support": tr_rejected,
            "held_TRAIN_rejected_role_support": held_rejected,
            "relation_TRAIN_internal_held_TP_FP_FN": {k: m[k] for k in ("TP", "FP", "FN")},
            "relation_TRAIN_internal_held_micro_F1_NOT_EVAL": score,
            "REJECTED_specific_precision_recall_NOT_AVAILABLE": True,
        })

    sumrelations = Counter(l.relation for row in rows for l in row.event_relations)
    require(dict(sumrelations) ==
            {"COORDINATION": 8, "CONTRAST": 16,
             "CONDITION_RESULT": 16, "TEMPORAL_SEQUENCE": 24},
            "original S20 relation distribution changed")
    require(sum(a.polarity == "REJECTED" for r in rows for a in r.arguments) == 16,
            "rare role distribution drift")
    unseen = [r for r in output if r["unseen_relation_labels_in_TRAIN_side"]]
    require(len(unseen) == 1 and
            unseen[0]["construction"] == "asserted_coordination" and
            unseen[0]["unseen_relation_labels_in_TRAIN_side"] == ["COORDINATION"],
            "at least one TRAIN fold is structurally zero-shot: re-audit curriculum")
    for axis, key in (("LEXICAL", "lexical_8fold_TRAIN_internal_NOT_independent"),
                      ("CONSTRUCTION", "syntax_12fold_TRAIN_internal_NOT_independent")):
        scores = report[key]
        for task, agg in scores.items():
            matching = [f["held_TRAIN_metrics"][task] for f in diagnostics
                        if f["axis"] == axis]
            for metric in ("TP", "FP", "FN", "TRAIN_INTERNAL_held_positive_support"):
                require(sum(int(m[metric]) for m in matching) == agg[metric],
                        "published S20d fold aggregation mismatch")

    return {
        "stage": "NRA3_S21_ACTUAL_S20D_TRAIN96_SUPPORT_AND_UNMEASURED_RARE_CLASS_AUDIT",
        "S20_source_sha256": FROZEN_S20_TRAIN96_SHA256,
        "S20d_paid_12381_FP16_SHA256": model_sha,
        "S20d_completed_CI": PINNED_S20D_CI,
        "original_S20b_616_J7_readonly_capture_CI": REAL43M_PROTECTED_CAPTURE_CI,
        "S20d_actual_completed_fold_count": len(output),
        "S20_TRAIN_gold_relation_counts": dict(sumrelations),
        "S20_TRAIN_gold_rejected_role_count": 16,
        "structurally_unseen_relation_fold_count": len(unseen),
        "structurally_unseen_relation_folds": unseen,
        "all_TRAIN_only_group_fold_diagnostics": output,
        "critical_findings": [
            "COORDINATION occurs in only one of twelve templates: construction holdout has zero TRAIN examples.",
            "CONDITION_RESULT has only two templates; reversed-order generalization cannot be certified.",
            "REJECTED appears in only two templates and no per-class TP/FP/FN was recorded in S20d.",
            "Fold F1 is synthetic TRAIN-internal only. No evidence of independent Russian semantic competence.",
        ],
        "S22_precommitted_design_constraints": [
            "Add disjoint independently written syntactic constructions for each event relation, including COORDINATION.",
            "Counterbalance chronological clause order versus semantic event direction; avoid a positional shortcut.",
            "Add varied REJECTED role scopes and report REJECTED-specific TP/FP/FN separately.",
            "Freeze source/lexical/syntax groups and model recipe before training or accessing fresh independent EVAL.",
        ],
        "physical_original_Calibri43M_GT_untouched": True,
        "no_new_optimizer_no_model_forward_no_new_accuracy_claim": True,
        "no_old_or_new_independent_EVAL_access": True,
        "no_616_J7_or_inclusive_bpp_or_full_CPU_certification_of_candidate": True,
        "NRA2_decoder_connection_not_attempted": True,
        "LIVE_Render_Supabase_Jessica_APK_untouched": True,
        "output_gate": 0,
        "candidate_accepted": False,
        "activation_allowed": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--s20d-report", type=Path, required=True)
    p.add_argument("--s20d-paid-model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    require(a.s20d_report.is_file() and a.s20d_paid_model.is_file(),
            "must download actual S20d CI artifact; tests cannot substitute synthetic results")
    require(not a.out.exists(), "do not overwrite research results")
    data = json.loads(a.s20d_report.read_text(encoding="utf-8"))
    result = audit(data, file_sha(a.s20d_paid_model))
    a.out.mkdir(parents=True, exist_ok=False)
    destination = a.out / "NRA3_S21_S20D_20FOLD_REAL_ARTIFACT_SUPPORT_AUDIT.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+"\n",
                           encoding="utf-8")
    print("NRA3_S21_ACTUAL_S20D_ARTIFACT_SHA", result["S20d_paid_12381_FP16_SHA256"])
    print("NRA3_S21_MISSING_COORDINATION_TRAIN_SUPPORT_FOLDS",
          result["structurally_unseen_relation_fold_count"])
    print("NRA3_S21_REJECTED_CLASS_SPECIFIC_RECALL_NOT_AVAILABLE", True)


if __name__ == "__main__":
    main()
