"""S119o: SHA-bound, stdlib-only diagnostic of S119n's unchanged native tokens.

NO model forward, new training, heldout GOLD, production IO, or claim of
generalization. Only reads the frozen S119n research artifact and quantifies
how many unchanged top-1 decisions are CERTIFIED by a logit L-infinity bound:
baseline top1-top2 margin > 2 * max_abs(onpath - baseline logits).
When the inequality fails, the ranking might still remain unchanged.
This is NOT a scalable training recipe or independent quality evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

STAGE = "S119O_S119N_NATIVE_NOCHANGE_LOGIT_MARGIN_AUDIT_ONLY"
PARENT_STAGE = "S119N_REAL_S142_43M_GT_S119M_ONPATH_AUTOREGRESSIVE_SIXPROMPTS"
REPORT_NAME = "S119O_NATIVE_ONPATH_NOCHANGE_MARGIN_AUDIT.json"
PROMPT_GROUPS = (
    "arithmetic", "event_word_order", "negation",
    "comparison", "time", "paraphrase",
)
REQUIRED_TRUE = (
    "genuine_original_decoder_forward_called_for_each_greedy_native_token",
    "genuine_after_layer5_onpath_through_original_layers6_to11",
    "each_independent_candidate_generated_native_token_is_fed_back",
    "prompt_route_S95_held_constant_through_both_streams",
    "zero_new_interlayer_gate_reproduces_original_full_native_logits",
    "original_pretrained_43m_GT_unchanged_digest",
    "all_prior_S115_53751_NRA3_FP16_exact_unchanged",
    "source_questions_unlabeled_not_independent_quality_scoring",
    "not_full_S101_J7_answer_controller",
    "not_one_pass_model_extra_original_decoder_cost",
    "normal_KV_streaming_CPU_NOT_MEASURED",
    "all_in_bpp_NOT_measured",
    "original_616_J7_EOS_NOT_replayed",
    "not_end_to_end_updating_original_base_43m",
    "no_heldout_QA_gold_answers",
    "sealed_S118_SynTagRusB_GOLD_not_accessed",
    "no_live_activation",
)


def demand(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("S119o FAIL CLOSED: " + message)


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nonnegative_finite(value: object, label: str) -> float:
    demand(isinstance(value, (int, float)) and not isinstance(value, bool),
           label + " not numeric")
    v = float(value)
    demand(math.isfinite(v) and v >= 0, label + " negative or non-finite")
    return v


def audit(parent: dict, parent_bytes: bytes, *, expected_forwards: int | None = 54) -> dict:
    """Examine only paired recorded native-ID traces, never re-run the model."""
    demand(isinstance(parent, dict), "parent must be JSON object")
    demand(parent.get("stage") == PARENT_STAGE, "not real S119n report")
    demand(parent.get("candidate_accepted") is False and
           parent.get("output_gate") == 0, "promoted/activated candidate")
    demand(parent.get("physical_full_original_native_vocab_size") == 49152,
           "not original full native BPE head")
    demand(parent.get("n_unlabeled_author_prompts") == 6,
           "expected six new unlabeled prompts")
    demand(parent.get("prompts_where_trained_onpath_changes_generated_native_BPE") == 0,
           "this bounded NO-CHANGE analysis is not valid for changed sequences")
    for field in REQUIRED_TRUE:
        demand(parent.get(field) is True, "missing research safety contract: " + field)
    for field in ("S119m_trained_complete_physical_FP16_SHA256",
                  "original_S93_CB61_SHA256", "original_S142_43M_GT_SHA256"):
        demand(_is_hash(parent.get(field)), "missing/invalid provenance: " + field)

    records = parent.get("research_records")
    demand(isinstance(records, list) and len(records) == 6,
           "six complete records required")
    demand(tuple(rec.get("category") for rec in records) == PROMPT_GROUPS,
           "the exact six S119n prompt groups/order required")

    all_steps = []
    guarded_count = 0
    forward_count = 0
    per_prompt = []
    for rec in records:
        question = rec.get("source_question")
        demand(isinstance(question, str) and bool(question.strip()),
               "source-only original prompt missing")
        demand(rec.get("initial_question_SHA256") == _sha(question.encode("utf-8")),
               "prompt SHA mismatch")
        demand(rec.get("answer_grade_NOT_MEASURED") is True,
               "cannot import self-graded or GOLD-labeled accuracy")
        orig = rec.get("original_native_greedy")
        onpath = rec.get("actual_deep_onpath_native_greedy")
        demand(isinstance(orig, dict) and isinstance(onpath, dict),
               "paired actual generation streams required")
        original_ids = orig.get("generated_native_token_ids")
        candidate_ids = onpath.get("generated_native_token_ids")
        demand(isinstance(original_ids, list) and len(original_ids) > 0
               and len(original_ids) <= 8 and original_ids == candidate_ids
               and all(isinstance(v, int) and 0 <= v < 49152 for v in original_ids),
               "full-vocab original/candidate BPE sequences must match")
        demand(rec.get("different_native_generated_token_sequences") is False,
               "unchanged sequence declaration disagrees with bytes")
        original_trace = orig.get("actual_native_steps")
        candidate_trace = onpath.get("actual_native_steps")
        demand(isinstance(original_trace, list) and
               isinstance(candidate_trace, list) and
               len(original_trace) == len(candidate_trace) == len(original_ids),
               "genuine per-token paired traces missing")
        demand(orig.get("original_full12layer_prefill_calls") == len(original_ids) and
               onpath.get("original_full12layer_prefill_calls") == len(candidate_ids) and
               orig.get("real_onpath_second_decoder_calls") == 0 and
               onpath.get("real_onpath_second_decoder_calls") == len(candidate_ids) and
               orig.get("additional_strict_zero_gate_control_calls") == 0 and
               onpath.get("additional_strict_zero_gate_control_calls") == 1,
               "claimed physical original CPU forward call counts inconsistent")
        demand(orig.get("emitted_EOS") == onpath.get("emitted_EOS") and
               orig.get("generated_text") == onpath.get("generated_text") and
               orig.get("native_original_question_BPE_count") ==
                   onpath.get("native_original_question_BPE_count"),
               "paired source or EOS/truncated decoded text mismatch")
        forward_count += len(original_ids) * 3 + 1
        protected = 0
        for step_no, (base_step, intervention_step, selected_id) in enumerate(
            zip(original_trace, candidate_trace, candidate_ids), 1
        ):
            expected_prefix = onpath["native_original_question_BPE_count"] + step_no - 1
            demand(base_step.get("step") == intervention_step.get("step") == step_no and
                   base_step.get("actual_native_current_prefix_length") ==
                   intervention_step.get("actual_native_current_prefix_length") ==
                   expected_prefix and
                   base_step.get("selected_token_id") ==
                   base_step.get("original_native_next_token_id") ==
                   intervention_step.get("selected_token_id") ==
                   intervention_step.get("original_native_next_token_id") ==
                   intervention_step.get("actual_trained_onpath_native_next_token_id") ==
                   selected_id,
                   "actual native top1/self-feedback trace mismatch")
            margin = _nonnegative_finite(
                intervention_step.get("original_top1_top2_logit_margin"), "margin")
            delta = _nonnegative_finite(
                intervention_step.get("actual_onpath_max_abs_logit_change"), "delta")
            headroom = margin - 2.0 * delta
            certified = margin > 2.0 * delta
            protected += int(certified)
            all_steps.append({
                "group": rec["category"],
                "native_step": step_no,
                "baseline_top1_top2_margin": margin,
                "onpath_max_abs_full_vocab_logit_delta": delta,
                "margin_minus_twice_max_logit_delta": headroom,
                "top1_unchanged_CERTIFIED_by_Linf_bound": certified,
            })
        guarded_count += protected
        per_prompt.append({
            "group": rec["category"],
            "generated_native_tokens": len(candidate_ids),
            "Linf_bound_CERTIFIED_stable_steps": protected,
            "Linf_bound_INCONCLUSIVE_steps": len(candidate_ids) - protected,
            "output_changed": False,
        })
    demand(forward_count == parent.get("total_actual_original43m_forward_calls"),
           "report forward-call total does not match trace")
    if expected_forwards is not None:
        demand(forward_count == expected_forwards,
               "physical S119n published 54-forward count changed")
    demand(len(all_steps) > 0, "empty true-native-token trace")
    return {
        "stage": STAGE,
        "parent_stage": PARENT_STAGE,
        "parent_report_SHA256": _sha(parent_bytes),
        "original_s142_SHA256": parent["original_S142_43M_GT_SHA256"],
        "original_s93_SHA256": parent["original_S93_CB61_SHA256"],
        "trained_s119m_checkpoint_SHA256":
            parent["S119m_trained_complete_physical_FP16_SHA256"],
        "original_full_native_BPE_vocabulary": 49152,
        "unlabeled_prompts": 6,
        "generated_native_steps": len(all_steps),
        "actual_original43m_forwards_from_trace": forward_count,
        "observed_changed_sequences": 0,
        "top1_stability_certified_by_margin_gt_twice_Linf_delta": guarded_count,
        "top1_stability_INCONCLUSIVE_by_bound": len(all_steps) - guarded_count,
        "min_baseline_margin": min(s["baseline_top1_top2_margin"] for s in all_steps),
        "max_onpath_Linf_logit_delta": max(
            s["onpath_max_abs_full_vocab_logit_delta"] for s in all_steps),
        "prompt_summaries": per_prompt,
        "native_step_bound_evidence": all_steps,
        "possible_bottleneck_ONLY_hypothesis":
            "If many steps are certified stable, the learned onpath residual is "
            "too small in logit L-infinity norm relative to original margin "
            "FOR THESE positions. It is not a proof of global cause or a "
            "recommendation to amplify the gate without independent controls.",
        "no_training_or_model_forward": True,
        "heldout_gold_or_spent_evaluation_used": False,
        "no_independent_answer_quality_result": True,
        "no_new_616J7_or_CPU_or_bpp_gate_claim": True,
        "candidate_accepted": False,
        "output_gate": 0,
        "no_live_main_render_supabase_deploy_or_Jessica": True,
        "decision": "DIAGNOSTIC_ONLY_NEEDS_DIVERSE_TRAIN_AND_BLIND_TEST_NOT_PROMOTED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-forwards", type=int, default=54)
    args = parser.parse_args()
    demand(args.parent_report.name ==
           "S119N_REAL43M_GT_NATIVE_ONPATH_GENERATED_SIX_SOURCE_PROMPTS.json",
           "S119n specific pinned research report filename")
    demand(not args.out.exists(), "output overwrite forbidden")
    raw = args.parent_report.read_bytes()
    result = audit(json.loads(raw.decode("utf-8")), raw,
                   expected_forwards=args.expected_forwards)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / REPORT_NAME).write_text(
        json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("S119O_NOCHANGE_MARGIN_AUDIT " + json.dumps({
        "generated_native_steps": result["generated_native_steps"],
        "changed_sequences": result["observed_changed_sequences"],
        "stable_steps_certified": result[
            "top1_stability_certified_by_margin_gt_twice_Linf_delta"],
        "inconclusive_steps": result["top1_stability_INCONCLUSIVE_by_bound"],
        "max_onpath_Linf_delta": result["max_onpath_Linf_logit_delta"],
        "min_baseline_margin": result["min_baseline_margin"],
        "independent_quality_measured": False,
        "accepted": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
