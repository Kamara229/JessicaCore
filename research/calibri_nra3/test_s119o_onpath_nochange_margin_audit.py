"""S119o contract tests: synthetic traces, NEVER hidden GOLD/model weights."""
import hashlib
import json
import unittest

from research.calibri_nra3.s119o_onpath_nochange_margin_audit import (
    PARENT_STAGE, PROMPT_GROUPS, REQUIRED_TRUE, audit,
)


def fixture():
    lengths = (1, 2, 2, 3, 4, 4)  # 16 generated steps => 3*16+6=54 decoder passes
    records = []
    for group, size in zip(PROMPT_GROUPS, lengths):
        question = "Unlabeled synthetic source only " + group
        native_ids = [41] * size
        trace = []
        for i in range(size):
            trace.append({
                "step": i + 1,
                "actual_native_current_prefix_length": 8 + i,
                "original_native_next_token_id": 41,
                "actual_trained_onpath_native_next_token_id": 41,
                "selected_token_id": 41,
                "original_top1_top2_logit_margin": 1.0,
                "actual_onpath_max_abs_logit_change": 0.1,
            })
        baseline = {
            "native_original_question_BPE_count": 8,
            "generated_native_token_ids": native_ids[:],
            "generated_text": "synthetic",
            "emitted_EOS": False,
            "original_full12layer_prefill_calls": size,
            "real_onpath_second_decoder_calls": 0,
            "additional_strict_zero_gate_control_calls": 0,
            "actual_native_steps": [dict(step) for step in trace],
        }
        candidate = {
            **baseline,
            "real_onpath_second_decoder_calls": size,
            "additional_strict_zero_gate_control_calls": 1,
            "actual_native_steps": [dict(step) for step in trace],
        }
        records.append({
            "category": group,
            "source_question": question,
            "initial_question_SHA256":
                hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "original_native_greedy": baseline,
            "actual_deep_onpath_native_greedy": candidate,
            "different_native_generated_token_sequences": False,
            "answer_grade_NOT_MEASURED": True,
        })
    report = {
        "stage": PARENT_STAGE,
        "candidate_accepted": False,
        "output_gate": 0,
        "physical_full_original_native_vocab_size": 49152,
        "n_unlabeled_author_prompts": 6,
        "prompts_where_trained_onpath_changes_generated_native_BPE": 0,
        "total_actual_original43m_forward_calls": 54,
        "S119m_trained_complete_physical_FP16_SHA256": "a" * 64,
        "original_S93_CB61_SHA256": "b" * 64,
        "original_S142_43M_GT_SHA256": "c" * 64,
        "research_records": records,
    }
    report.update({key: True for key in REQUIRED_TRUE})
    return report


def assess(report):
    raw = json.dumps(report, sort_keys=True).encode("utf-8")
    return audit(report, raw, expected_forwards=54)


class S119OMarginAudit(unittest.TestCase):
    def test_01_mathematically_stable_original_top1_certified(self):
        out = assess(fixture())
        self.assertEqual(out["generated_native_steps"], 16)
        self.assertEqual(out["actual_original43m_forwards_from_trace"], 54)
        self.assertEqual(out["observed_changed_sequences"], 0)
        self.assertEqual(
            out["top1_stability_certified_by_margin_gt_twice_Linf_delta"], 16)
        self.assertEqual(out["top1_stability_INCONCLUSIVE_by_bound"], 0)
        self.assertFalse(out["candidate_accepted"])
        self.assertEqual(out["output_gate"], 0)
        self.assertTrue(out["no_independent_answer_quality_result"])

    def test_02_not_certified_does_not_mean_changed(self):
        report = fixture()
        trace = report["research_records"][0][
            "actual_deep_onpath_native_greedy"]["actual_native_steps"]
        trace[0]["actual_onpath_max_abs_logit_change"] = 0.6
        out = assess(report)
        self.assertEqual(out["top1_stability_INCONCLUSIVE_by_bound"], 1)
        self.assertEqual(out["observed_changed_sequences"], 0)

    def test_03_reject_provenance_spoof_or_corrupted_hash(self):
        report = fixture()
        report["original_S142_43M_GT_SHA256"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "provenance"):
            assess(report)

    def test_04_reject_actual_token_change_on_nochange_report(self):
        report = fixture()
        report["research_records"][0][
            "actual_deep_onpath_native_greedy"]["generated_native_token_ids"][0] = 40
        with self.assertRaisesRegex(ValueError, "sequences must match"):
            assess(report)

    def test_05_reject_activated_or_promoted_model(self):
        for field, value in (("candidate_accepted", True), ("output_gate", 1)):
            with self.subTest(field=field):
                report = fixture()
                report[field] = value
                with self.assertRaisesRegex(ValueError, "promoted"):
                    assess(report)

    def test_06_reject_fake_cpu_or_bpp_or_j7_claim(self):
        for field in (
            "normal_KV_streaming_CPU_NOT_MEASURED",
            "all_in_bpp_NOT_measured",
            "original_616_J7_EOS_NOT_replayed",
            "sealed_S118_SynTagRusB_GOLD_not_accessed",
        ):
            with self.subTest(field=field):
                report = fixture()
                report[field] = False
                with self.assertRaisesRegex(ValueError, "safety contract"):
                    assess(report)

    def test_07_reject_wrong_step_or_partial_traffic(self):
        report = fixture()
        report["research_records"][1][
            "actual_deep_onpath_native_greedy"]["actual_native_steps"][0][
                "actual_native_current_prefix_length"] += 1
        with self.assertRaisesRegex(ValueError, "top1/self-feedback"):
            assess(report)

    def test_08_reject_unsupported_gold_scoring(self):
        report = fixture()
        report["research_records"][0]["answer_grade_NOT_MEASURED"] = False
        with self.assertRaisesRegex(ValueError, "GOLD"):
            assess(report)

    def test_09_avoid_ill_defined_nan_logit_deltas(self):
        report = fixture()
        report["research_records"][0][
            "actual_deep_onpath_native_greedy"]["actual_native_steps"][0][
                "actual_onpath_max_abs_logit_change"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            assess(report)

    def test_10_protect_parent_stage_and_total_forward_count(self):
        for field, value in (("stage", "fake"), ("total_actual_original43m_forward_calls", 21)):
            with self.subTest(field=field):
                report = fixture()
                report[field] = value
                with self.assertRaises(ValueError):
                    assess(report)


if __name__ == "__main__":
    unittest.main()
