"""Synthetic 428-vs-18 offline margin comparison; no model/QA/LIVE tests."""
import unittest

from calibri2.s180.compare_correct_and_wrong_second_margins import compare
from calibri2.s180.test_second_token_margin_posthoc import realistic_fixture


class S180CorrectWrongMarginContracts(unittest.TestCase):
    def test_exact_groups_with_unconditional_raw_margins(self):
        trace, gold = realistic_fixture()
        result = compare(trace, gold)
        self.assertEqual(result["frozen_correct_428"]["total"], 428)
        self.assertEqual(result["first_correct_second_wrong_18"]["total"], 18)
        self.assertEqual(result["other_wrong_10"]["total"], 10)
        self.assertEqual(result["frozen_correct_428"]["actual_token2_logits_observed"], 428)
        self.assertEqual(result["first_correct_second_wrong_18"]["top1_minus_top2_logits"]["median"], 1.0)
        self.assertEqual(result["within_route_comparison"]["S93"]["comparable_both_groups"], True)
        self.assertTrue(result["gold_join_after_inference_only"])
        self.assertFalse(result["model_accepted"])
        self.assertNotIn("target_ids", str(result))
        self.assertNotIn("full_generated_ids", str(result))
        self.assertNotIn("prompt_sha256", str(result))

    def test_s93_descriptive_overlap_no_learned_threshold(self):
        trace, labels = realistic_fixture()
        report = compare(trace, labels)
        stats = report["within_route_S93_descriptive_overlap"]
        self.assertEqual(stats["correct_with_observed_AR_logits"], 428)
        self.assertEqual(stats["wrong_second_with_observed_AR_logits"], 18)
        self.assertEqual(stats["pairwise_probability_correct_margin_gt_wrong"], 0.5)
        self.assertEqual(stats["wrong_margins_in_correct_minmax"], 18)
        self.assertEqual(stats["correct_margins_in_wrong_minmax"], 428)
        self.assertTrue(stats["no_threshold_chosen"])
        self.assertNotIn("expected", str(stats))

    def test_missing_logits_not_imputed_from_other_routes(self):
        trace, gold = realistic_fixture()
        trace["cases"][30]["route"] = "S87"
        trace["cases"][30]["conditional_step2"] = None
        result = compare(trace, gold)
        self.assertEqual(result["frozen_correct_428"]["no_actual_token2_logits"], 1)
        self.assertEqual(result["frozen_correct_428"]["top1_minus_top2_logits"]["count"], 427)
        self.assertIn("S87", result["frozen_correct_428"]["routes"])

    def test_reject_non_ar_logits_and_label_leaks(self):
        trace, gold = realistic_fixture()
        trace["cases"][0]["conditional_step2"]["score_type"] = "surrogate_span_head"
        with self.assertRaisesRegex(AssertionError, "surrogate"):
            compare(trace, gold)
        trace, gold = realistic_fixture()
        trace["cases"][0]["expected"] = [10, 20, 0]
        with self.assertRaisesRegex(AssertionError, "leakage"):
            compare(trace, gold)

    def test_invalid_source_cohort_fails_closed(self):
        trace, gold = realistic_fixture()
        trace["cases"][-1]["full_generated_ids"] = [10, 21, 0]
        with self.assertRaisesRegex(AssertionError, "source drift"):
            compare(trace, gold)


if __name__ == "__main__":
    unittest.main()
