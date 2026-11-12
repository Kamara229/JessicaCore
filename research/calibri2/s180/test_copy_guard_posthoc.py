"""S180 guard/copy observational contracts: synthetic ONLY, no Jessica/model."""
import unittest

from calibri2.s180.copy_guard_posthoc import assess
from calibri2.s180.test_second_token_margin_posthoc import realistic_fixture


class S180CopyGuardReadOnlyContracts(unittest.TestCase):
    def test_full_456_partition_and_no_causal_claim(self):
        traces, labels = realistic_fixture()
        result = assess(traces, labels)
        self.assertEqual([result["group_counts"][s]["total"] for s in
                         ("full_correct", "first_wrong",
                          "second_wrong_first_right", "other_wrong")],
                         [428, 9, 18, 1])
        self.assertEqual(result["group_counts"]["full_correct"]["actual_conditional_second_logits"], 428)
        self.assertFalse(result["model_accepted"])
        self.assertTrue(result["no_counterfactual_ablation"])
        self.assertTrue(result["no_causal_claim_or_threshold_tuning"])

    def test_copy_override_mismatch_is_not_guessed(self):
        traces, labels = realistic_fixture()
        row = traces["cases"][0]
        row["guard_events"] = ["candidate_present", "neural_override_applied"]
        row["conditional_step2"]["top5_token_ids_and_logits"] = [
            [22, 4.5], [21, 4.0], [20, 3.0], [0, -1.0]]
        result = assess(traces, labels)
        self.assertEqual(result["group_counts"]["second_wrong_first_right"]["copy_net_override"], 1)
        self.assertEqual(result["group_counts"]["second_wrong_first_right"]["raw_top1_differs_generated_second"], 2)
        self.assertEqual(result["group_counts"]["second_wrong_first_right"]["raw_final_disagreement_without_logged_copy_override"], 1)
        self.assertEqual(len(result["copy_net_override_sha_only"]), 1)
        self.assertEqual(len(result["raw_logit_disagreement_sha_only"]), 2)
        self.assertNotIn("expected", str(result))
        self.assertNotIn("target_ids", str(result))

    def test_override_requires_preexisting_candidate(self):
        traces, labels = realistic_fixture()
        traces["cases"][0]["guard_events"] = ["neural_override_applied"]
        with self.assertRaisesRegex(AssertionError, "without candidate"):
            assess(traces, labels)

    def test_fastpath_cannot_have_conditional_logits(self):
        traces, labels = realistic_fixture()
        traces["cases"][0]["guard_events"] = ["fastpath_no_autoregressive_step2"]
        with self.assertRaisesRegex(AssertionError, "no conditional"):
            assess(traces, labels)

    def test_guard_fallback_and_parser_abstain_are_observed_not_causes(self):
        traces, labels = realistic_fixture()
        traces["cases"][17]["guard_events"] = [
            "S160_confidence_fallback", "S167_owner_parser_abstained"]
        result = assess(traces, labels)
        self.assertEqual(result["event_x_group"]["low_confidence_fallback"]["first_wrong"], 1)
        self.assertEqual(result["event_x_group"]["owner_parser_abstain"]["first_wrong"], 1)

    def test_non_s93_cannot_appear_with_fake_logits(self):
        traces, labels = realistic_fixture()
        traces["cases"][29]["route"] = "S87"
        with self.assertRaisesRegex(AssertionError, "non-S93"):
            assess(traces, labels)
        traces["cases"][29]["conditional_step2"] = None
        result = assess(traces, labels)
        self.assertEqual(result["group_counts"]["full_correct"]["no_actual_conditional_second_logits"], 1)


if __name__ == "__main__":
    unittest.main()
