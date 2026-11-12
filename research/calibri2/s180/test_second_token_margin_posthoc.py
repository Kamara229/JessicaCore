"""S180 margin tests use synthetic prompts and source tokens; no model calls."""
import copy
import unittest

from calibri2.s180.second_token_margin_posthoc import margins
from calibri2.s180.test_offline_failure_attribution import fixture


def realistic_fixture():
    trace, report = fixture()
    # S180 exact profile: 9 wrong token1, 18 wrong token2 after correct
    # token1 (including 1 early EOS), and one EOS-only disagreement.
    for index in range(17, 26):
        trace["cases"][index]["full_generated_ids"] = [99, 21, 0]
    trace["cases"][26]["full_generated_ids"] = [10, 0]
    trace["cases"][27]["full_generated_ids"] = [10, 20, 20]
    trace["cases"][27]["terminated_in_eos"] = False
    return trace, report


class S180FrozenLogitMargins(unittest.TestCase):
    def test_analyzes_only_18_first_correct_second_wrong(self):
        data = margins(*realistic_fixture())
        self.assertEqual(data["original_train_correct"], 428)
        self.assertEqual(data["token2_errors_after_correct_first"], 18)
        self.assertEqual(len(data["sha_only_cases"]), 18)
        self.assertEqual(data["gold_second_in_observed_step2_top5"], 18)
        self.assertEqual(data["top1_minus_gold_logit_distribution"]["median"], 1.0)
        self.assertTrue(data["gold_was_offline_only"])
        self.assertFalse(data["model_accepted"])
        self.assertNotIn("target_ids", str(data))
        self.assertNotIn("full_generated_ids", str(data))

    def test_conditional_logits_absent_are_not_invented(self):
        t, r = realistic_fixture()
        t["cases"][0]["conditional_step2"] = None
        result = margins(t, r)
        self.assertEqual(result["conditional_step2_available"], 17)
        self.assertEqual(result["missing_or_out_of_top5"]["no_conditional_step2"], 1)
        self.assertFalse(result["sha_only_cases"][0]["step2_logits_present"])

    def test_unexpected_subset_fails_closed(self):
        t, r = fixture()  # All 28 are wrong on second token
        with self.assertRaisesRegex(AssertionError, "18"):
            margins(t, r)

    def test_top5_malformed_fails_closed(self):
        t, r = realistic_fixture()
        t["cases"][0]["conditional_step2"]["top5_token_ids_and_logits"] = [
            [20, 2.0], [21, 4.0]]
        with self.assertRaisesRegex(AssertionError, "malformed observed top5"):
            margins(t, r)

    def test_raw_top1_vs_frozen_output_must_not_be_conflated(self):
        data = margins(*realistic_fixture())
        self.assertEqual(data["observed_raw_top1_matches_generated_second"]["false"], 1)
        self.assertEqual(data["observed_raw_top1_matches_generated_second"]["true"], 17)


if __name__ == "__main__":
    unittest.main()
