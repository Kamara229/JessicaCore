"""S178 paired posthoc scorer contract: labels never influence inference."""
import unittest

from calibri2.s178.paired_counterfactual_cpu_j7 import compare_after_decode


class S178PairedPosthocTests(unittest.TestCase):
    def test_gains_and_losses_are_not_conflated(self):
        gold = {"a": {"question": "first prompt", "expected": [2, 3, 0]},
                "b": {"question": "second prompt", "expected": [4, 5, 0]}}
        before = {"a": [2, 3, 0], "b": [4, 8, 0]}
        after = {"a": [2, 9, 0], "b": [4, 5, 0]}
        result = compare_after_decode("fixture", before, after, gold, {})
        self.assertEqual(result["baseline_full_correct"], 1)
        self.assertEqual(result["counterfactual_full_correct"], 1)
        self.assertEqual(result["lost_previous_full_EOS_ids"], ["a"])
        self.assertEqual(result["newly_correct_full_EOS_ids"], ["b"])
        self.assertEqual(result["lost_previous_first_token_ids"], [])
        self.assertEqual(len(result["changed_final_EOS_ids"]), 2)

    def test_first_token_regression_detected_independently(self):
        gold = {"a": {"question": "first prompt", "expected": [2, 3, 0]}}
        result = compare_after_decode("fixture", {"a": [2, 3, 0]},
                                     {"a": [7, 3, 0]}, gold, {})
        self.assertEqual(result["lost_previous_first_token_ids"], ["a"])

    def test_no_missing_identity_or_gold(self):
        gold = {"a": {"question": "first prompt", "expected": [2, 3, 0]}}
        with self.assertRaises(AssertionError):
            compare_after_decode("fixture", {"a": [2, 3, 0]},
                                 {"b": [2, 3, 0]}, gold, {})
        with self.assertRaises(AssertionError):
            compare_after_decode("fixture", {"a": [2, 3, 0]},
                                 {"a": [2, 3, 0]}, {}, {})


if __name__ == "__main__":
    unittest.main()
