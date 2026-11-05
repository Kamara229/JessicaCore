"""Deterministic adversarial unit tests for TRAIN-only source rollback."""
import unittest
from copy import deepcopy
from v198.guard import anti_collapse_gate, summarize_guard


def freeze(probes=4):
    result = {}
    for i in range(probes):
        left = 100 + 2*i
        right = left + 1
        result[i] = {"domain": ["numeric", "factual", "arithmetic", "binding"][i % 4],
                     "sides": {
                         "left": {"first_token_id": left, "expected": left,
                                  "first_token_correct": True},
                         "right": {"first_token_id": right, "expected": right,
                                   "first_token_correct": True}}}
    return result


class TestTrainOnlyGuard(unittest.TestCase):
    def test_identical_signed_train_behavior_survives(self):
        initial = freeze()
        result = anti_collapse_gate(initial, deepcopy(initial))
        self.assertTrue(result["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])
        self.assertTrue(result["TRAIN_only_not_full_answer_quality"])

    def test_constant_red_collapse_rejected_even_without_a_loss(self):
        initial = freeze()
        degenerate = deepcopy(initial)
        for row in degenerate.values():
            for side in row["sides"].values():
                side["first_token_id"] = 444
                side["first_token_correct"] = False
        result = anti_collapse_gate(initial, degenerate)
        self.assertFalse(result["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])
        self.assertEqual(result["distinct_first_tokens_after"], 1)
        self.assertTrue(result["lost_original_both_correct_opposite_TRAIN_pairs"])

    def test_previous_correct_token_cannot_be_sacrificed(self):
        initial = freeze()
        regressed = deepcopy(initial)
        regressed[1]["sides"]["right"]["first_token_id"] = 222
        regressed[1]["sides"]["right"]["first_token_correct"] = False
        result = anti_collapse_gate(initial, regressed)
        self.assertFalse(result["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])
        self.assertIn((1,"right"), map(tuple,result["lost_previous_correct_TRAIN_first_tokens"]))

    def test_unmatched_or_rewritten_samples_rejected(self):
        initial = freeze()
        bad = deepcopy(initial)
        bad.pop(3)
        with self.assertRaises(ValueError):
            anti_collapse_gate(initial,bad)
        rewritten = deepcopy(initial)
        rewritten[0]["sides"]["left"]["expected"] += 1000
        with self.assertRaises(ValueError):
            anti_collapse_gate(initial,rewritten)

    def test_source_diversity_cannot_decrease_even_with_some_correctness(self):
        initial = freeze()
        candidate = deepcopy(initial)
        # Second token was not correct in immutable v186; candidate remains
        # non-correct, but a mode collapse still cannot be accepted.
        initial[0]["sides"]["left"]["first_token_id"] = 777
        initial[0]["sides"]["left"]["first_token_correct"] = False
        candidate[0]["sides"]["left"]["first_token_id"] = 101
        candidate[0]["sides"]["left"]["first_token_correct"] = False
        result = anti_collapse_gate(initial,candidate)
        self.assertFalse(result["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])
        self.assertTrue(result["diversity_regressed"])

    def test_train_guard_has_no_model_parity_claim(self):
        x = summarize_guard(freeze())
        self.assertEqual(x["prompt_count"],8)
        self.assertEqual(x["first_token_diversity"],1)


if __name__ == "__main__":
    unittest.main()
