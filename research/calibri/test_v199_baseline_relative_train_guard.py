"""v199 regression test: full TRAIN diversity is baseline-relative, never QA-fitted."""
import unittest

from v198.guard import anti_collapse_gate, summarize_guard


def probes(token_count):
    pairs = {}
    for i in range(24):
        pairs[f"TRAIN-{i:02d}"] = {
            "domain": ("math", "logic", "fact", "language")[i // 6],
            "sides": {
                "left": {"first_token_id": i % token_count,
                         "expected": 111, "first_token_correct": False},
                "right": {"first_token_id": (i + 1) % token_count,
                          "expected": 222, "first_token_correct": False},
            },
        }
    return pairs


class BaselineRelativeFullTrainGateTest(unittest.TestCase):
    def test_25pct_full_floor_false_rejects_improvement_over_weak_v186(self):
        baseline, better = probes(4), probes(6)
        initial = summarize_guard(baseline)["first_token_diversity"]
        self.assertAlmostEqual(initial, 4 / 48)
        self.assertAlmostEqual(summarize_guard(better)["first_token_diversity"], 6 / 48)
        old = anti_collapse_gate(baseline, better, minimum_distinct_fraction=.25)
        self.assertFalse(old["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])
        self.assertTrue(old["diversity_below_precommitted_floor"])
        safe = anti_collapse_gate(baseline, better,
                                  minimum_distinct_fraction=min(.25, initial))
        self.assertTrue(safe["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])
        self.assertFalse(safe["diversity_regressed"])

    def test_relative_floor_still_rejects_full_train_collapse(self):
        baseline, collapsed = probes(4), probes(1)
        initial = summarize_guard(baseline)["first_token_diversity"]
        safe = anti_collapse_gate(baseline, collapsed,
                                  minimum_distinct_fraction=min(.25, initial))
        self.assertFalse(safe["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])
        self.assertTrue(safe["diversity_regressed"])

    def test_absolute_floor_continues_on_original_eight_pair_step_guard(self):
        baseline, better = probes(4), probes(5)
        keep = list(baseline)[:8]
        sampled_old = {k: baseline[k] for k in keep}
        sampled_new = {k: better[k] for k in keep}
        safety = anti_collapse_gate(sampled_old, sampled_new,
                                    minimum_distinct_fraction=.25)
        self.assertTrue(safety["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])


if __name__ == "__main__":
    unittest.main()
