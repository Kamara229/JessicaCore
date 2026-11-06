"""v208: precommitted progressive physical source-layer continuation checks."""
import unittest

from v208.run import (
    PILOT_STEPS, SOURCE_STEP_OFFSET, SOURCE_V206_SHA256,
    balanced_domain_index, TRAIN_GUARD_PAIR_IDS,
)


class TestPaid48OperatorContinuation(unittest.TestCase):
    def test_exact_next_forty_eight_no_overlap_with_v206(self):
        schedule=[f"paid_matrix_{i}" for i in range(211)]
        previous=set(schedule[:48])
        next48=schedule[SOURCE_STEP_OFFSET:SOURCE_STEP_OFFSET+PILOT_STEPS]
        self.assertEqual(SOURCE_STEP_OFFSET,48)
        self.assertEqual(PILOT_STEPS,48)
        self.assertEqual(len(next48),48)
        self.assertEqual(len(set(next48)),48)
        self.assertTrue(previous.isdisjoint(next48))
        self.assertEqual(len(schedule),211)

    def test_frozen_train_pairs_twice_and_eight_guard_pairs(self):
        self.assertEqual(len(TRAIN_GUARD_PAIR_IDS),8)
        self.assertEqual(len(set(TRAIN_GUARD_PAIR_IDS)),8)
        one=[balanced_domain_index(s) for s in range(24)]
        two=[balanced_domain_index(s) for s in range(24,48)]
        self.assertEqual(sorted(one),list(range(24)))
        self.assertEqual(one,two)

    def test_source_sha256_locked_before_any_new_model_fitting(self):
        self.assertEqual(SOURCE_V206_SHA256,
            "5a6d5f07f9e49fa6063dd772d7e2aee53beeb3527d10d5a429715593b68939de")
        self.assertEqual(len(SOURCE_V206_SHA256),64)


if __name__=="__main__":
    unittest.main()
