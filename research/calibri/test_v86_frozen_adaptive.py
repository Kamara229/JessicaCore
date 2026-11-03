#!/usr/bin/env python3
"""Pure independent TEST source isolation, paired statistics and physical limits."""
import unittest
from v86.provenance import (
    TEST_SCIQ,TEST_BOOLQ,TEST_WIKI_START,TEST_NEWS,
    ORIGINAL_CB59_SHA,FROZEN_ADAPTIVE_CB85_SHA,
    FROZEN_ADAPTIVE_CB85_BYTES)
from v86.stats import paired_agreement


class FrozenIndependentTest(unittest.TestCase):
    def test_genuine_unused_question_spans(self):
        self.assertTrue(TEST_SCIQ[0]>=240 and TEST_SCIQ[1]<=1000)
        self.assertTrue(TEST_BOOLQ[0]>=3070 and TEST_BOOLQ[1]<=3270)
        self.assertTrue(TEST_WIKI_START>=150000)
        self.assertTrue(TEST_NEWS[0]>=7510 and TEST_NEWS[1]<=7600)

    def test_nonempty_frozen_verified_pretest_SHA(self):
        self.assertEqual(len(ORIGINAL_CB59_SHA),64)
        self.assertEqual(len(FROZEN_ADAPTIVE_CB85_SHA),64)
        self.assertNotEqual(ORIGINAL_CB59_SHA,FROZEN_ADAPTIVE_CB85_SHA)
        self.assertGreater(FROZEN_ADAPTIVE_CB85_BYTES,11229303)

    def test_paired_exact_significance_not_overstated(self):
        x=[1,0,1,0]
        y=[0,1,1,1]
        z=paired_agreement(x,y)
        self.assertEqual(z["base_only_correct"],1)
        self.assertEqual(z["adaptive_only_correct"],2)
        self.assertEqual(z["base_correct"],2)
        self.assertEqual(z["adaptive_correct"],3)
        self.assertEqual(z["paired_cases"],4)
        self.assertEqual(z["exact_two_sided_McNemar_p"],1.0)

    def test_unpaired_lists_rejected(self):
        with self.assertRaises(AssertionError):
            paired_agreement([1,0,1],[1,0])


if __name__=="__main__":
    unittest.main()
