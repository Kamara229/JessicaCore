"""Independent unit tests for precommitted paired-exact S3BH significance.
No EVAL prompt, labels, tokens, model or results are imported here.
"""
from __future__ import annotations
import unittest
from research.nra2_bridge.stage3bh_once_only_s3ba_s3be_eval128 import exact_two_sided_mcnemar


class McNemarPairTest(unittest.TestCase):
    def test_null_pairs(self):
        self.assertEqual(exact_two_sided_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_two_sided_mcnemar(3, 3), 1.0)

    def test_exact_extremes(self):
        self.assertEqual(exact_two_sided_mcnemar(0, 1), 1.0)
        self.assertAlmostEqual(exact_two_sided_mcnemar(0, 6), 0.03125)
        self.assertAlmostEqual(exact_two_sided_mcnemar(0, 5), 0.0625)

    def test_symmetric_and_bounded(self):
        for a in range(17):
            for b in range(17):
                x=exact_two_sided_mcnemar(a,b)
                self.assertEqual(x,exact_two_sided_mcnemar(b,a))
                self.assertLessEqual(x,1.0)
                self.assertGreaterEqual(x,0.0)


if __name__=="__main__":
    unittest.main()
