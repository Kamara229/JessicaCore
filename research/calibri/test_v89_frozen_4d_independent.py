"""Pre-registration, physically paid source, independent TEST sampling."""
import unittest
from v89.provenance import (
    CB88_FROZEN_PRETEST_SHA, CB88_BYTES, SCI_Q, BOOL_Q,
    WIKI_START, NEWS_ROWS, N_WINDOWS, CONTEXT, PROMPTS)
from v88.physical import COMPLETE_BYTES,COMPLETE_BPW
from v89.stats import paired_kl

class FrozenPretest(unittest.TestCase):
    def test_full_source_all_in_model_bytes(self):
        self.assertEqual(CB88_BYTES,COMPLETE_BYTES)
        self.assertLess(COMPLETE_BPW,0.700)
        self.assertEqual(len(CB88_FROZEN_PRETEST_SHA),64)
    def test_official_true_unseen_test_ranges_in_dataset_bounds(self):
        self.assertTrue(344<SCI_Q[0]<SCI_Q[1]<840)
        self.assertTrue(3164<BOOL_Q[0]<BOOL_Q[1]<=3270)
        self.assertTrue(160512<WIKI_START)
        self.assertTrue(0<=NEWS_ROWS[0]<NEWS_ROWS[1]<=7600)
        self.assertEqual(N_WINDOWS*CONTEXT,1024)
        self.assertEqual(len(PROMPTS),4)
    def test_paired_window_KL_bootstrap(self):
        a=[{"teacher_student_KL":3.0} for _ in range(8)]
        b=[{"teacher_student_KL":2.0} for _ in range(8)]
        out=paired_kl(a,b,"teacher_student_KL")
        self.assertAlmostEqual(out["mean_change"],-1.0)
        self.assertEqual(out["original_independent_windows"],8)
        self.assertFalse(out["confidence_interval_crosses_zero"])
    def test_rejects_unpaired_language_windows(self):
        with self.assertRaises(AssertionError):
            paired_kl([{"teacher_student_KL":1.0}],[],"teacher_student_KL")
if __name__=="__main__":
    unittest.main()
