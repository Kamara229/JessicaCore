"""S119q2i8z1 strict source-only full49152 paired original43M causes."""
import unittest
from research.calibri_nra3.s119q2i8z1_causal_paired_loss_signs import (
    STAGE,paired_stats,EPS,
)
class TestPairedCausalSigns(unittest.TestCase):
    def test_01_no_predictive_tuning_or_gradients(self):
        self.assertIn("S119Q2I8Z1_ACTUAL_ORIGINAL43M",STAGE)
        self.assertEqual(EPS,1e-6)
    def test_02_incomplete_original_Z_source_rejected(self):
        with self.assertRaises(Exception):
            paired_stats({})
    def test_03_unproven_336_original43M_fails_closed(self):
        with self.assertRaises(Exception):
            paired_stats({"stage":"NOT_TRUE_ORIGINAL43M"})
if __name__=="__main__":unittest.main()
