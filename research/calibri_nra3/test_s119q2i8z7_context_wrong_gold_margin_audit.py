"""S119q2i8z7 exact original43M 84 REAL completed 336 source only evidence."""
import unittest
from research.calibri_nra3.s119q2i8z7_context_wrong_gold_margin_audit import (
    STAGE,analyze,GAIN
)
class TestZ7Strict(unittest.TestCase):
    def test_01_no_fake_original43m_causal_source(self):
        self.assertIn("ORIGINAL43M_NATIVE49152",STAGE)
        self.assertEqual(GAIN,.25)
        for data in ({},{"stage":"NOT_REAL43M"},{"candidate_accepted":True}):
            with self.assertRaises(Exception):analyze(data)
    def test_02_claimed_physical_count_without_real_sha_not_enough(self):
        with self.assertRaises(Exception):
            analyze({"physical_original_S142_43M_all12GT_native49152_full_forwards":84,
                     "analytical_original49k_same_true_hidden_and_same_source_virtual_variants":336})
if __name__=="__main__":unittest.main()
