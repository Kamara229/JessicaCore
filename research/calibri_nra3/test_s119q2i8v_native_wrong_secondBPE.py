"""S119q2i8v true 49k native EOS-vs-other-wrong classifiers after goldBPE1."""
import unittest
from research.calibri_nra3.s119q2i8v_native_wrong_secondBPE_eos_vs_other import (
    STAGE,EOS,VOCAB,choice_category,analyze,
)
class TestWrongSecondBPE(unittest.TestCase):
    def test_01_full_native_classifier_EOS_vs_other(self):
        self.assertEqual(EOS,2)
        self.assertEqual(VOCAB,49152)
        self.assertIn("EOS_VS_WRONG_OTHER",STAGE)
        self.assertEqual(choice_category({"gold_native_id":10,"greedy_original_native_id":2}),
                         "WRONG_PREMATURE_EOS")
        self.assertEqual(choice_category({"gold_native_id":10,"greedy_original_native_id":33}),
                         "WRONG_OTHER_NON_EOS_NATIVE_BPE")
        self.assertEqual(choice_category({"gold_native_id":10,"greedy_original_native_id":10}),
                         "CORRECT_NONTERMINAL_GOLD")
    def test_02_gold_EOS_target_for_secondBPE_rejected(self):
        with self.assertRaises(ValueError):
            choice_category({"gold_native_id":2,"greedy_original_native_id":5})
        with self.assertRaises(ValueError):
            choice_category({"gold_native_id":10,"greedy_original_native_id":49152})
    def test_03_reject_noncompleted_sources(self):
        with self.assertRaises(Exception):analyze({}, {})
if __name__=="__main__":unittest.main()
