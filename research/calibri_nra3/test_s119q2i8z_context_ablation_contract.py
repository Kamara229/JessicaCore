"""S119q2i8z strict original43M physical GT12 native head CORE gate masks."""
import unittest
from research.calibri_nra3.s119q2i8z_context_ablation_contract import (
    STAGE,VOCAB,SOURCE_COUNT,MODELS,CONDITIONS,FULL,config,aggregate,
    EXPECTED_REAL_ORIGINAL_43M_FORWARDS,
)
class TestZReadOnlyCausal(unittest.TestCase):
    def test_01_predeclared_true_original43m_budgets(self):
        self.assertEqual(VOCAB,49152)
        self.assertEqual(SOURCE_COUNT,14)
        self.assertEqual(len(MODELS),3)
        self.assertEqual(len(CONDITIONS),8)
        self.assertEqual(EXPECTED_REAL_ORIGINAL_43M_FORWARDS,336)
        self.assertIn("ORIGINAL43M_GT12",STAGE)
    def test_02_every_GT_head_and_region_toggle_is_within_original(self):
        self.assertEqual(config(FULL),("GT_RECURRENT",1.,.25,(1,1,1)))
        self.assertEqual(config("ZERO_ALL12_GT_GATES_ALL_REGIONS_HEAD_GAIN025"),
                         ("GT_RECURRENT",0.,.25,(1,1,1)))
        self.assertEqual(config("HEAD_GAIN_ZERO_GT12_AND_REGIONS_FITTED"),
                         ("GT_RECURRENT",1.,0.,(1,1,1)))
        for key in CONDITIONS:
            self.assertEqual(len(config(key)),4)
        with self.assertRaises(Exception):config("TUNE_GAIN_TO_FIT_DEV")
    def test_03_aggregator_fail_closed_on_missing_original_physical_336(self):
        with self.assertRaises(Exception):aggregate([])
if __name__=="__main__":unittest.main()
