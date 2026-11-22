"""S119q2i8z2 newly authored RU 21TRAIN and unused14DEV SHA gates."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8z2_new_ru_train_dev_source import KINDS,rows,verify
class TestNewZ2Source(unittest.TestCase):
    def test_01_new_21_and_14_true_disjoint(self):
        v=verify()
        self.assertEqual(v["fresh_Z2_TRAIN_rows"],21)
        self.assertEqual(v["fresh_Z2_DEV_rows"],14)
        self.assertTrue(v["all_Z2_prompts_disjoint_prior_E_G_K_R_T_W_X"])
    def test_02_exact_each_domain(self):
        self.assertEqual(Counter(x["kind"] for x in rows("TRAIN")),{k:3 for k in KINDS})
        self.assertEqual(Counter(x["kind"] for x in rows("DEV")),{k:2 for k in KINDS})
        self.assertEqual(len({x["prompt"] for x in rows("DEV")+rows("TRAIN")}),35)
    def test_03_no_protected(self):
        for bad in ("616","J7","GOLD","EVAL128","TEST"):
            with self.assertRaises(ValueError):rows(bad)
if __name__=="__main__":unittest.main()
