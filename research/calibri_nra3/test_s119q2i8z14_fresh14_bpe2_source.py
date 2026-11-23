"""Z14 SOURCE-ONLY preregistration / budget protection (not model scoring)."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8z14_fresh14_bpe2_coverage_source import (
    BUDGETS,KINDS,rows,verify,digest,
)
from research.calibri_nra3.s119q2i8z12_parallel_native_new14_source import rows as spent_z12

class TestZ14NewSource(unittest.TestCase):
    def test_exact_sha_counts_and_budget(self):
        info=verify()
        self.assertEqual(info["new_DEV14"],14)
        self.assertEqual(info["source_sha256"],digest(rows("DEV")))
        self.assertEqual(info["budget_k"],(8,16,32,64,128,512))
        self.assertEqual(BUDGETS,(8,16,32,64,128,512))
        self.assertTrue(info["all_source_prompts_disjoint_E_G_K_R_T_W_X_Z2_Z9_Z12"])
    def test_seven_domains_and_z12_disjoint(self):
        self.assertEqual(Counter(r["kind"] for r in rows()),{k:2 for k in KINDS})
        self.assertEqual(len({r["id"] for r in rows()}),14)
        self.assertTrue({v["prompt"].strip().casefold() for v in rows()}.isdisjoint(
            x["prompt"].strip().casefold() for x in spent_z12()))
    def test_never_protected_or_training(self):
        for split in ("TRAIN","EVAL128","GOLD","J7","616","TEST"):
            with self.subTest(split=split),self.assertRaises(ValueError):
                rows(split)
if __name__=="__main__":
    unittest.main()
