"""S119q2i8x protected source-only original RU 21TRAIN+14DEV gates."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8x_new_ru_train_dev_source import (
    STAGE,KINDS,rows,verify,
)
class TestNewXAuthorSource(unittest.TestCase):
    def test_01_exact_old_disjoint_provenance(self):
        x=verify()
        self.assertEqual(x["fresh_X_TRAIN_rows"],21)
        self.assertEqual(x["fresh_X_DEV_rows"],14)
        self.assertTrue(x["all_X_prompts_disjoint_prior_E_G_K_R_T_W"])
        self.assertEqual(len(x["X21_TRAIN_sha256"]),64)
        self.assertEqual(len(x["X14_DEV_sha256"]),64)
    def test_02_all_seven_domains_train_dev_no_overlap(self):
        a,b=rows("TRAIN"),rows("DEV")
        self.assertEqual(Counter(r["kind"] for r in a),{k:3 for k in KINDS})
        self.assertEqual(Counter(r["kind"] for r in b),{k:2 for k in KINDS})
        self.assertTrue({r["id"] for r in a}.isdisjoint(r["id"] for r in b))
    def test_03_protected_not_servable(self):
        for split in ("EVAL128","GOLD","J7","616","TEST"):
            with self.assertRaises(ValueError):
                rows(split)
if __name__=="__main__":unittest.main()
