"""S119q2i8w author new21 TRAIN / new14 original RU DEV provenance."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8w_new_ru_train_dev_source import (
    STAGE,KINDS,ORIGIN,rows,verify,
)
class TestAuthorDataW(unittest.TestCase):
    def test_strict_provenance(self):
        evidence=verify()
        self.assertEqual(evidence["fresh_TRAIN"],21)
        self.assertEqual(evidence["fresh_internal_DEV"],14)
        self.assertEqual(len(evidence["new_21_TRAIN_sha256"]),64)
        self.assertEqual(len(evidence["new_14_DEV_sha256"]),64)
        self.assertTrue(evidence["source_disjoint_E70_G_K_R_T"])
    def test_exact_7_domain_3train_and_2dev(self):
        self.assertEqual(Counter(x["kind"] for x in rows("TRAIN")),{k:3 for k in KINDS})
        self.assertEqual(Counter(x["kind"] for x in rows("DEV")),{k:2 for k in KINDS})
        self.assertTrue(all(x["origin"]==ORIGIN for x in rows("TRAIN")+rows("DEV")))
        self.assertTrue({x["id"] for x in rows("TRAIN")}.isdisjoint(
                         x["id"] for x in rows("DEV")))
    def test_protected_split_denied(self):
        for split in ("J7","616","EVAL128","GOLD","EVAL"):
            with self.assertRaises(ValueError):rows(split)
if __name__=="__main__":unittest.main()
