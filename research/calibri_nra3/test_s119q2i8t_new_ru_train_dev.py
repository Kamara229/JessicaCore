"""S119q2i8t source registration vs all previous TRAIN and scored DEV."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8t_new_ru_train_dev_source import (
    KINDS,rows,verify,TRAIN_PER_KIND,DEV_PER_KIND,
)
class TestSource(unittest.TestCase):
    def test_01_source_disjoint_all_previous(self):
        x=verify()
        self.assertEqual(len(x["original_21_TRAIN_SHA256"]),64)
        self.assertEqual(len(x["original_14_DEV_SHA256"]),64)
        self.assertEqual(x["new_train_rows"],21)
        self.assertEqual(x["new_dev_rows"],14)
    def test_02_7x3_and_7x2(self):
        self.assertEqual(Counter(x["kind"] for x in rows("TRAIN")),{k:3 for k in KINDS})
        self.assertEqual(Counter(x["kind"] for x in rows("DEV")),{k:2 for k in KINDS})
        self.assertTrue(set(x["id"] for x in rows("TRAIN")).isdisjoint(
                        x["id"] for x in rows("DEV")))
    def test_03_protected_not_accessible(self):
        for split in ("GOLD","EVAL","EVAL128","J7","616","TEST"):
            with self.assertRaises(ValueError):rows(split)
if __name__=="__main__":unittest.main()
