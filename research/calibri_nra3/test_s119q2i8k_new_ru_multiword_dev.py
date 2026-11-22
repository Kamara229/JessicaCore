"""S119q2i8k 14 NEW source-disjoint multiword RU DEV hard gates."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8k_new_ru_multiword_dev_source import (
    rows,verify,KINDS,ORIGIN,
)
class TestNewKDev(unittest.TestCase):
    def test_01_new_14_disjoint_sha(self):
        meta=verify()
        self.assertEqual(meta["new_DEV"],14)
        self.assertEqual(len(meta["sha256"]),64)
    def test_02_two_per_domain(self):
        self.assertEqual(Counter(x["kind"] for x in rows()),{k:2 for k in KINDS})
    def test_03_no_gradient_source(self):
        for split in ("TRAIN","EVAL","SEALED_EVAL"):
            with self.assertRaises(ValueError):rows(split)
    def test_04_answer_is_multiple_russian_words(self):
        self.assertTrue(all(x["origin"]==ORIGIN and x["split"]=="DEV"
                            and len(x["completion"].split())>=3 for x in rows()))
if __name__=="__main__":unittest.main()
