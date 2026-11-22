"""S119q2i8r 14 new original DEV provenance, no TRAIN conversion."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8r_new_ru_DEV_eos_source import (
    rows,verify,KINDS,ORIGIN,
)
class TestNewDev(unittest.TestCase):
    def test_exact14_balanced_disjoint(self):
        meta=verify()
        self.assertEqual(meta["exact_14_new_source_DEV"],14)
        self.assertEqual(len(meta["source_sha256"]),64)
        self.assertEqual(Counter(x["kind"] for x in rows()),{k:2 for k in KINDS})
    def test_dev_cannot_be_called_train(self):
        for split in ("TRAIN","EVAL","EVAL128","J7","GOLD"):
            with self.assertRaises(ValueError):
                rows(split)
    def test_multiword_answers_and_origin(self):
        self.assertTrue(all(len(x["completion"].split())>=3 and
                            x["origin"]==ORIGIN and x["split"]=="DEV"
                            for x in rows()))
if __name__=="__main__":
    unittest.main()
