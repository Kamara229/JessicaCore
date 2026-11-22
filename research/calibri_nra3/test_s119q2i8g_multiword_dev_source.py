"""S119q2i8g source split guards and balanced target labels."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import (
    KINDS,ORIGIN,rows,verify,
)

class TestMultiwordNewDEV(unittest.TestCase):
    def test_01_two_new_each_domain(self):
        meta=verify()
        self.assertEqual(meta["original_new_author_heldout_DEV_count"],14)
        self.assertEqual(len(meta["source_new_DEV_sha256"]),64)
        self.assertEqual(Counter(x["kind"] for x in rows()),{k:2 for k in KINDS})
    def test_02_DEV_never_can_be_train(self):
        for split in ("TRAIN","EVAL","SEALED_EVAL"):
            with self.assertRaises(ValueError):rows(split)
    def test_03_completion_is_multitoken_and_provenance_locked(self):
        self.assertTrue(all(x["split"]=="DEV" and x["origin"]==ORIGIN
                            and len(x["completion"].split())>=2 for x in rows()))
if __name__=="__main__":unittest.main()
