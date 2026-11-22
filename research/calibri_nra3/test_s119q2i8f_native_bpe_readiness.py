"""S119q2i8f strict source and reserved eval boundary tests (no models)."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8f_native_bpe_readiness import (
    STAGE,MODEL,REV,SELECTION_PER_KIND,MAX_ANSWER_NATIVE,
)
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import rows,verify

class ReadinessTests(unittest.TestCase):
    def test_01_source_hash_exists(self):
        self.assertEqual(len(verify()["TRAIN_sha256"]),64)
        self.assertEqual(len(rows()),70)
    def test_02_same_frozen_model_source(self):
        self.assertEqual(MODEL,"HuggingFaceTB/SmolLM2-135M-Instruct")
        self.assertEqual(len(REV),40)
        self.assertEqual(MAX_ANSWER_NATIVE,48)
    def test_03_even_split_locked_before_tokenization(self):
        self.assertEqual(SELECTION_PER_KIND,3)
        self.assertEqual(Counter(r["kind"] for r in rows()),{
            k:10 for k in verify()["domains"]})
    def test_04_protected_labels_not_exposed(self):
        for split in ("DEV","EVAL","SEALED_EVAL"):
            with self.assertRaises(ValueError):rows(split)
if __name__=="__main__":unittest.main()
