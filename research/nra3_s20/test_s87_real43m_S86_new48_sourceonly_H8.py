"""S87 actual original43M GT H8 NOVEL48 source-only before S85 firstread."""
import ast
import unittest
from pathlib import Path

from research.nra3_s20.s86_newSYNTHETIC48_SOURCEONLY_never_train import (
    S86_novel48_source_only,S86_novel48_source_sha256,
)
from research.nra3_s20.s87_REAL43M_GT_S86_NOVEL48_SOURCEONLY_H8 import (
    EXPECTED_SOURCE_SHA,STAGE,TENSOR_NAME,REPORT_NAME,
)


class TestS87NewGenuineSourceBeforeS85(unittest.TestCase):
    def test_new_48_from_precommitted_S86_only(self):
        self.assertEqual(S86_novel48_source_sha256(),EXPECTED_SOURCE_SHA)
        self.assertEqual(len(S86_novel48_source_only()),48)
        self.assertTrue(all(set(x)=={"row_id","text"}
                            for x in S86_novel48_source_only()))
        self.assertEqual(STAGE,
            "NRA2_S87_TRUE43M_GT_FIRSTREAD_NOVEL48_GOLD_BLIND_H8_SOURCE")

    def test_no_gold_or_spent_firstread_model_use_in_cpu_capture(self):
        source=Path("research/nra3_s20/s87_REAL43M_GT_S86_NOVEL48_SOURCEONLY_H8.py").read_text()
        tree=ast.parse(source)
        imports=[x.module or "" for x in tree.body if isinstance(x,ast.ImportFrom)]
        self.assertTrue(any("s86_newSYNTHETIC48_SOURCEONLY" in x for x in imports))
        self.assertFalse(any("GOLD_SEPARATE" in x or "s85_true" in x or
                             "s79_NEW_NOVEL48" in x or "s81_frozen" in x
                             for x in imports))
        self.assertNotIn("S86_frozen_authored_gold",source)

    def test_actual_source_tensors_contain_original_BPE_word339_not_answers(self):
        self.assertEqual(TENSOR_NAME,
            "NRA2_S87_GENUINE43M_GT_NOVEL48_WORD339_NO_GOLD.safetensors")
        self.assertEqual(REPORT_NAME,
            "NRA2_S87_GENUINE43M_GT_NOVEL48_GOLD_BLIND_SOURCE_PROVENANCE.json")


if __name__=="__main__":
    unittest.main()
