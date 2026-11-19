"""S84 original43M GT 72 NEW TRAIN source-only true BPE capture ABI."""
import ast
import unittest
from pathlib import Path
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,source_only_new72_sha256,
)
from research.nra3_s20.s84_real43m_diverse72_rawonly_native_H8 import (
    EXPECTED_SOURCE_SHA,STAGE,TENSOR_NAME,REPORT_NAME,
)


class TestS84Genuine72Only(unittest.TestCase):
    def test_source_sha_precommitted_and_rows_exactly_72(self):
        self.assertEqual(source_only_new72_sha256(),EXPECTED_SOURCE_SHA)
        self.assertEqual(len(source_only_new72()),72)
        self.assertTrue(all(set(z)=={"row_id","group","text"}
                            for z in source_only_new72()))
        self.assertEqual(STAGE,
            "NRA2_S84_TRUE43M_GT_FIRSTREAD_DIVERSE72_GOLD_BLIND_H8_SOURCE")

    def test_source_only_original_cpu_never_imports_gold_S79_spent_or_model_scores(self):
        path=Path("research/nra3_s20/s84_real43m_diverse72_rawonly_native_H8.py")
        code=path.read_text(encoding="utf8")
        tree=ast.parse(code)
        modules=[z.module or "" for z in tree.body if isinstance(z,ast.ImportFrom)]
        self.assertTrue(any("s83_NEW_DIVERSE72_source_ONLY" in s for s in modules))
        self.assertTrue(all("gold_TRAIN_LOSS_ONLY" not in s and
                            "s79_NEW_NOVEL48_SOURCE" not in s and
                            "s78_train_true43m" not in s
                            for s in modules))
        self.assertNotIn("S83_train_loss_gold_only",code)
        self.assertNotIn("frozen_novel48_labels_for_ONE_FINAL_READ_only",code)

    def test_tensor_report_new_72_without_correct_answers(self):
        self.assertEqual(TENSOR_NAME,
            "NRA2_S84_GENUINE43M_GT_DIVERSE72_WORD339_NO_GOLD.safetensors")
        self.assertEqual(REPORT_NAME,
            "NRA2_S84_GENUINE43M_GT_DIVERSE72_GOLD_BLIND_SOURCE_PROVENANCE.json")


if __name__=="__main__":
    unittest.main()
