"""S80 original 43M BPE-firstread source-only NOVEL48 contracts, not scores."""
import ast
import unittest
from pathlib import Path

from research.nra3_s20.s79_NEW_NOVEL48_SOURCE_ONLY_locked_before_S78_firstread import (
    new_novel48_source_sha256,new_novel48_source_only,
)
from research.nra3_s20.s80_real43m_novel48_sourceonly_native_H8 import (
    EXPECTED_SOURCE_SHA,STAGE,TENSOR_NAME,REPORT_NAME,
)


class TestS80True43MNovel48SourceOnly(unittest.TestCase):
    def test_precommitted_new_novel48_source_locked(self):
        self.assertEqual(new_novel48_source_sha256(),EXPECTED_SOURCE_SHA)
        self.assertEqual(len(new_novel48_source_only()),48)
        self.assertEqual(
            STAGE,"NRA2_S80_TRUE43M_GT_FIRSTREAD_NOVEL48_GOLD_BLIND_H8_SOURCE")

    def test_pretraining_original_module_never_import_constructed_gold_or_S78(self):
        path=Path("research/nra3_s20/s80_real43m_novel48_sourceonly_native_H8.py")
        code=path.read_text(encoding="utf8")
        tree=ast.parse(code)
        imports=[]
        for node in tree.body:
            if isinstance(node,ast.ImportFrom):
                imports.append(node.module or "")
            if isinstance(node,ast.Import):
                imports.extend(x.name for x in node.names)
        self.assertFalse(any("s79_NEW_NOVEL48_LABELS" in s or
                             "s78_train" in s or "EVAL128" in s or
                             "test_s79" in s for s in imports))
        self.assertTrue(any(
            "s79_NEW_NOVEL48_SOURCE_ONLY" in s for s in imports))
        self.assertNotIn("frozen_novel48_labels_for_ONE_FINAL_READ_only",code)

    def test_sourceonly_artifact_types_not_gold(self):
        self.assertEqual(TENSOR_NAME,
            "NRA2_S80_GENUINE43M_GT_NOVEL48_WORD339_NO_GOLD.safetensors")
        self.assertEqual(REPORT_NAME,
            "NRA2_S80_GENUINE43M_GT_NOVEL48_GOLD_BLIND_SOURCE_PROVENANCE.json")


if __name__=="__main__":
    unittest.main()
