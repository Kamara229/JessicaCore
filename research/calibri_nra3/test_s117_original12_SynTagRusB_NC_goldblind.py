"""S117 protected original12 SynTagRus-B prospective SOURCE-ONLY NC test ABI."""
import ast
from pathlib import Path
import unittest
from research.calibri_nra3.s117_true_original12_SynTagRusB_NC_sourceonly_53751 import (
    STAGE,SOURCE_SHA,TENSOR,WEIGHTS,REPORT,
)
class TestS117SourceOnly(unittest.TestCase):
    def test_immutable_B_train_source_sha_new_licensed(self):
        self.assertEqual(SOURCE_SHA,"f687b1f677a3795ba4c4dbf3b5246e1105cf19f236f023e4d9c4d18b5418f9bf")
        self.assertEqual(STAGE,"CALIBRI_NRA3_S117_REAL43MGT_ALL12_SYNTAGRUS_TRAINB256_ALLVERBS_SOURCE_ONLY_AFTER616")
        self.assertEqual(TENSOR,"NRA3_S117_SYNTAGRUS_TRAINB256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors")
        self.assertEqual(WEIGHTS,"NRA3_S117_IN_CALIBRI_TRAINED_TRUST_53751_FP16.safetensors")
        self.assertEqual(REPORT,"NRA3_S117_REAL43MGT_SYNTAGRUS_TRAINB256_ALLVERBS_SOURCE_PROVENANCE.json")
    def test_real_previous_frozen_S115_child_in_CALIBRI_gold_never_loaded(self):
        s=Path("research/calibri_nra3/s117_true_original12_SynTagRusB_NC_sourceonly_53751.py").read_text()
        for x in ("NRA3BoundedStructuralTrustInCalibriWeights()",
                  "child.attach_to_RESEARCH_calibri(model)",
                  "child.read_only_all12_source_scope(model)",
                  "old_calibri_parameter_sha(model)",
                  "old_616_J7_physically_replayed_BEFORE_NRA3_hooks",
                  "NRA3_S115_SYNTAGRUS_NC_NEW8_BOUNDED_TRUST_53751_IN_CALIBRI_FP16.safetensors",
                  "NRA3_S115_SYNTAGRUS_NC_NEW8_TRAIN_ONLY_NO_DEV_REPORT.json",
                  "S115_trained_IN_CALIBRI_53751_FP16_child_SHA256",
                  "S115_trained_53751_in_model_child_FP16_SHA256",
                  "source_ONLY_no_S116_SYNTAGRUS_UD_GOLD_loaded_in_original_CPU",
                  "S116_SYNTAGRUS_TRAIN_PINNED_SOURCE256_SHA256",
                  "source_row_indices_in_PINNED_SYNTAGRUS_TRAINB256",
                  "noncommercial_research_only_CC_BY_NC_SA_4_license",
                  "new_SynTagRus_TRAINB_firstread_RAW_ONLY_NO_PERFORMANCE_SCORED"):
            self.assertIn(x,s,x)
        mods=[x.module or '' for x in ast.walk(ast.parse(s)) if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s116_SynTagRusB256" in x or "s115_original12" in x for x in mods))
    def test_wrapper_S117_after_protected616(self):
        s=Path("research/nra3_s20/s66_true43m_gt_new32_goldblind_cpu_and_KV_parity.py").read_text()
        self.assertIn("nra3_s117_capture(",s)
        self.assertIn("nra3_s117_save(",s)
        self.assertIn('os.environ["NRA3_S117_SYNTAGRUS_B_SOURCE_ONLY_JSON"]',s)
        self.assertIn('os.environ["NRA3_S117_S115_CHILD_FP16"]',s)
        self.assertIn('os.environ["NRA3_S117_S115_CHILD_MANIFEST_JSON"]',s)
if __name__=='__main__':
    unittest.main()
