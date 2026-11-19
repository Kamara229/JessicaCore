"""S114 source-only original true Calibri+GT after616, 53751 neural trust8 ABI."""
import ast
from pathlib import Path
import unittest
from research.calibri_nra3.s114_original12_SynTagRusA_53751_NC_sourceonly_CPU import (
    STAGE,SOURCE_SHA,TENSOR,WEIGHTS,REPORT,
)
class TestS114Original12SynTagRusNC(unittest.TestCase):
    def test_new_source_SHA_and_only_noncommercial_golden_separation(self):
        self.assertEqual(STAGE,"CALIBRI_NRA3_S114_REAL43MGT_ALL12_SYNTAGRUS_TRAIN256_ALLVERBS_SOURCE_ONLY_AFTER616")
        self.assertEqual(SOURCE_SHA,"e9972e4f2204887cd1a507289dc0e4936f18c9f8ec9e18a0a2630e1009a1f209")
        self.assertEqual(TENSOR,"NRA3_S114_SYNTAGRUS_TRAIN256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors")
        self.assertEqual(WEIGHTS,"NRA3_S114_IN_CALIBRI_ZERO_TRUST_53751_FP16.safetensors")
        self.assertEqual(REPORT,"NRA3_S114_REAL43MGT_SYNTAGRUS_TRAIN256_ALLVERBS_SOURCE_PROVENANCE.json")
    def test_physical_S112_53751_embedded_original12_source_only(self):
        s=Path("research/calibri_nra3/s114_original12_SynTagRusA_53751_NC_sourceonly_CPU.py").read_text()
        for z in (
            "NRA3BoundedStructuralTrustInCalibriWeights()",
            "child.attach_to_RESEARCH_calibri(model)",
            "child.read_only_all12_source_scope(model)",
            "old_calibri_parameter_sha(model)",
            "old_616_J7_physically_replayed_BEFORE_NRA3_hooks",
            "NRA3_S112_ORIGINAL12_BOUNDED_53751_IN_CALIBRI_ZERO_TRUST_FP16.safetensors",
            "new_S112_53751_8trust_in_calibri_FP16_SHA256",
            "S112_pretrained_53751_in_model_child_FP16_SHA256",
            "pretrained_child_manifest_path",
            "source_ONLY_no_S113_SYNTAGRUS_UD_GOLD_loaded_in_original_CPU",
            "S113_SYNTAGRUS_TRAIN_PINNED_SOURCE256_SHA256",
            "noncommercial_research_only_CC_BY_NC_SA_4_license",
            "source_row_indices_in_PINNED_SYNTAGRUS_TRAIN256",
            "original_S5_H8_port_exactly_matches_all12_H8_for_all_accepted",
        ):
            self.assertIn(z,s,z)
        imported=[z.module or '' for z in ast.walk(ast.parse(s)) if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("s113_new_SynTagRus" in x or "train_c" in x.lower()
                             or "s110_true12_GSD_16656" in x for x in imported))
    def test_wrapper_has_genuine_original616_then_s114(self):
        s=Path("research/nra3_s20/s66_true43m_gt_new32_goldblind_cpu_and_KV_parity.py").read_text()
        self.assertIn("nra3_s114_capture(",s)
        self.assertIn("nra3_s114_save(",s)
        self.assertIn('os.environ["NRA3_S114_SYNTAGRUS_SOURCE_ONLY_JSON"]',s)
        self.assertIn('os.environ["NRA3_S114_S112_CHILD_FP16"]',s)
        self.assertIn('os.environ["NRA3_S114_S112_CHILD_MANIFEST_JSON"]',s)
        self.assertNotIn("nra3_s109_capture(",s)
if __name__=="__main__":
    unittest.main()
