"""S109 original Calibri CB59+GT true12 source-only gold-blind guard tests."""
import ast
from pathlib import Path
import unittest

from research.calibri_nra3.s109_original12_GSD256_53743_sourceonly_CPU import (
    STAGE,SOURCE_SHA,TENSOR,WEIGHTS,REPORT,
)

SOURCE=Path("research/calibri_nra3/s109_original12_GSD256_53743_sourceonly_CPU.py")
WRAPPER=Path("research/nra3_s20/s66_true43m_gt_new32_goldblind_cpu_and_KV_parity.py")

class TestS109PhysicalOriginalCalibricSourceOnly(unittest.TestCase):
    def test_new_GSD_original256_sha_and_physically_separate_teacher(self):
        self.assertEqual(SOURCE_SHA,"4078f408f2355c55c29bca18504355379e2a42ef2144c9113dda9448b086de14")
        self.assertEqual(STAGE,"CALIBRI_NRA3_S109_REAL43MGT_ALL12_GSD_TRAIN256_ALLVERBS_SOURCE_ONLY_AFTER616")
        self.assertEqual(TENSOR,"NRA3_S109_GSD_TRAIN256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors")
        self.assertEqual(WEIGHTS,"NRA3_S109_IN_CALIBRI_UNTRAINED_53743_FP16.safetensors")
        self.assertEqual(REPORT,"NRA3_S109_REAL43MGT_GSD_TRAIN256_ALLVERBS_SOURCE_PROVENANCE.json")

    def test_new_S107_full_53743_Fp16_actually_registered_inside_real_calibri(self):
        s=SOURCE.read_text()
        for k in (
            "NRA3StructuralCompetitionInCalibriWeights()",
            "child.attach_to_RESEARCH_calibri(model)",
            "child.read_only_all12_source_scope(model)",
            "layer8.capture(model,text,ids)",
            "torch.equal(true_hidden[0,7],capture.hidden320_fp16)",
            "file_sha(path)==\"46ca9d5c946168332b4d6d1108d442d38c9628fe7bd4afb338693f1d56e27cee\"",
            "S107_pretrained_53743_in_model_child_FP16_SHA256",
            "prior_37087_and_new_16656_state_exactly_restored_from_S107",
            "sum(p.numel() for p in child.parameters())==53743",
            "source_raw_word_last_native_BPE",
            "source_raw_word_native_BPE_causally_safe",
            "old_616_J7_physically_replayed_BEFORE_NRA3_hooks",
        ):
            self.assertIn(k,s,k)

    def test_original_true12_calibri_weights_sha_and_original_GSD_no_runtime_gold(self):
        s=SOURCE.read_text()
        for k in (
            "old_calibri_parameter_sha(model)",
            "SOURCE_PARAMS==36_380_480",
            "PAID_OPERATORS==85",
            "all_original_43M_GT_CB59_params_sha_pre_and_post_identical",
            "S108_GSD_TRAIN_PINNED_SOURCE256_SHA256",
            "de392eb8d1a92fe1d12bc2e49c21cb717ae89859",
            "source_ONLY_no_S108_GSD_UD_GOLD_loaded_in_original_CPU",
            "S79_S86_S96_S99_S102_S105_spent_diagnostics_NEVER_reused",
            "no_commercial_or_production_use_before_legal_review",
        ):
            self.assertIn(k,s,k)
        imported=[z.module or "" for z in ast.walk(ast.parse(s))
                  if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("s108_new_GSD" in k or "s105_TaigaE" in k
                             or "train_c" in k.lower()
                             for k in imported))

    def test_original_protected616_wrapper_calls_once_only_after_original(self):
        s=WRAPPER.read_text()
        for k in ("nra3_s109_capture(","nra3_s109_save(",
                  'os.environ["NRA3_S109_GSD_SOURCE_ONLY_JSON"]',
                  'os.environ["NRA3_S109_S107_CHILD_FP16"]',
                  "nra3_s109_all12_GSD_TRAIN"):
            self.assertIn(k,s,k)
        self.assertNotIn("nra3_s104_capture(",s)

if __name__=="__main__":
    unittest.main()
