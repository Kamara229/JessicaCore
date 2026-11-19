"""S98 gold-blind REAL original Calibri+GT 12-layer Taiga TRAIN-b source contracts.

No morphology gold or original frozen model loaded in these tests.
Original protected616 J7 replay and in-model S97 morphology physical
capture are separate REAL GitHub Actions CPU steps.
"""
import ast
import unittest
from pathlib import Path

from research.calibri_nra3.s98_newTaigaB256_true12_morph_inmodel_goldblind_CPU import (
    STAGE,SOURCE_SHA,TENSOR,WEIGHTS,REPORT,
)

CAPTURE=Path("research/calibri_nra3/s98_newTaigaB256_true12_morph_inmodel_goldblind_CPU.py")
WRAPPER=Path("research/nra3_s20/s66_true43m_gt_new32_goldblind_cpu_and_KV_parity.py")


class TestS98SourceOnlyTrueCalibricModel(unittest.TestCase):
    def test_new_Taiga_B256_source_precommitted_before_original_CPU(self):
        self.assertEqual(SOURCE_SHA,
            "437b6d62564e4cda6c58975910ccc98221c3e81938bda22429418f16700b1dad")
        self.assertEqual(STAGE,
            "CALIBRI_NRA3_S98_REAL43MGT_ALL12_TAIGA_TRAINB256_MORPH_SOURCE_ONLY_AFTER616")
        self.assertEqual(TENSOR,
            "NRA3_S98_TAIGA_TRAINB256_REAL12_ORIGINAL_H320_NO_UD_GOLD.safetensors")
        self.assertEqual(WEIGHTS,
            "NRA3_S98_IN_CALIBRI_UNTRAINED_MORPH_28887_FP16.safetensors")
        self.assertEqual(REPORT,
            "NRA3_S98_REAL43MGT_TAIGA_TRAINB256_ALL12_SOURCEONLY_PROVENANCE.json")

    def test_same_actual_12layer_calibri_new_morphology_IN_model_child(self):
        s=CAPTURE.read_text(encoding="utf8")
        for marker in (
            "NRA3MorphologyInCalibriWeights()",
            "child.attach_to_RESEARCH_calibri(model)",
            "child.read_only_all12_source_scope(model)",
            "layer8.capture(model,text,ids)",
            "torch.equal(true_hidden[0,7],capture.hidden320_fp16)",
            "graph.upos_logits.shape==(1,len(ids),17)",
            "graph.case_logits.shape==(1,len(ids),9)",
            "graph.number_logits.shape==(1,len(ids),3)",
            "graph.voice_logits.shape==(1,len(ids),3)",
            "source_raw_word_last_native_BPE",
            "source_raw_word_native_BPE_causally_safe",
            "old_616_J7_physically_replayed_BEFORE_NRA3_hooks",
        ):
            self.assertIn(marker,s)

    def test_original_calibri_old_weights_physical_SHA_and_new_LICENCE(self):
        s=CAPTURE.read_text(encoding="utf8")
        for marker in (
            "old_calibri_parameter_sha(model)",
            "SOURCE_PARAMS==36_380_480",
            "PAID_OPERATORS==85",
            "all_original_43M_GT_CB59_params_sha_pre_and_post_identical",
            "S97_TAIGA_B_PINNED",
            "3b0f6947bd0d6ac022d91d5fb17269450a876bb3",
            "no_commercial_or_production_use_before_legal_review",
            "source_only_no_Taiga_TRAIN_B_UD_morph_or_role_GOLD_loaded_during_CPU",
        ):
            self.assertIn(marker,s)

    def test_old_spent_S96_dev_and_S79_S86_test_excluded_from_original_CPU(self):
        astree=ast.parse(CAPTURE.read_text(encoding="utf8"))
        imported=[x.module or "" for x in ast.walk(astree)
                  if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s94_ud_taiga_trainonly" in v.lower()
                             or "s96_taiga" in v.lower()
                             or "s97_pin_Taiga" in v
                             or "s79_" in v.lower()
                             or "s86_" in v.lower()
                             for v in imported))
        self.assertTrue(any("s97_in_calibri_morphological" in v for v in imported))
        wrapper=WRAPPER.read_text(encoding="utf8")
        self.assertIn("nra3_s98_capture(",wrapper)
        self.assertIn('os.environ["NRA3_S98_TAIGA_B_SOURCE_ONLY_JSON"]',wrapper)
        self.assertIn("nra3_s98_save(",wrapper)
        self.assertNotIn("nra3_s95_capture(",wrapper)


if __name__=="__main__":
    unittest.main()
