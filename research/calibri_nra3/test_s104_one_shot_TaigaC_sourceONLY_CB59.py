"""S104 prospective ONE-SHOT Taiga TRAIN-c original Calibri all12 source-only ABI.

Do NOT import Taiga TRAIN-c UD gold, original S99/S102 internal
diagnostics or any frozen candidate weights in source-side real43M
model forward. Only the SHA-frozen original prospectively precommitted
S103 source rows are allowed before physical old616 J7 replay.
"""
import ast
import unittest
from pathlib import Path

from research.calibri_nra3.s104_original43mGT_all12_TaigaC_prospective_SOURCE_ONLY import (
    STAGE,SOURCE_SHA,TENSOR,WEIGHTS,REPORT,
)
CAP=Path("research/calibri_nra3/s104_original43mGT_all12_TaigaC_prospective_SOURCE_ONLY.py")
ORIGINAL=Path("research/nra3_s20/s66_true43m_gt_new32_goldblind_cpu_and_KV_parity.py")


class TestS104OnceProspectiveFrozenSource(unittest.TestCase):
    def test_firstread256_SHA_precommitted_BEFORE_original_model_CPU(self):
        self.assertEqual(STAGE,"CALIBRI_NRA3_S104_TRUE43MGT_ALL12_TAIGA_TRAINC256_ONESHOT_SOURCE_ONLY_AFTER616")
        self.assertEqual(SOURCE_SHA,"7c960fd28b14449cfb9fa3611576aa21a88dbd6afccbbde4e2a02d015707bcdb")
        self.assertEqual(TENSOR,"NRA3_S104_TAIGA_TRAINC256_FIRSTREAD_ALL12_ORIGINAL_H320_NO_GOLD.safetensors")
        self.assertEqual(WEIGHTS,"NRA3_S104_IN_CALIBRI_UNTRAINED_28887_FP16.safetensors")
        self.assertEqual(REPORT,"NRA3_S104_REAL43MGT_TAIGA_TRAINC256_ONESHOT_SOURCE_ONLY_PROVENANCE.json")

    def test_no_prospective_teacher_labels_loaded_or_trainable_in_real_original_cpu(self):
        s=CAP.read_text()
        self.assertNotIn("S103_precommit_ONE_SHOT",s)
        tree=ast.parse(s)
        imported=[x.module or "" for x in ast.walk(tree)
                  if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s103_precommit" in x or
                             "s102_TaigaD_real12" in x or
                             "s102_real12_Taiga" in x or
                             "s99_REAL12" in x or
                             "s79_" in x or "s86_" in x for x in imported))
        self.assertIn('source_ONLY_no_S103_PROSPECTIVE_C_UD_gold_loaded_in_original_CPU',s)
        self.assertIn('"prospective_one_shot_source_never_training_and_no_candidate_fit":True',s)

    def test_exact_all12_original_CB59_GT_H8_independent_port_and_28887_in_calibri(self):
        s=CAP.read_text()
        for marker in (
            "SOURCE_PARAMS==36_380_480",
            "PAID_OPERATORS==85",
            "old_calibri_parameter_sha(model)",
            "NRA3MorphologyInCalibriWeights()",
            "child.attach_to_RESEARCH_calibri(model)",
            "child.read_only_all12_source_scope(model)",
            "layer8.capture(model,text,ids)",
            "torch.equal(true_hidden[0,7],capture.hidden320_fp16)",
            "source_offset_alignment(tok,text,ids)",
            "source_raw_word_char_start",
            "source_raw_word_last_native_BPE",
            "source_raw_word_native_BPE_causally_safe",
            "old_616_J7_physically_replayed_BEFORE_NRA3_hooks",
            "IN_CALIBRI_28887_MORPH_FP16_WEIGHTS_SHA256",
        ):
            self.assertIn(marker,s)

    def test_prospective_model_shas_frozen_before_ANY_new_SOURCE_FORWARD(self):
        s=CAP.read_text()
        self.assertIn('"frozen_S99_child_sha_precommitted":"c77a6964ae4e02cb3aac98d69978a25cec2b491a9624cc9668b3b307699bff8a"',s)
        self.assertIn('"frozen_S102_child_sha_precommitted":"f40dddbc03282a6c45c4f807746275d76a4ba776aa613247e73fbdf7f43329df"',s)
        self.assertIn('S79_S86_S96_S99_S102_dev_spent_NEVER_reused',s)
        self.assertIn('ONLY_UD_TAIGA_TRAIN_C_PROSPECTIVE_NOT_TRAIN_a_b_d_or_dev_test',s)
        self.assertIn('output_gate":0',s)

    def test_original_protected_616_wrapper_references_only_S104_Taiga_C_SOURCE(self):
        s=ORIGINAL.read_text()
        self.assertIn('nra3_s104_capture(',s)
        self.assertIn('nra3_s104_save(',s)
        self.assertIn('os.environ["NRA3_S104_TAIGA_C_ONESHOT_SOURCE_ONLY_JSON"]',s)
        self.assertNotIn('nra3_s101_capture(',s)
        self.assertNotIn('nra3_s101_save(',s)


if __name__=="__main__":
    unittest.main()
