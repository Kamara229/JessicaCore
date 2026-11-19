"""S95 original Calibri all12 licensed Taiga TRAIN256 source-only contracts.

Pure syntax/provenance tests, no teacher gold/UD, no actual CPU weights.
Full original protected616 J7 and 12layer CPU CI runs separately.
"""
import ast
import unittest
from pathlib import Path

from research.calibri_nra3.s95_taiga256_genuine43mGT_all12_sourceonly import (
    STAGE,SOURCE_SHA,TENSOR,WEIGHTS,REPORT,serialized_source_sha,
)

S95=Path("research/calibri_nra3/s95_taiga256_genuine43mGT_all12_sourceonly.py")
WRAPPER=Path("research/nra3_s20/s66_true43m_gt_new32_goldblind_cpu_and_KV_parity.py")


class TestS95Actual43MGTAll12SourceOnly(unittest.TestCase):
    def test_licensed_upstream_train_source_SHA_locked_before_frozen_original_cpu(self):
        self.assertEqual(SOURCE_SHA,
            "257fcc7518debb81e779c1419c34fdd9e565ad7e5ed9b04d08e55484fddba917")
        self.assertEqual(
            STAGE,"CALIBRI_NRA3_S95_REAL43M_GT_ALL12_UD_TAIGA_TRAIN_SOURCE_ONLY_AFTER616")
        self.assertEqual(TENSOR,
            "NRA3_S95_REAL43M_GT_TAIGA_TRAIN_NATIVE_ALL12_H320_NO_GOLD.safetensors")
        self.assertEqual(WEIGHTS,
            "NRA3_S95_IN_CALIBRI_UNTRAINED_22311_NEURAL_WEIGHTS_FP16.safetensors")
        self.assertEqual(REPORT,
            "NRA3_S95_REAL43M_GT_TAIGA_SOURCE_ONLY_ALIGNMENT_PROVENANCE.json")
        self.assertEqual(serialized_source_sha([{"row_id":"a","text":"b"}]),
                         serialized_source_sha([{"text":"b","row_id":"a"}]))

    def test_no_licensed_teacher_labels_or_old_spent_TEST_in_original_model_imports(self):
        syntax=ast.parse(S95.read_text(encoding="utf8"))
        mods=[z.module or "" for z in ast.walk(syntax)
              if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("s94_ud_taiga_trainonly" in x
                             or "s94_OFFLINE_UD" in x
                             or "s86_" in x or "s79_" in x
                             or "s93_offline_ud_grammatical" in x
                             for x in mods))
        self.assertTrue(any("s91_neural_multilayer_event_graph" in x
                            for x in mods))
        self.assertTrue(any("s75_original43m" in x for x in mods))

    def test_original12_core_sha_and_word_to_native_BPE_offsets_without_oracle(self):
        t=S95.read_text(encoding="utf8")
        for name in (
            "old_calibri_parameter_sha(model)",
            "source_offset_alignment(tok,text,ids)",
            "raw_model_input(text).tokens",
            "child.attach_to_RESEARCH_calibri(model)",
            "child.read_only_all12_source_scope(model)",
            "layer8.capture(model,text,ids)",
            "torch.equal(true_hidden[0,7],capture.hidden320_fp16)",
            "source_raw_word_char_start",
            "source_raw_word_char_end",
            "source_raw_word_last_native_BPE",
            "source_raw_word_native_BPE_causally_safe",
            "source_row_indices_in_PINNED_TAIGA_TRAIN256",
            "old_616_J7_physically_replayed_BEFORE_NRA3_hooks",
            "no_commercial_or_production_use_before_legal_review",
        ):
            self.assertIn(name,t)

    def test_research_original_J7_wrapper_calls_S95_source_after_old616(self):
        t=WRAPPER.read_text(encoding="utf8")
        self.assertIn("nra3_s95_capture(",t)
        self.assertIn('os.environ["NRA3_S95_TAIGA_SOURCE_ONLY_JSON"]',t)
        self.assertIn("nra3_s95_save(",t)
        self.assertNotIn("nra3_s92_capture(",t)
        self.assertNotIn("nra3_s92_save(",t)


if __name__=="__main__":
    unittest.main()
