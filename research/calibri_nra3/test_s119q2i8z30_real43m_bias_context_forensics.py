"""Z30 original S142 physical head component bias/lowrank root-cause forensic contract.

Already SPENT Z28 DEV only. Zero new optimizer, no independent benchmark.
"""
import inspect
import unittest
from research.calibri_nra3.s119q2i8z28_precommit_84train14fresh_native_head_multistep import verify,rows
from research.calibri_nra3 import s119q2i8z30_real43m_head_bias_context_ablation as actual

class TestZ30FrozenHeadComponentReadOnly(unittest.TestCase):
    def test_01_previous_Z28_author_synthetic_spent_source_only(self):
        src=verify()
        self.assertEqual(len(rows("TRAIN")),84)
        self.assertEqual(len(rows("DEV")),14)
        self.assertTrue(src["strict_no_previous_SPENT_DEV_in_training"])
        self.assertIn("z29_dir",inspect.signature(actual.run).parameters)
        self.assertIn("S119Q2I8Z30",actual.STAGE)
    def test_02_real_original43m_rank4_head_bias_vs_context(self):
        script=inspect.getsource(actual.run)
        for phrase in (
            "load_actual_original(s93,side,s142)",
            "scoped_exact_native_gain(",
            "real_joint_native(",
            "full.last_forward_coverage==tuple(range(12))",
            "z29_dir/PRIOR_Z29_REPORT",
            "sha(path_fp16)==FROZEN_NEW_HEAD_SHA[arm]",
            '"NEW_Z29_LOW_RANK_OLD_GLOBAL_BIAS"',
            '"NEW_Z29_GLOBAL_BIAS_OLD_LOW_RANK"',
            '"output_bias":original_head["output_bias"]',
            '"output_bias":new_head["output_bias"]',
            "original_child.language_head.state_dict()",
            "copy.deepcopy(original_child)",
            "old_original_param_digest(model)==original_digest",
        ):
            with self.subTest(phrase=phrase):self.assertIn(phrase,script)
    def test_03_no_NEW_train_no_gold_in_generation_parity_fail_closed(self):
        script=inspect.getsource(actual.run)
        self.assertIn('fresh_new_rows("DEV")',script)
        self.assertNotIn('for row in fresh_train:',script)
        self.assertNotIn('optimizer.step()',script)
        self.assertNotIn('fit.step()',script)
        self.assertNotIn("full_native_trainonly_pos_loss(",script)
        self.assertIn("parent_original_and_new_head_predictions_replayed_exactly",script)
        self.assertIn("human",inspect.getsource(actual).lower())
        self.assertIn("NO_new_TRAIN_or_DEV_supervision_or_hyperparameter_selection",script)
        self.assertIn('"candidate_accepted":False,"output_gate":0',script)
if __name__=="__main__":
    unittest.main()
