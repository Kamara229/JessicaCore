"""Z29 TRUE S142 43M full native head-only training, autonomous new DEV14 first6."""
import inspect
import unittest
from research.calibri_nra3.s119q2i8z28_precommit_84train14fresh_native_head_multistep import (
    verify,rows,HEAD_LR,HEAD_GAIN,TRAIN_EPOCHS,EVAL_NATIVE_FIRST_K,
)
from research.calibri_nra3.s119q2i8z28_fullnative_trainonly_multitoken_head_loss import HEAD_TOTAL_PARAMS
from research.calibri_nra3 import s119q2i8z29_real43m_multitoken_headonly_train_eval as physical

class TestZ29PhysicalHeadOnlyBPE1To4EOS(unittest.TestCase):
    def test_strict_new_source_and_true_original_s142_parameters(self):
        s=verify()
        self.assertEqual((s["prior_legal_TRAIN_count"],s["NEW_author_synthetic_DEV_count"]),(84,14))
        self.assertEqual(len(rows("TRAIN")),84)
        self.assertEqual(len(rows("DEV")),14)
        self.assertEqual((HEAD_TOTAL_PARAMS,HEAD_GAIN,HEAD_LR,TRAIN_EPOCHS,EVAL_NATIVE_FIRST_K),
                         (247040,.25,.006,2,6))
        self.assertIn("z10_dir",inspect.signature(physical.run).parameters)
        self.assertEqual(physical.NATIVE_VOCAB,49152)
        self.assertIn("S119Q2I8Z29",physical.STAGE)
    def test_real_S142_original_12GT_and_head_copy_only(self):
        c=inspect.getsource(physical.run)
        for needle in (
            "load_actual_original(s93,side,s142)",
            "copy.deepcopy(original_child)",
            "new_head_child.child.parameters()",
            "new_head_child.language_head.parameters()",
            "torch.optim.AdamW(",
            "new_head_child.language_head.parameters():",
            "p.requires_grad_(True)",
            "new_head_child.child.zero_grad(set_to_none=True)",
            "full49152 true CE must reach EVERY copied rank4 native head tensor",
            "full.last_forward_coverage==tuple(range(12))",
            "scoped_exact_native_gain(",
            "real_joint_native(",
            "select_train_positions(gold)",
            "full_native_trainonly_pos_loss(",
            "old_original_param_digest(model)==original_digest",
            "previous_A_B_fitted_native_head_and_all_Core_NRA3_NPA1_12GT_unchanged",
        ):
            self.assertIn(needle,c)
    def test_train_only_human_native_targets_and_dev_no_teacher_forcing(self):
        c=inspect.getsource(physical.run)
        self.assertLess(c.index("for row in fresh_train:"),
                        c.index("for row in fresh_dev:"))
        self.assertIn('new_head_child,history,',c)
        self.assertIn("row[\"prefix\"]+own",c)
        self.assertIn('"correct_first6_native":correct>=6',c)
        self.assertIn('"BPE2_BPE3_BPE4_identical"',c)
        self.assertIn("new_Z28_14DEV_NOW_SPENT_internal_not_independently_blind",c)
        self.assertIn("zero_DEV_teacher_forcing_or_target_in_output_selection",c)
        self.assertIn('"candidate_accepted":False,"output_gate":0',c)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",c)
if __name__=="__main__":
    unittest.main()
