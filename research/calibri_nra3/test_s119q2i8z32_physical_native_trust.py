"""Z32 genuine original S142 43M/GT12 frozen Z10 native trust repair."""
import inspect
import unittest
from research.calibri_nra3.s119q2i8z31_84legaltrain_new14dev_bounded_context_source import (
    rows,verify,NATIVE_REFERENCE_KL,CONTEXT_PREGAIN_RESIDUAL_BOUND
)
from research.calibri_nra3 import s119q2i8z32_real43m_bounded_native_contextual_train_first6 as runner

class TestZ32PhysicalNativeContextTrust(unittest.TestCase):
    def test_01_new_holdout_not_any_previous_spent(self):
        s=verify()
        self.assertEqual((s["prior_legal_TRAIN84"],s["brandNEW_author_synthetic_DEV14"]),(84,14))
        self.assertTrue(s["new_DEV_disjoint_all_prior_spent_including_Z28"])
        self.assertEqual(len(rows("TRAIN")),84)
        self.assertEqual(len(rows("DEV")),14)
        self.assertEqual((NATIVE_REFERENCE_KL,CONTEXT_PREGAIN_RESIDUAL_BOUND),(1.0,2.0))
        self.assertIn("z10_dir",inspect.signature(runner.run).parameters)
        self.assertIn("S119Q2I8Z32",runner.STAGE)
    def test_02_train_only_downup_frozen_bias_and_same_paid_S142_nativeKL(self):
        c=inspect.getsource(runner.run)
        for s in (
            "load_actual_original(s93,side,s142)",
            "real_joint_native(",
            "scoped_exact_native_gain(",
            "full.last_forward_coverage==tuple(range(12))",
            "frozen_new_head_params(",
            "learn_params=frozen_new_head_params(",
            "bounded_frozen_original_native_delta(",
            "trust_kl_after_same_paid_forward(",
            "new_head_child.language_head.output_bias.grad is None",
            "sum(x.numel() for x in learn_params)==197888",
            "only_197888_contextual_down_up_params_updated_per_arm",
            "old_original_param_digest(model)==original_digest",
            "previous_A_B_fitted_native_head_and_all_Core_NRA3_NPA1_12GT_unchanged",
        ):
            with self.subTest(s=s): self.assertIn(s,c)
        self.assertIn("new_original_S142_model_forwards_for_reference",c)
        self.assertIn("reference_kl.append",c)
    def test_03_teacher_only_legal_TRAIN_and_autonomous_OWN_new14(self):
        c=inspect.getsource(runner.run)
        self.assertLess(c.index("for row in fresh_train:"),
                        c.index("for row in fresh_dev:"))
        self.assertIn('fresh_new_rows(split)',c)
        self.assertIn("history=row[\"prefix\"]+gold[:idx]",c)
        self.assertIn("row[\"prefix\"]+own",c)
        self.assertIn('"NEW_CONTEXT_LOWRANK_BOUND2_KL1_BIAS_FROZEN"',c)
        self.assertIn("autonomous_first6_native_BPE_ids",c)
        self.assertIn('"candidate_accepted":False,"output_gate":0',c)
        self.assertNotIn("fit.step()\n                    new_head_child.language_head.output_bias",c)
if __name__=="__main__":
    unittest.main()
