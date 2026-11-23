"""Z24 actual S142 original43M read-only previous trained Z22 4 SHA-frozen states."""
import inspect
import unittest
from research.calibri_nra3.s119q2i8z23_new28dev_frozen_future_calibration_source import (
    verify,rows,SCALES,TOP_K,BASE_FUTURE_FP16_SHA,BASE_PHYSICAL_RUN_ID,
)
from research.calibri_nra3.s119q2i8z21_future_state_lowrank16_matched_contract import STATES
from research.calibri_nra3 import s119q2i8z24_true43m_frozen_future_scale_diagnostics as physical

class TestZ24RealOriginalFrozenState(unittest.TestCase):
    def test_preregister_new_28_and_immutable_prev_trained_shadows(self):
        s=verify()
        self.assertEqual(s["new_unseen_synthetic_DEV"],28)
        self.assertEqual((SCALES,TOP_K),( (0.,2.,8.,16.),64))
        self.assertEqual(len(BASE_FUTURE_FP16_SHA),4)
        self.assertEqual(BASE_PHYSICAL_RUN_ID,45221922458)
        self.assertTrue(s["disjoint_all_prior_spent_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16_Z18_Z21"])
        self.assertEqual(len(rows("DEV")),28)
        self.assertIn("z22_dir",inspect.signature(physical.run).parameters)
    def test_locked_physical_real_43m_fullnative_and_shas_in_runner(self):
        c=inspect.getsource(physical.run)
        self.assertIn("load_actual_original(s93,side,s142)",c)
        self.assertIn("z22_report_path=z22_dir/FROZEN_Z22_REPORT",c)
        self.assertIn("sha(fp)==expected_sha",c)
        self.assertIn("load_file(str(fp),device=\"cpu\")",c)
        self.assertIn("model_shadow.requires_grad_(False)",c)
        self.assertIn("capture_existing_full43m_hidden(model)",c)
        self.assertIn("scoped_exact_native_gain(",c)
        self.assertIn("real_joint_native(",c)
        self.assertIn("old_original_param_digest(model)==original_digest",c)
        self.assertIn("all_scales_target_blind_frozen_Z22_state_decisions",c)
    def test_only_28_dev_after_all_decisions_and_no_training(self):
        c=inspect.getsource(physical.run)
        self.assertIn('for row in fresh_dev:',c)
        self.assertIn('fresh_new_rows("DEV")',c)
        self.assertIn("target_blind_calibrated_candidates(",c)
        self.assertLess(c.index("target_blind_calibrated_candidates("),
                        c.index("posthoc_human_gold("))
        self.assertIn("Z23_new_DEV_NOW_SPENT_not_independently_blind",c)
        self.assertIn("zero_training_or_optimizer_steps_on_Z23_DEV",c)
        self.assertIn("no_chosen_best_scale_from_DEV_or_deployable_oracle",c)
        self.assertNotIn("train_matched_aligned_vs_permuted(",c)
        self.assertNotIn("optimizer.step()",c)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",c)
        self.assertIn('"candidate_accepted":False,"output_gate":0',c)
if __name__=="__main__":
    unittest.main()
