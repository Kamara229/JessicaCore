"""Z27 causal source-only test: true base native vs learned head .0/.25/1.
Old DEV only, exact past Z26 first4 parity, no new training.
"""
import inspect
import unittest
from research.calibri_nra3.s119q2i8z25_84historicaltrain_new14dev_hardnegative_source import verify,rows
from research.calibri_nra3 import s119q2i8z27_real43m_native_head_gain_rootcause as physical

class TestZ27RootCauseOriginal43M(unittest.TestCase):
    def test_readonly_previous_spent_author_source(self):
        x=verify()
        self.assertEqual(len(rows("DEV")),14)
        self.assertEqual(x["brand_new_author_synthetic_DEV_count"],14)
        self.assertIn("z26_dir",inspect.signature(physical.run).parameters)
        self.assertIn("S119Q2I8Z27",physical.STAGE)
    def test_protected_weights_and_exact_archived_025_parity(self):
        c=inspect.getsource(physical.run)
        for text in (
            "load_actual_original(s93,side,s142)",
            "first_original_pass(", "real_joint_native(",
            "scoped_exact_native_gain(", "set_gt_route(",
            "old_original_param_digest(model)==original_digest",
            '"FULL_CHILD_EXACT_PREVIOUS_GAIN_025":0.25',
            '"FULL_CHILD_NO_LEARNED_NATIVE_HEAD":0.0',
            '"FULL_CHILD_GAIN_1":1.0',
            '"ORIGINAL_S142_PRETRAINED_NO_NRA3_CHILD":None',
            '"auto_generated_modes_no_teacher_forcing"',
            '"ORIGINAL_NATIVE_GREEDY"',
            '"second_third_fourth_identical_native_BPE"',
        ):
            self.assertIn(text,c)
        self.assertIn("same_source_previous_real_Z26_first4_parity",c)
        self.assertIn("gold_human_first4_only_after_autonomous_model_choices",c)
    def test_no_training_no_new_dev_and_gate0(self):
        c=inspect.getsource(physical.run)
        self.assertNotIn("train_matched_control_on_historical_train(",c)
        self.assertNotIn("optimizer.step()",c)
        self.assertNotIn("for row in fresh_train:",c)
        self.assertIn('fresh_new_rows("DEV")',c)
        self.assertIn("no_new_training_or_hyperparam_selection",c)
        self.assertIn('"candidate_accepted":False,"output_gate":0',c)
if __name__=="__main__":
    unittest.main()
