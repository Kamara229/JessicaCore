"""Z33 PHYSICAL original S142 43M teacher gold vs own prior context read-only diagnosis."""
import inspect,unittest
from research.calibri_nra3.s119q2i8z31_84legaltrain_new14dev_bounded_context_source import rows,verify
from research.calibri_nra3 import s119q2i8z33_true43m_teacher_vs_own_prefix_rank_diagnostic as runner

class TestZ33TeacherOwnReal43MReadOnly(unittest.TestCase):
    def test_only_previous_SPENT_original_author_Z31(self):
        source=verify()
        self.assertEqual((source["prior_legal_TRAIN84"],source["brandNEW_author_synthetic_DEV14"]),(84,14))
        self.assertEqual(len(rows("DEV")),14)
        self.assertEqual(len(rows("TRAIN")),84)
        self.assertIn("z32_dir",inspect.signature(runner.run).parameters)
        self.assertIn("S119Q2I8Z33",runner.STAGE)
    def test_physical_same_model_fullnative49152_rank_and_parent_parity(self):
        c=inspect.getsource(runner.run)
        for term in (
            "load_actual_original(s93,side,s142)",
            "real_joint_native(",
            "scoped_exact_native_gain(",
            "bounded_frozen_original_native_delta(",
            "full.last_forward_coverage==tuple(range(12))",
            'sha(fp)==saved["sha256"]==frozen_fp16_sha[arm]',
            '"NEW_DEV14_own_native_autoregressive_source"',
            '"autonomous_first6_native_BPE_ids"',
            'z32_dir/PREV_Z32_REPORT',
            '"ON_POLICY_Z32_OWN_GREEDY_PREFIX"',
            '"TEACHER_HUMAN_CORRECT_PREFIX"',
            "native_gold_BPE_counterfactual_rank",
            "human_next_gold_native_token_POSTHOC",
            "old_original_param_digest(model)==original_digest",
        ):
            with self.subTest(term=term):self.assertIn(term,c)
    def test_no_new_train_never_optimise_on_consumed_dev_gold(self):
        c=inspect.getsource(runner.run)
        self.assertIn('fresh_new_rows("DEV")',c)
        self.assertNotIn('fresh_new_rows("TRAIN")',c)
        self.assertNotIn("optimizer.step()",c)
        self.assertNotIn("fit.step()",c)
        self.assertNotIn("full_native_trainonly_pos_loss(",c)
        self.assertIn("teacher_after_own_wrong_prefix_human_gold_is_counterfactual_NOT_valid_supervised_target",c)
        self.assertIn('"candidate_accepted":False,"output_gate":0',c)
if __name__=="__main__":unittest.main()
