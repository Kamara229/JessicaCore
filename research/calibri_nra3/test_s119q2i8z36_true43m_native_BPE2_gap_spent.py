"""Z36: physical S142 original43M read-only BPE2 residual/margin gap contract."""
import inspect,unittest
from research.calibri_nra3.s119q2i8z34_fresh14dev_BPE2_only_source import rows,verify
from research.calibri_nra3 import s119q2i8z36_true43m_BPE2_gold_logit_gap_and_ranker_residual as physical

class TestZ36FrozenBPE2NativeLogitGap(unittest.TestCase):
    def test_only_previously_spent_Z34_original_author_DEV14(self):
        p=verify()
        self.assertEqual((p["prior_legal_TRAIN84"],p["brandNEW_author_synthetic_DEV14"]),(84,14))
        self.assertTrue(p["new_DEV_disjoint_all_prior_spent_including_Z31"])
        self.assertIn("z35_dir",inspect.signature(physical.run).parameters)
        self.assertEqual(len(rows("DEV")),14)
        self.assertIn("S119Q2I8Z36",physical.STAGE)
    def test_original_true43M_actual_native49k_margin_residual_on_real_own_first(self):
        c=inspect.getsource(physical.run)
        for s in (
            "load_actual_original(s93,side,s142)",
            "real_joint_native(",
            "capture_existing_full43m_hidden(model)",
            "scoped_exact_native_gain(",
            "full.last_forward_coverage==tuple(range(12))",
            "native_embedding[first]",
            "native_bpe_position=2",
            "pinned_fits",
            'sha(fp)==checkpoint["sha256"]==pinned_fits[name][mode]',
            'prev_z35_path=z35_dir/',
            'original_greedy2=int(native.argmax())',
            'gold_inside_original_top64',
            'gap>3.0',
            'net>gap',
            'learned_id==prior_selected',
            'int(logits[0].argmax())==baseline_native[1]',
            'previously_SPENT_DEV14_not_training_or_validation',
            'no_training_and_no_new_independent_DEV',
            'old_original_param_digest(model)==original_digest',
        ):
            with self.subTest(s=s):self.assertIn(s,c)
    def test_no_new_dev_no_training_no_ranker_selection_from_spent_labels(self):
        c=inspect.getsource(physical.run)
        self.assertIn('fresh_new_rows(split)',c)
        self.assertIn('for row in fresh_dev:',c)
        self.assertNotIn('for row in fresh_train:\n                    with scoped_exact_native_gain',c)
        self.assertNotIn("optimizer.step()",c)
        self.assertNotIn("fit_two_trainonly_shadow_controls(",c)
        self.assertIn('"candidate_accepted":False,"output_gate":0',c)
if __name__=="__main__":unittest.main()
