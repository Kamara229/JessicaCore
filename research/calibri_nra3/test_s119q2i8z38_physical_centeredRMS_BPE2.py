"""Z38: strict physical original S142 43M native BPE2-only on-policy ranker."""
import inspect,unittest
from research.calibri_nra3.s119q2i8z37_new14dev_candidate_centered_rms_source import (
    rows,verify,TOP_K,RANK,EVAL_FIRST_NATIVE_BPE,
)
from research.calibri_nra3.s119q2i8z37_native_BPE2_centered_RMS_reranker import (
    MODES,SHADOW_PARAMS,
)
from research.calibri_nra3 import s119q2i8z38_true43m_centeredRMS_BPE2_only_rank32_physical as physical

class TestZ38TrueOriginalBPE2Only(unittest.TestCase):
    def test_original_new_source_and_precommitted_train_only(self):
        s=verify()
        self.assertEqual((s["prior_legal_TRAIN84"],s["brandNEW_author_synthetic_DEV14"]),(84,14))
        self.assertTrue(s["new_DEV_disjoint_all_prior_spent_including_Z34"])
        self.assertEqual((TOP_K,RANK,SHADOW_PARAMS,EVAL_FIRST_NATIVE_BPE),(64,32,30720,6))
        self.assertEqual(len(rows("TRAIN")),84)
        self.assertEqual(len(rows("DEV")),14)
        self.assertEqual(len(MODES),2)
    def test_real_full49152_original43M_and_ONLY_BPE2_positional_intervention(self):
        c=inspect.getsource(physical.run)
        for s in (
            "load_actual_original(s93,side,s142)",
            "real_joint_native(",
            "capture_existing_full43m_hidden(model)",
            "native_embedding[own_first]",
            "native_bpe_position=2",
            'mode=="ORIGINAL_NATIVE_GREEDY" or next_index!=1',
            "fit_two_trainonly_shadow_controls(",
            "SHADOW_PARAMS==30720",
            "scoped_exact_native_gain(",
            "full.last_forward_coverage==tuple(range(12))",
            "old_original_param_digest(model)==original_digest",
            "strict_ONLY_second_native_BPE_candidate_decoding_intervention",
            "candidate_scores_centered_RMS_before_bounded_tanh",
            "zero_mean_native_candidate_residual_by_construction",
            "Z37_brandNEW_author_DEV14_NOW_SPENT_not_independently_blind",
        ):
            with self.subTest(s=s):self.assertIn(s,c)
    def test_real_train_then_brand_new_DEV_no_gold_teacher_on_DEV(self):
        c=inspect.getsource(physical.run)
        self.assertLess(c.index("for row in fresh_train:"),
                        c.index("for row in fresh_dev:"))
        self.assertIn('fresh_new_rows(split)',c)
        self.assertIn('row["prefix"]+continuation',c)
        self.assertIn("autonomous_first_six_native_ids",c)
        self.assertIn("first_six_native_correct",c)
        self.assertNotIn("optimizer.step()",c)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",c)
        self.assertIn('"candidate_accepted":False,"output_gate":0',c)
if __name__=="__main__":unittest.main()
