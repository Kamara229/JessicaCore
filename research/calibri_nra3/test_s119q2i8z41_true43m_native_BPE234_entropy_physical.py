"""Z41 physically loaded original S142 43M test *source contract* only.

Actual physical evidence is exclusively GitHub Actions result, not these
lightweight static checks. Source verifies source-only Z40 DEV prereg.
"""
import inspect
import unittest
from research.calibri_nra3 import s119q2i8z41_true43m_causal_native_BPE234_entropy_physical as physical
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as src
from research.calibri_nra3.s119q2i8z40_causal_BPE234_entropy_guard import MODES

class TestZ41Original43MNativeBPE234(unittest.TestCase):
    def test_source_prereg_and_legal_train84_new14(self):
        a=src.verify()
        self.assertEqual((a["previous_legal_TRAIN84"],a["NEW_original_author_internal_DEV14"]),(84,14))
        self.assertTrue(a["zero_past_spent_or_train_prompt_overlap"])
        self.assertEqual(src.POSITIONS,(2,3,4))
        self.assertEqual(src.SHADOW_PARAMS,31680)
        self.assertFalse(a["candidate_accepted"])
        self.assertEqual(len(MODES),2)
    def test_actual_original43M_sha_verified_and_no_DEV_gold_prefix(self):
        c=inspect.getsource(physical.run)
        for key in (
            "load_actual_original(s93,side,s142)",
            "fresh=fresh_new_verify()",
            "for row in fresh_train:",
            "for position in POSITIONS:",
            'row["prefix"]+prefix_gold',
            "gold_prefix_only",
            'len(train_cache)==252',
            "fit_controls(train_cache,native_embedding)",
            "for row in fresh_dev:",
            "for mode in (\"ORIGINAL_NATIVE_GREEDY\",)+MODES:",
            'row["prefix"]+continuation',
            "position=next_index+1",
            "guarded_scores(",
            '"first_three_native_correct":contiguous>=3',
            "only_native_BPE2_BPE3_BPE4_can_be_reranked",
            "old_original_param_digest(model)==original_digest",
            '"candidate_accepted":False,"output_gate":0',
        ):
            with self.subTest(key=key):
                self.assertIn(key,c)
        self.assertLess(c.index("for row in fresh_train:"),c.index("for row in fresh_dev:"))
        self.assertNotIn("optimizer.step()",c)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",c)
    def test_correct_model_capability_without_deployment(self):
        self.assertIn("Z41_REAL43M",physical.STAGE)
        self.assertIn("BPE234",physical.REPORT)
        self.assertEqual(physical.NATIVE_VOCAB,49152)
        self.assertEqual(physical.TOP_K,64)
        self.assertEqual(physical.SHADOW_PARAMS,31680)

if __name__=="__main__":
    unittest.main()
