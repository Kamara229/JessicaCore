"""Z22 research-only physical original43M, true OWN-first and learned future native state."""
import inspect
import unittest
from research.calibri_nra3.s119q2i8z21_fresh28train14dev_future_state_source import (
    verify,TRAIN_EPOCHS,LATENT_DIM,TOP_K,
)
from research.calibri_nra3.s119q2i8z21_future_state_lowrank16_matched_contract import STATES
from research.calibri_nra3 import s119q2i8z22_real43m_predictive_future_state_vs_shuffle as physical

class TestZ22TrueOriginalPhysicalShadow(unittest.TestCase):
    def test_01_precommitted_source_new_only(self):
        x=verify()
        self.assertEqual((x["TRAIN_count"],x["DEV_count"]),(28,14))
        self.assertEqual((TRAIN_EPOCHS,LATENT_DIM,TOP_K),(30,16,64))
        self.assertEqual(len(STATES),2)
        self.assertIn("S119Q2I8Z22",physical.STAGE)
        self.assertIn("z10_dir",inspect.signature(physical.run).parameters)
        self.assertIn("s142",inspect.signature(physical.run).parameters)
    def test_02_real_original43m_after_own_first_and_same_last_hidden(self):
        c=inspect.getsource(physical.run)
        self.assertIn("load_actual_original(s93,side,s142)",c)
        self.assertIn("capture_existing_full43m_hidden(model)",c)
        self.assertIn('with scoped_exact_native_gain(',c)
        self.assertIn('real_joint_native(',c)
        self.assertIn("own_first=int(first_logits[0].argmax().item())",c)
        self.assertIn('"original_native_logits":second_logits.detach()',c)
        self.assertIn('"target_future_embedding":',c)
        self.assertIn('target_next3_original_embeddings(native_embedding,gold)',c)
        self.assertIn("old_original_param_digest(model)==original_digest",c)
    def test_03_train_only_human_future_before_blind_dev_once(self):
        c=inspect.getsource(physical.run)
        self.assertLess(c.index("for row in fresh_train:"),
                        c.index("train_matched_aligned_vs_permuted("))
        self.assertLess(c.index("train_matched_aligned_vs_permuted("),
                        c.index("for row in fresh_dev:"))
        self.assertIn("no_DEV_gold_in_state_prediction_or_candidate_generation",c)
        self.assertIn("human_gold_only_POSTHOC_after_selection",c)
        self.assertIn("Z21_DEV14_NOW_SPENT_NOT_independent_blind",c)
        self.assertIn("original43M_A_B_total_real_native49152_forwards",c)
        self.assertNotIn("optimizer.step()",c)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",c)
        self.assertIn('"candidate_accepted":False,"output_gate":0',c)
if __name__=="__main__":
    unittest.main()
