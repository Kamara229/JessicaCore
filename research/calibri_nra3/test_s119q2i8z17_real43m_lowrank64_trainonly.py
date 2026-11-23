"""S119q2i8z17 strict model readout and train/DEV split physical wrapper contracts."""
import inspect
import unittest
from collections import Counter
from research.calibri_nra3 import s119q2i8z17_true43m_trainonly_native64_rerank_AB as physical
from research.calibri_nra3.s119q2i8z16_fresh28train14dev_rerank_source import (
    verify, rows, KINDS,
)
from research.calibri_nra3.s119q2i8z16_trainonly_lowrank_native64_reranker import (
    HIDDEN,RANK,K,TRAIN_EPOCHS,exact_fixed_trainonly_reranker,
)

class TestZ17NoLeakage(unittest.TestCase):
    def test_preregistered_new_ru_train_and_dev_count(self):
        x=verify()
        self.assertEqual((x["TRAIN_n"],x["DEV_n"]),(28,14))
        self.assertEqual(Counter(r["kind"] for r in rows("TRAIN")),{k:4 for k in KINDS})
        self.assertEqual(Counter(r["kind"] for r in rows("DEV")),{k:2 for k in KINDS})
        self.assertTrue(x["disjoint_all_prior_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14"])
        self.assertEqual((HIDDEN,RANK,K,TRAIN_EPOCHS),(320,2,64,25))
    def test_only_frozen_z10_original43m_physical_inputs(self):
        self.assertIn("TRUE43M_TRAINONLY_NATIVE64",physical.STAGE)
        params=inspect.signature(physical.run).parameters
        for k in ("s93","side","s142","q2i8w_dir","z10_dir","out"):
            self.assertIn(k,params)
    def test_true_hidden_from_same_actual_original_lm_head(self):
        code=inspect.getsource(physical.run)
        self.assertIn("capture_existing_full43m_hidden(model)",code)
        self.assertIn("real_joint_native(",code)
        self.assertIn("native_embedding=model.get_input_embeddings().weight.detach()",code)
        self.assertIn("scoped_exact_native_gain(",code)
        self.assertIn("old_original_param_digest(model)",code)
    def test_new_human_28_training_ONLY_then_14_dev(self):
        code=inspect.getsource(physical.run)
        self.assertIn('for row in fresh_train:',code)
        self.assertIn('for row in fresh_dev:',code)
        self.assertIn('exact_fixed_trainonly_reranker(',code)
        self.assertIn('len(train_cache)==28',code)
        self.assertIn('len(observed)==14',code)
        self.assertIn('"gold_only_posthoc_after_ranker_selection":True',code)
        self.assertIn('"NO_DEV_gradient_or_gold_in_inference_candidate_generation":True',code)
        self.assertNotIn("optimizer.step()",code)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",code)
    def test_not_any_live_or_acceptance_in_physical_report(self):
        code=inspect.getsource(physical.run)
        self.assertIn('"candidate_accepted":False,"output_gate":0',code)
        self.assertIn("original_full43M_native49152_forwards_A_B_total",code)
        self.assertIn("only_new_1280_rank2_shadow_params_learned_per_arm",code)
if __name__=="__main__":
    unittest.main()
