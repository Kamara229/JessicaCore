"""Z19 enforces true original43M and matched Z18 TRAIN-only h320 fixed ensemble source."""
import inspect
import unittest
from research.calibri_nra3 import s119q2i8z19_real43m_matched_multilayer_vs_final as physical
from research.calibri_nra3.s119q2i8z19_trainonly_matched_multilayer_rankers import (
    train_matched_shadow_rerankers,HIDDEN_KEYS,MODES,
)
from research.calibri_nra3.s119q2i8z18_fresh28train14dev_multilayer_source import (
    verify,BASELINE,ALTERNATE,
)

class TestZ19PhysicalSourceOnly(unittest.TestCase):
    def test_01_new_disjoint_and_preregistered_layers(self):
        s=verify()
        self.assertEqual((s["TRAIN_count"],s["DEV_count"]),(28,14))
        self.assertEqual(s["multilayer_selected_layer_indices"],(2,6,10))
        self.assertEqual(s["fixed_mix_weights"],(.25,.25,.25,.25))
        self.assertTrue(s["disjoint_old_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16"])
        self.assertEqual(set(MODES),{BASELINE,ALTERNATE})
        self.assertIn("z10_dir",inspect.signature(physical.run).parameters)
    def test_02_equal_real_logits_and_origin_h320_one_physical_pass(self):
        src=inspect.getsource(physical.run)
        self.assertIn("capture_original_selected_layers(model)",src)
        self.assertIn("capture_existing_full43m_hidden(model)",src)
        self.assertIn("fixed_multi_layer_h320(layers,h[0])",src)
        self.assertIn('train_cache.append({',src)
        self.assertIn('"hidden_final":h[0]',src)
        self.assertIn('"hidden_multilayer":mixed',src)
        self.assertIn("torch.equal(ids,top64)",src)
        self.assertIn('witnessed[0]==2',src)
    def test_03_28_train_first_and_14_only_after_fit(self):
        src=inspect.getsource(physical.run)
        self.assertLess(src.index("for row in fresh_train:"),
                        src.index("train_matched_shadow_rerankers("))
        self.assertLess(src.index("train_matched_shadow_rerankers("),
                        src.index("for row in fresh_dev:"))
        self.assertIn("old_original_param_digest(model)",src)
        self.assertNotIn("optimizer.step()",src)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",src)
        self.assertIn('"no_DEV_gradient_or_gold_in_candidate_selection":True',src)
        self.assertIn('"candidate_accepted":False,"output_gate":0',src)
if __name__=="__main__":
    unittest.main()
