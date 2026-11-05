"""v172 source-conditioned original operator repair must not spend a bit."""
import unittest
import numpy as np
import torch
from direct_codebook_v59 import encode,CB59
from v172.protocol import check,ARMS,CASE_IDS
from v172.codec import activation_fit_one,weighted_nearest

class WeightedNativeCodec(unittest.TestCase):
    def test_exact_budget_predeclared(self):
        self.assertTrue(check())
        self.assertEqual(len(CASE_IDS),40)
        self.assertEqual(ARMS,("original_FP32","original_CB59","activation_aware_CB172"))

    def test_real_8D_input_energy_changes_same_physical_index(self):
        # Equal Euclidean distances, but original input feature 0 is
        # 100x more frequently active: prioritize preserving that output.
        c=np.zeros((32,8),dtype=np.float32)
        c[1,0]=1
        c[2,1]=1
        c[3:]=100
        x=np.zeros((1,8),dtype=np.float32)
        x[0,0]=0.7
        x[0,1]=0.8
        w1=np.ones((1,8),dtype=np.float32)
        w2=np.ones((1,8),dtype=np.float32)
        w2[0,0]=100
        plain=weighted_nearest(x,c,w1)
        weighted=weighted_nearest(x,c,w2)
        self.assertEqual(int(plain[0]),2)
        self.assertEqual(int(weighted[0]),1)

    def test_synthetic_pretrained_operator_exact_same_codebook_bytes(self):
        torch.manual_seed(172)
        original=torch.randn(16,16,dtype=torch.float32)
        raw,_,_=encode({"layer.weight":original})
        handle=CB59(raw).handles["layer.weight"]
        act=np.array([1,2,1,3,4,1,1,1,1,1,2,1,1,1,1,1],dtype=np.float32)
        table,codes,stats=activation_fit_one(original.numpy(),handle,act)
        self.assertEqual(len(codes),len(handle.packed))
        self.assertEqual(len(table),32*8*2)
        self.assertLessEqual(stats["FIT_activation_weighted_vector_cost_after"],
                             stats["FIT_activation_weighted_vector_cost_before"]+1e-4)
        self.assertTrue(stats["exact_original_5bit_storage"])

    def test_unlabeled_operator_fit_is_not_source_QA_scorer(self):
        from v172.activations import collect_original_linear_input_energy
        import inspect
        source=inspect.getsource(collect_original_linear_input_energy)
        self.assertNotIn("correct_answer",source)
        self.assertNotIn("gold_label",source)
        self.assertIn("register_forward_pre_hook",source)

if __name__=="__main__":
    unittest.main()
