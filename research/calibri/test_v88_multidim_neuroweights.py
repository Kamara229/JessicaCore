"""Four pure original tensor arithmetic, source budget, and data gates."""
import unittest
import torch
from v88.tensor import delta,normalized,FACTOR_ELEMENTS,TRAINABLE_ELEMENTS,RANK
from v88.physical import COMPLETE_BPW,COMPLETE_BYTES,HEADER,PAYLOAD_BYTES
from v88.sources import REGIONS
class TestGenuineFourAxis(unittest.TestCase):
    def test_true_4d_operator_matches_full_materialized_small_weight(self):
        torch.manual_seed(88)
        a=torch.randn(2,2,3)
        b=torch.randn(3,4,2)
        x=torch.randn(5,4)
        y=delta(x,a,b)
        dense=torch.einsum("air,rbj->abij",a,b).reshape(8,4)
        self.assertEqual(tuple(y.shape),(5,8))
        self.assertTrue(torch.allclose(y,x@dense.T,atol=1e-5,rtol=1e-5))
    def test_four_axis_parameters_physically_accounted(self):
        self.assertEqual(FACTOR_ELEMENTS,43008)
        self.assertEqual(TRAINABLE_ELEMENTS,92160)
        self.assertEqual(PAYLOAD_BYTES,184320)
        self.assertEqual(COMPLETE_BYTES,11229303+HEADER.size+184320+36)
        self.assertLess(COMPLETE_BPW,0.7)
        self.assertGreater(COMPLETE_BPW,0.667)
    def test_only_new_original_train_regions_not_test(self):
        self.assertEqual(set(REGIONS),{"wiki_fit","news_fit","wiki_valid","news_valid"})
        self.assertNotEqual(REGIONS["wiki_fit"],REGIONS["wiki_valid"])
        self.assertNotEqual(REGIONS["news_fit"],REGIONS["news_valid"])
    def test_normalization_and_invalid_shapes(self):
        x=torch.tensor([[0.,0.,0.,0.]])
        self.assertTrue(torch.isfinite(normalized(x)).all())
        with self.assertRaises(AssertionError):
            delta(x,torch.zeros(2,3,1),torch.zeros(1,2,2))
if __name__=="__main__":
    unittest.main()
