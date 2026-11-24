import unittest
import torch
from research.calibri_nra3.s119q2i8z69_gradient_interaction import comparison
class TestZ69(unittest.TestCase):
    def test_gradient_conflict(self):
        a=torch.tensor([1.,0.]);b=torch.tensor([-2.,0.])
        r=comparison(a,b)
        self.assertEqual(r["cosine"],-1.)
        self.assertEqual(r["dot"],-2.)
        self.assertEqual(r["aux_norm"],2.)
        self.assertIsNone(comparison(a,torch.zeros_like(a))["cosine"])
        self.assertEqual(comparison(a,a)["cosine"],1.)
if __name__=="__main__":unittest.main()
