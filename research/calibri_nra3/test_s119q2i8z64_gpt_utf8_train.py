import unittest
import torch
from research.calibri_nra3.s119q2i8z64_gpt_utf8_train import objective_loss,ARMS
class TestZ64(unittest.TestCase):
    def test_added_utf8_gradient(self):
        x=torch.zeros(1,8,requires_grad=True)
        m=torch.tensor([False,False,False,False,True,True,False,False])
        v=objective_loss(x,[4,5,2],1,arm=ARMS[0],contrast=(1,5,6),mask=m,vocab=8)
        self.assertGreater(float(v[4]),0.)
        self.assertTrue(torch.allclose(v[0],sum(v[1:])))
        g=torch.autograd.grad(v[4],x)[0]
        self.assertLess(float(g[0,5]),0.)
        self.assertGreater(float(g[0,6]),0.)
    def test_terminal_eos_ce_not_removed(self):
        x=torch.zeros(1,8,requires_grad=True)
        m=torch.tensor([True,False,True,False,True,True,True,True])
        v=objective_loss(x,[4,5,2],2,arm=ARMS[0],contrast=(1,5,6),mask=m,vocab=8)
        self.assertEqual(float(v[2]),0.)
        self.assertEqual(float(v[3]),0.)
        self.assertGreater(float(v[1]),0.)
        self.assertGreater(float(v[4]),0.)
if __name__=="__main__":unittest.main()
