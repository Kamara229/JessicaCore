import unittest
import torch
from research.calibri_nra3.s119q2i8z61_gpt_joint_train import objective_loss,ARMS
class TestZ61(unittest.TestCase):
    def test_early_EOS_gradient_only_nonterminal(self):
        x=torch.zeros(1,8,requires_grad=True)
        x.data[0,2]=6
        gold=[4,5,2]
        a=objective_loss(x,gold,1,arm=ARMS[1],contrast=(1,5,6),vocab=8)
        b=objective_loss(x,gold,1,arm=ARMS[2],contrast=(1,5,6),vocab=8)
        self.assertTrue(torch.equal(a[1],b[1]))
        self.assertTrue(torch.equal(a[2],b[2]))
        self.assertGreater(float(b[3]),0.)
        g=torch.autograd.grad(b[0]-a[0],x)[0]
        self.assertGreater(float(g[0,2]),0.)
        self.assertLess(float(g[0,5]),0.)
        for arm in ARMS:
            v=objective_loss(x,gold,2,arm=arm,contrast=(1,5,6),vocab=8)
            self.assertEqual(float(v[2]),0.)
            self.assertEqual(float(v[3]),0.)
            self.assertTrue(torch.equal(v[0],v[1]))
    def test_stable_extreme_logits(self):
        x=torch.zeros(1,8,requires_grad=True)
        x.data[0,2]=1000.
        v=objective_loss(x,[4,2],0,arm=ARMS[2],vocab=8)
        self.assertTrue(torch.isfinite(v[0]))
        v[0].backward()
        self.assertTrue(torch.isfinite(x.grad).all())
if __name__=="__main__":unittest.main()
