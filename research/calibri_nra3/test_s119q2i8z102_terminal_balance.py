import unittest
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z102_terminal_balance import ARMS,TERMINAL_SHARE,weights,objective
class TestZ102(unittest.TestCase):
    def test_weights(self):
        for y in (torch.tensor([7,2,8,2]),torch.tensor([7,8,9,10,2])):
            w=weights(y,ARMS[1]);self.assertAlmostEqual(float(w.sum()),1.,places=6)
            self.assertAlmostEqual(float(w[y==2].sum()),TERMINAL_SHARE,places=6)
            self.assertTrue(bool((w>0).all()))
    def test_full_vocab(self):
        y=torch.tensor([3,2,5,2]);x=torch.zeros(4,49152,requires_grad=True)
        self.assertTrue(torch.equal(objective(x,y,ARMS[0]),F.cross_entropy(x,y)))
        objective(x,y,ARMS[1]).backward()
        self.assertTrue(bool((x.grad[torch.arange(4),y]<0).all()))
        self.assertGreater(float(x.grad[0,2]),0.)
        self.assertGreater(float(x.grad[1,3]),0.)
    def test_no_mask(self):
        y=torch.tensor([2,3]);x=torch.zeros(2,49152)
        before=x.clone();objective(x,y,ARMS[1])
        self.assertTrue(torch.equal(x,before))
        with self.assertRaises(Exception):weights(torch.tensor([2,2]),ARMS[1])
if __name__=="__main__":unittest.main()
