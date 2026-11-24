import unittest
import torch
from research.calibri_nra3.s119q2i8z107_context_discrimination import pair_penalty,short_objective,ARMS
class TestContext(unittest.TestCase):
    def test_direction(self):
        x=torch.zeros(204,49152)
        p=torch.tensor([[0,1,3,4]])
        base=pair_penalty(x,p)
        x[0,3]=2;x[1,4]=2
        self.assertLess(pair_penalty(x,p),base)
        x[0,4]=4;x[1,3]=4
        self.assertGreater(pair_penalty(x,p),base)
    def test_gradient(self):
        x=torch.zeros(204,49152,requires_grad=True)
        p=torch.tensor([[0,1,3,4]])
        pair_penalty(x,p).backward()
        self.assertLess(x.grad[0,3],0)
        self.assertGreater(x.grad[0,4],0)
        self.assertLess(x.grad[1,4],0)
        self.assertGreater(x.grad[1,3],0)
    def test_control(self):
        x=torch.zeros(204,49152)
        y=torch.tensor([3]*156+[2]*48)
        p=torch.tensor([[0,1,3,4]])
        self.assertGreater(short_objective(x,y,ARMS[1],p),short_objective(x,y,ARMS[0],p))
if __name__=="__main__":unittest.main()
