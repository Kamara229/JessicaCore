"""Gate=0 means exact original native packed operation; gain=2 doubles correction."""
import unittest
import numpy as np
import torch
from torch import nn
from v121.gates import Gate
class Paid(nn.Module):
    def __init__(self):
        super().__init__()
        self.original=nn.Linear(3,4,bias=False)
        self.u=torch.tensor([[1.],[2.],[3.],[4.]])
        self.v=torch.tensor([[.1,.2,.3]])
        self.zero=False
    def forward(self,x):
        return self.original(x)+torch.nn.functional.linear(
            torch.nn.functional.linear(x,self.v),self.u)
class TestIntervene(unittest.TestCase):
    def test_all_gains(self):
        # Gating real v118.Block rather than fabricated values.
        from v118.runtime import Block
        base=nn.Module()
        base.out_features=4
        base.in_features=3
        base.forward=lambda x:x[:,0:1].expand(-1,4)
        src=Block(base,np.ones((4,1),dtype=np.float16),
                  np.ones((1,3),dtype=np.float16))
        x=torch.tensor([[.4,.5,.6]])
        zero=Gate(src,0)(x)
        one=Gate(src,1)(x)
        double=Gate(src,2)(x)
        torch.testing.assert_close(zero,base(x))
        torch.testing.assert_close(double-zero,2*(one-zero))
        with self.assertRaises(ValueError):Gate(src,3)
if __name__=="__main__":
    unittest.main()
