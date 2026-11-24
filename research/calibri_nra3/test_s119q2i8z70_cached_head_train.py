import unittest
import torch
from research.calibri_nra3.s119q2i8z70_cached_head_train import row_loss,ARMS
class SmallHead(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.zeros(49152))
    def forward(self,h):return self.p.view(1,1,-1).expand(h.shape[0],1,-1)
class TestZ70(unittest.TestCase):
    def test_only_pending_reweighted(self):
        h=SmallHead();m=torch.ones(49152,dtype=torch.bool);m[3]=False
        pending=torch.zeros_like(m);pending[4]=True
        row={"gold":[4,2],"hidden":torch.zeros(2,1,320),"base":torch.zeros(2,49152),"masks":[pending,m]}
        a=row_loss(h,row,None,ARMS[0]);b=row_loss(h,row,None,ARMS[1])
        self.assertAlmostEqual(a[1],b[1],places=6)
        self.assertGreater(float(b[0]),float(a[0]))
        b[0].backward()
        self.assertTrue(torch.isfinite(h.p.grad).all())
if __name__=="__main__":unittest.main()
