import unittest
import torch
from research.calibri_nra3.s119q2i8z65_stable_gate_train import objective_loss,ARMS,STABLE_GT_LR,GT_GATE_LR
class TestZ65(unittest.TestCase):
    def test_matched_loss_difference(self):
        x=torch.zeros(1,8,requires_grad=True)
        m=torch.tensor([False,False,False,False,True,True,False,False])
        a=objective_loss(x,[4,5,2],1,arm=ARMS[0],contrast=(1,5,6),mask=m,vocab=8)
        b=objective_loss(x,[4,5,2],1,arm=ARMS[1],contrast=(1,5,6),mask=m,vocab=8)
        self.assertEqual(float(a[4]),0.)
        self.assertGreater(float(b[4]),0.)
        self.assertTrue(all(torch.equal(a[i],b[i]) for i in (1,2,3)))
        self.assertTrue(torch.allclose(b[0]-a[0],b[4]))
        self.assertEqual(STABLE_GT_LR,GT_GATE_LR/2)
if __name__=="__main__":unittest.main()
