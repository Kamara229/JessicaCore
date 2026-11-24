import unittest
import torch
from research.calibri_nra3.s119q2i8z78_pending_gold_margin import pending_margin,manifest
class TestZ78(unittest.TestCase):
    def test_gradient_on_gold_and_hardest_invalid(self):
        x=torch.tensor([[0.,1.,5.,3.,2.]],requires_grad=True)
        mask=torch.tensor([False,True,False,True,False])
        loss=pending_margin(x,1,mask,1)
        loss.backward()
        self.assertLess(float(x.grad[0,1]),0)
        self.assertGreater(float(x.grad[0,2]),0)
        self.assertEqual(float(x.grad[0,3]),0.)
        self.assertEqual(float(x.grad[0,0]),0.)
        self.assertEqual(float(x.grad[0,4]),0.)
        with torch.no_grad():
            better=x.clone();better[0,1]+=3
            self.assertLess(float(pending_margin(better,1,mask,1)),float(loss))
    def test_complete_state_unchanged_and_bad_target_refused(self):
        x=torch.zeros(1,5)
        complete=torch.tensor([False,True,True,True,False])
        self.assertEqual(float(pending_margin(x,1,complete,1)),0.)
        with self.assertRaises(ValueError):pending_margin(x,0,complete,1)
    def test_normalization(self):
        x=torch.zeros(1,5);mask=torch.tensor([False,True,False,True,False])
        self.assertAlmostEqual(float(pending_margin(x,1,mask,1)),2*float(pending_margin(x,1,mask,2)),places=6)
        self.assertEqual(manifest()["planned_Adam_updates"],784)
if __name__=="__main__":unittest.main()
