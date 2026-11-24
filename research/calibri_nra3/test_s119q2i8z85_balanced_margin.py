import unittest
import torch
from research.calibri_nra3.s119q2i8z85_balanced_margin import state_margin,manifest,TRAIN_ROWS,WEIGHT,PENDING,COMPLETE_NONTERMINAL
class TestZ85(unittest.TestCase):
    def test_pending(self):
        x=torch.tensor([[0.,1.,6.,3.,2.]],requires_grad=True)
        mask=torch.tensor([False,True,False,True,False])
        original=mask.clone()
        state_margin(x,1,mask).backward()
        self.assertLess(float(x.grad[0,1]),0.)
        self.assertGreater(float(x.grad[0,2]),0.)
        self.assertEqual(float(x.grad[0,3]),0.)
        self.assertTrue(torch.equal(mask,original))
    def test_complete_early_eos(self):
        x=torch.tensor([[0.,1.,6.,3.,2.]],requires_grad=True)
        mask=torch.tensor([False,True,True,True,False])
        state_margin(x,1,mask).backward()
        self.assertGreater(float(x.grad[0,2]),0.)
        self.assertLess(float(x.grad[0,1]),0.)
        self.assertEqual(float(state_margin(x,2,mask)),0.)
    def test_complete_invalid(self):
        x=torch.tensor([[8.,1.,2.,3.,0.]],requires_grad=True)
        mask=torch.tensor([False,True,True,True,False])
        state_margin(x,1,mask).backward()
        self.assertGreater(float(x.grad[0,0]),0.)
        self.assertEqual(float(x.grad[0,2]),0.)
    def test_balance(self):
        x=torch.zeros(1,5)
        a=state_margin(x,1,torch.tensor([False,True,False,True,False]))
        b=state_margin(x,1,torch.tensor([False,True,True,True,False]))
        self.assertAlmostEqual(float(a)*PENDING/TRAIN_ROWS,float(b)*COMPLETE_NONTERMINAL/TRAIN_ROWS,places=6)
        self.assertEqual(manifest()["total_positions"],PENDING+COMPLETE_NONTERMINAL+98)
        with self.assertRaises(ValueError):state_margin(x,0,torch.tensor([False,True,True,True,False]))
if __name__=="__main__":unittest.main()
