import unittest
import torch
from research.calibri_nra3.s119q2i8z95_native_feasibility import feasibility
class TestZ95(unittest.TestCase):
    def test_full_competition(self):
        h=torch.eye(3)
        base=torch.tensor([[0.,0.,8.,1.],[0.,0.,8.,1.],[0.,0.,8.,1.]])
        gold=torch.tensor([0,1,3])
        r=feasibility(h,base,gold)
        self.assertEqual(r["top1"],3)
        self.assertTrue(r["all_non_target_logits_unchanged"])
        self.assertEqual(r["target_columns"],3)
        self.assertEqual(feasibility(torch.zeros(3,1),base,gold)["top1"],1)
    def test_invalid(self):
        with self.assertRaises(Exception):feasibility(torch.zeros(1,1),torch.zeros(1,2),torch.tensor([2]))
if __name__=="__main__":unittest.main()
