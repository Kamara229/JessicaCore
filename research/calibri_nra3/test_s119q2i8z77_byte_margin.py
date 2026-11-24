import unittest
import torch
from research.calibri_nra3.s119q2i8z77_byte_margin import measure_margin
class TestZ77(unittest.TestCase):
    def test_margin(self):
        x=torch.tensor([[0.,1.,5.,3.]])
        m=torch.tensor([False,True,False,True])
        r=measure_margin(x,1,m)
        self.assertEqual(r["best_valid_rank"],2)
        self.assertEqual(r["best_valid_minus_invalid"],-2.)
        self.assertEqual(r["gold_minus_best_valid"],-2.)
        self.assertEqual(r["gold_minus_best_invalid"],-4.)
        self.assertFalse(r["valid_top1"])
if __name__=="__main__":unittest.main()
