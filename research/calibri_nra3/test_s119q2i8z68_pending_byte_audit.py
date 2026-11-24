import unittest
import torch
from research.calibri_nra3.s119q2i8z68_pending_byte_audit import measure,summarize
class TestZ68(unittest.TestCase):
    def test_mass_distinct_from_correct(self):
        x=torch.tensor([[0.,4.,0.,3.]])
        m=torch.tensor([True,False,False,True])
        r=measure(x,0,m)
        self.assertFalse(r["valid_top1"])
        self.assertFalse(r["correct"])
        self.assertGreater(r["valid_mass"],0.)
        self.assertEqual(r["gold_rank"],3)
        s=summarize([r]*57)
        self.assertEqual(s["valid_top1"],0)
        self.assertEqual(s["top_invalid_ids"],[(1,57)])
if __name__=="__main__":unittest.main()
