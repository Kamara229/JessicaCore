import unittest
import torch
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import NativeHeadAlignment
from research.calibri_nra3.s119q2i8z73_head_revival import revive,activation_summary
class TestZ73(unittest.TestCase):
    def test_exact_zero_revive_preserves_logits(self):
        h=NativeHeadAlignment()
        with torch.no_grad():
            h.down.weight[2:].zero_()
            h.up.weight.normal_(0,.01)
        x=torch.randn(5,1,320)
        cache=[{"hidden":x,"base":torch.zeros(5,49152),"gold":[1]*5}]
        before=activation_summary(h,cache)
        self.assertEqual(before["exact_zero_units"],[2,3])
        r=revive(h,cache)
        self.assertTrue(r["initial_top1_parity"])
        self.assertEqual(activation_summary(h,cache)["exact_zero_units"],[])
        self.assertTrue(torch.equal(h.up.weight[:,2:],torch.zeros_like(h.up.weight[:,2:])))
if __name__=="__main__":unittest.main()
