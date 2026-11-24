import unittest
import torch
from research.calibri_nra3.s119q2i8z60_native_pair_objective import pair_targets,token_loss,OBJECTIVES
class TestZ60(unittest.TestCase):
    def test_first_difference(self):
        self.assertEqual(pair_targets([4,5,2],[4,6,2],vocab=8),(1,5,6))
        self.assertIsNone(pair_targets([4,2],[4,6,2],vocab=8))
        with self.assertRaises(ValueError):pair_targets([4,2],[4,2],vocab=8)
    def test_pair_gradient_and_same_ce(self):
        x=torch.zeros(1,8,requires_grad=True)
        gold=[4,5,2]
        a,ce,extra=token_loss(x,gold,1,objective=OBJECTIVES[0],contrast=(1,5,6),vocab=8)
        b,ce2,extra2=token_loss(x,gold,1,objective=OBJECTIVES[1],contrast=(1,5,6),vocab=8)
        self.assertEqual(float(extra),0.)
        self.assertEqual(float(ce),float(ce2))
        self.assertGreater(float(extra2),0.)
        delta=torch.autograd.grad(b-a,x)[0]
        self.assertLess(float(delta[0,5]),0.)
        self.assertGreater(float(delta[0,6]),0.)
        self.assertAlmostEqual(float(delta[0,2]),0.)
    def test_replay_and_eos_unchanged(self):
        x=torch.zeros(1,8,requires_grad=True)
        for t in (0,1,2):
            a=token_loss(x,[4,5,2],t,objective=OBJECTIVES[0],vocab=8)[0]
            b=token_loss(x,[4,5,2],t,objective=OBJECTIVES[1],vocab=8)[0]
            self.assertEqual(float(a),float(b))
        _,_,extra=token_loss(x,[4,5,2],2,objective=OBJECTIVES[1],contrast=(1,5,6),vocab=8)
        self.assertEqual(float(extra),0.)
        with self.assertRaises(ValueError):
            token_loss(x,[4,5,2],2,objective=OBJECTIVES[1],contrast=(2,2,6),vocab=8)
if __name__=="__main__":unittest.main()
