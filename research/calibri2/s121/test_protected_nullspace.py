"""Synthetic tests for S121 FP64 old-gradient nullspace projection."""
from __future__ import annotations
import unittest
import torch
from calibri2.s121.protected_nullspace import protected_nullspace_direction


class NullSpaceTest(unittest.TestCase):
    def test_old_gradient_component_removed_new_component_kept(self):
        g=torch.zeros(340);g[0]=-2;g[1]=1
        old=torch.zeros(23,340)
        old[0,0]=2
        old[1,2]=1
        p=protected_nullspace_direction(g,old)
        self.assertEqual(p.protected_gradient_rank,2)
        self.assertEqual(p.remaining_violations,0)
        self.assertAlmostEqual(float(p.direction[0]),0,places=8)
        self.assertAlmostEqual(float(p.direction[1]),-1,places=8)
        self.assertGreater(p.new_loss_first_order_decrease,0)
        self.assertTrue(bool(((old.double()@p.direction).abs()<1e-8).all()))

    def test_compatible_subspace_preserved(self):
        g=torch.zeros(340);g[20]=-4
        old=torch.zeros(23,340);old[:23,:23]=torch.eye(23)
        old[20,:]=0
        p=protected_nullspace_direction(g,old)
        self.assertLess(p.protected_max_normalized_dot,1e-8)
        self.assertAlmostEqual(float(p.direction[20]),4,places=8)
        self.assertEqual(p.protected_gradient_rank,22)

    def test_opposed_single_protected_requires_zero(self):
        g=torch.zeros(340);g[0]=1
        old=torch.zeros(23,340);old[0,0]=-1
        p=protected_nullspace_direction(g,old)
        self.assertEqual(p.remaining_violations,0)
        self.assertLess(float(p.direction.norm()),1e-8)

    def test_dependent_old_gradients_do_not_overconstrain(self):
        g=torch.zeros(340);g[0]=-1;g[2]=3
        old=torch.zeros(23,340)
        old[:,0]=torch.arange(1,24)
        p=protected_nullspace_direction(g,old)
        self.assertEqual(p.protected_gradient_rank,1)
        self.assertAlmostEqual(float(p.direction[2]),-3,places=8)
        self.assertTrue(bool(((old.double()@p.direction).abs()<1e-8).all()))

    def test_wrong_sizes_nan_zero_gradient_rejected(self):
        old=torch.zeros(23,340)
        good=torch.ones(340)
        for g,matrix in [
            (torch.zeros(340),old),
            (good,torch.zeros(23,341)),
            (torch.full((340,),float("nan")),old),
            (good,torch.full((23,340),float("nan"))),
        ]:
            with self.assertRaises(ValueError):
                protected_nullspace_direction(g,matrix)


if __name__=="__main__":
    unittest.main(verbosity=2)
