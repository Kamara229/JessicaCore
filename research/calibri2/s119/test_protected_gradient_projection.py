"""Synthetic 340D S117 GEM-inspired projected-gain tests. No model/QA."""
import unittest
import torch
from calibri2.s119.protected_gradient_projection import project_for_old_skills


class TestProtectedGradientProjection(unittest.TestCase):
    def test_conflict_preserves_old_loss_first_order(self):
        g=torch.zeros(340);g[0]=-2;g[1]=1
        old=torch.zeros((3,340))
        old[0,0]=1
        old[1,2]=1
        old[2,3]=-1
        p=project_for_old_skills(g,old)
        self.assertEqual(p.originally_violated,1)
        self.assertEqual(p.remaining_violations,0)
        self.assertGreater(p.new_loss_first_order_decrease,0)
        self.assertAlmostEqual(float(p.direction[0]),0,places=6)
        self.assertTrue(bool(((old.double()@p.direction)<=1e-7).all()))

    def test_compatible_target_must_remain_intact(self):
        g=torch.zeros(340);g[0]=1
        old=torch.zeros((1,340));old[0,0]=1
        p=project_for_old_skills(g,old)
        self.assertEqual(p.remaining_violations,0)
        self.assertAlmostEqual(p.projected_norm_ratio,1.0,places=6)
        self.assertAlmostEqual(float(p.direction[0]),-1.0,places=6)

    def test_opposite_old_target_requires_zero_direction(self):
        g=torch.zeros(340);g[0]=1
        old=torch.zeros((1,340));old[0,0]=-1
        p=project_for_old_skills(g,old)
        self.assertEqual(p.remaining_violations,0)
        self.assertLess(p.projected_norm_ratio,1e-6)

    def test_many_conflicting_constraints_and_numerical_failclosed(self):
        g=torch.linspace(-1,1,340)
        old=torch.eye(340)
        p=project_for_old_skills(g,old,max_iter=500)
        self.assertEqual(p.constraints,340)
        self.assertEqual(p.remaining_violations,0)
        self.assertTrue(bool((old.double()@p.direction<=1e-7).all()))
        for gradient,guards in [
            (torch.full((340,),float('nan')),old),
            (g,torch.zeros((2,339))),
            (torch.zeros(340),old),
        ]:
            with self.assertRaises(ValueError):
                project_for_old_skills(gradient,guards)


if __name__=="__main__":
    unittest.main(verbosity=2)
