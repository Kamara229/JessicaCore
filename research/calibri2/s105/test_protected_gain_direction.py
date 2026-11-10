"""Synthetic-only CPU checks for guarded 10D gain directions."""
import unittest
import torch

from calibri2.s105.protected_gain_direction import project_protected_gain_direction


class TestProtectedGainDirection(unittest.TestCase):
    def test_project_conflicting_old_gradients(self):
        wrong = torch.tensor([-2., 0.5] + [0.] * 8)
        old = torch.tensor([[1., 0.] + [0.] * 8])
        sol = project_protected_gain_direction(wrong, old)
        self.assertEqual(sol.original_violations, 1)
        self.assertEqual(sol.remaining_violations, 0)
        self.assertLessEqual(torch.dot(old[0].double(), sol.direction).item(), 1e-7)
        self.assertGreater(sol.projected_improvement, 0)
        self.assertAlmostEqual(sol.direction[0].item(), 0., places=6)

    def test_unconstrained_new_gradient_preserved(self):
        wrong = torch.tensor([1.] + [0.] * 9)
        old = torch.tensor([[1.] + [0.] * 9])
        sol = project_protected_gain_direction(wrong, old)
        self.assertAlmostEqual(sol.norm_ratio, 1., places=6)
        self.assertEqual(sol.original_violations, 0)

    def test_adversarial_opposing_protections_zero_direction(self):
        wrong = torch.tensor([1.] + [0.] * 9)
        old = torch.tensor([[-1.] + [0.] * 9])
        sol = project_protected_gain_direction(wrong, old)
        self.assertEqual(sol.remaining_violations, 0)
        self.assertLessEqual(sol.direction.norm().item(), 1e-7)

    def test_incompatible_shape_and_nonfinite_fail_closed(self):
        target = torch.ones(10)
        with self.assertRaises(ValueError):
            project_protected_gain_direction(target, torch.ones(9, 9))
        with self.assertRaises(ValueError):
            project_protected_gain_direction(target, torch.full((1, 10), float("nan")))
        with self.assertRaises(ValueError):
            project_protected_gain_direction(torch.zeros(10), torch.ones(1, 10))


if __name__=="__main__":
    unittest.main(verbosity=2)
