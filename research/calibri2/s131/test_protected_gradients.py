"""S131 first-order TRAIN constraint tests; actual 456+128 guard remains decisive."""
import unittest
import torch
from calibri2.s131.protected_gradients import project_against_old_loss_increase


class TestS131ProtectedGradients(unittest.TestCase):
    def test_conflicting_old_losses_projected(self):
        proposed=torch.tensor([-3.,-5.,9.,1.])
        protected=[torch.tensor([1.,0.,0.,0.]),
                   torch.tensor([0.,1.,0.,0.])]
        result,audit=project_against_old_loss_increase(proposed,protected)
        self.assertGreaterEqual(float(result[0]),-1e-7)
        self.assertGreaterEqual(float(result[1]),-1e-7)
        self.assertEqual(audit["negative_dot_before"],2)
        self.assertEqual(audit["negative_dot_after"],0)
        self.assertTrue(torch.equal(result[2:],proposed[2:]))

    def test_no_unneeded_change(self):
        proposed=torch.tensor([4.,2.,1.])
        protected=[torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])]
        result,audit=project_against_old_loss_increase(proposed,protected)
        self.assertTrue(torch.equal(proposed,result))
        self.assertEqual(audit["cyclic_projection_corrections"],0)

    def test_numerics_and_all_zero_protection_fail_closed(self):
        with self.assertRaises(AssertionError):
            project_against_old_loss_increase(torch.tensor([1.,2.]),
                                               [torch.zeros(2)])
        with self.assertRaises(ValueError):
            project_against_old_loss_increase(torch.tensor([1.,float('nan')]),
                                               [torch.ones(2)])
        with self.assertRaises(ValueError):
            project_against_old_loss_increase(torch.tensor([1.,2.]),
                                               [torch.ones(3)])

    def test_cyclic_nonorthogonal(self):
        proposed=torch.tensor([-1.,-1.,3.])
        protected=[torch.tensor([1.,1.,0.]),torch.tensor([1.,-0.2,0.])]
        result,audit=project_against_old_loss_increase(proposed,protected,passes=24)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(all(float(torch.dot(result,p)) >= -1e-5 for p in protected))


if __name__=="__main__":
    unittest.main()
