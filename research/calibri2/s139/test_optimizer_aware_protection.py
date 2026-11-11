"""S139 synthetic CPU tests only; not a real CALIBRI model proof."""
from __future__ import annotations

from pathlib import Path
import unittest
import torch
from calibri2.s139.optimizer_aware_protection import correct_adam_stage2_displacement


class TestPostAdamProtection(unittest.TestCase):
    def test_conflicting_actual_adam_step_is_repaired(self):
        # Adam can move in an old-loss increasing direction even though
        # the pre-Adam combined gradient has been projected.
        p=torch.nn.Parameter(torch.zeros(2))
        opt=torch.optim.Adam([p],lr=.05)
        p.grad=torch.tensor([-0.4, 1.0])
        before=p.detach().clone()
        opt.step()
        after=p.detach().clone()
        old=[torch.tensor([1.0,0.0])]
        self.assertGreater(float(torch.dot(after-before,old[0])),0)
        corrected,audit=correct_adam_stage2_displacement(before,after,old)
        self.assertLessEqual(float(torch.dot(corrected-before,old[0])),1e-6)
        self.assertLess(float(corrected[1]),0)
        self.assertTrue(audit["post_adam_projection_required"])

    def test_nonconflicting_step_is_unchanged(self):
        start=torch.zeros(3)
        candidate=torch.tensor([-0.1,-0.2,0.4])
        old=[torch.tensor([1.0,0.,0.]),torch.tensor([0.,1.0,0.])]
        chosen,audit=correct_adam_stage2_displacement(start,candidate,old)
        self.assertTrue(torch.equal(chosen,candidate))
        self.assertFalse(audit["post_adam_projection_required"])

    def test_multiple_old_constraints_hold_after_correction(self):
        before=torch.tensor([1.0,2.0,3.0])
        after=torch.tensor([1.8,1.0,3.4])
        old=[torch.tensor([1.,0.,0.]),torch.tensor([1.,1.,0.]),
             torch.tensor([0.,1.,0.])]
        selected,audit=correct_adam_stage2_displacement(before,after,old)
        for grad in old:
            self.assertLessEqual(float(torch.dot(selected-before,grad)),1e-5)
        self.assertGreaterEqual(audit["post_adam_min_cosine_after"],-1e-6)

    def test_empty_or_wrong_device_shape_or_nonfinite_fail_closed(self):
        a=torch.zeros(2)
        with self.assertRaises(AssertionError):
            correct_adam_stage2_displacement(a,a,[torch.zeros_like(a)])
        with self.assertRaises(ValueError):
            correct_adam_stage2_displacement(a,a,[])
        with self.assertRaises(ValueError):
            correct_adam_stage2_displacement(a,a,[torch.zeros(3)])
        with self.assertRaises(ValueError):
            correct_adam_stage2_displacement(a,torch.tensor([float("nan"),0]),[torch.ones(2)])

    def test_original_s135_j7_integration_does_not_bypass_cpu(self):
        path=Path(__file__).parents[1]/"s135"/"train_stage2_43m_five_cycles.py"
        source=path.read_text()
        self.assertIn("correct_adam_stage2_displacement(",source)
        self.assertIn("expansion.up[:,:,448:472].copy_(safe_stage2)",source)
        self.assertIn("post_adam_guard",source)
        self.assertIn("persisted_cpu_guard",source)
        self.assertIn("atomic_train_block(",source)
        self.assertIn("mask_gradient_to_new_channels_only(expansion)",source)


if __name__=="__main__":
    unittest.main()
