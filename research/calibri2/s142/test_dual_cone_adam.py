"""S142 synthetic CPU checks for dual QP only; not a full-model result."""
from __future__ import annotations

import unittest
import torch
from calibri2.s142.dual_cone_adam import correct_adam_stage2_displacement


class TestS142DualConeAdam(unittest.TestCase):
    def test_near_antiparallel_constraints_preserve_unconstrained_skill_axis(self):
        # S139's 32 cyclic passes zeroed even a large orthogonal safe move.
        before = torch.zeros(3)
        after = torch.tensor([1.0, 1.0, -2.0])
        protected = [torch.tensor([1.0, 0.0, 0.0]),
                     torch.tensor([-1.0, 0.01, 0.0])]
        safe, audit = correct_adam_stage2_displacement(before, after, protected)
        self.assertTrue(audit["post_adam_projection_required"])
        self.assertFalse(audit["post_adam_zero_fallback"])
        self.assertGreater(float(safe[2]), -2.1)
        self.assertLess(float(safe[2]), -1.9)
        for g in protected:
            self.assertLessEqual(float(torch.dot(safe - before, g)), 1e-5)

    def test_actual_adam_step_protected_and_nonzero(self):
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.Adam([p], lr=.05)
        p.grad = torch.tensor([-.4, 1.0])
        before = p.detach().clone()
        opt.step()
        safe, audit = correct_adam_stage2_displacement(
            before, p.detach(), [torch.tensor([1.0, 0.0])])
        self.assertLessEqual(float(safe[0] - before[0]), 1e-6)
        self.assertLess(float(safe[1]), 0.0)
        self.assertEqual(audit["post_adam_solver"], "S142_cpu_fp64_dual_LBFGSB")

    def test_original_safe_candidate_unchanged(self):
        before = torch.zeros(2)
        after = torch.tensor([-.1, -.2])
        safe, audit = correct_adam_stage2_displacement(
            before, after, [torch.tensor([1.0, 0.0])])
        self.assertTrue(torch.equal(safe, after))
        self.assertFalse(audit["post_adam_projection_required"])

    def test_full_59_constraints_24_stage2_channels_nonzero_and_protected(self):
        torch.manual_seed(142)
        shape = (12, 320, 24)
        before = torch.zeros(shape)
        after = 1e-4 * torch.randn(shape)
        protected = [torch.randn(shape) for _ in range(59)]
        selected, audit = correct_adam_stage2_displacement(
            before, after, protected)
        self.assertEqual(audit["old_gradient_constraints"], 59)
        self.assertGreater(int(torch.count_nonzero(selected)), 0)
        self.assertGreaterEqual(audit["post_adam_min_cosine_after"], -1e-6)
        self.assertLessEqual(max(
            float(torch.sum((selected - before) * g)) /
            (float(torch.linalg.vector_norm(selected - before)) *
             float(torch.linalg.vector_norm(g))) for g in protected
        ), 1.0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple MPS")
    def test_real_mps_float32_stage_to_cpu_fp64_dual_qp_and_back(self):
        # Regression: torch.to(device="cpu", dtype=float64) may attempt
        # unsupported float64 on MPS. First stage CPU, then promote.
        before = torch.zeros((1, 4, 24), dtype=torch.float32, device="mps")
        after = before.clone()
        after[0, 0, 0] = 0.05
        after[0, 0, 1] = -0.04
        first = torch.zeros_like(before)
        first[0, 0, 0] = 1.0
        second = torch.zeros_like(before)
        second[0, 0, 1] = 1.0
        safe, audit = correct_adam_stage2_displacement(
            before, after, [first, second])
        self.assertEqual(safe.device.type, "mps")
        self.assertFalse(audit["post_adam_zero_fallback"])
        self.assertLessEqual(float((safe * first).sum().cpu()), 1e-6)
        self.assertLessEqual(float((safe * second).sum().cpu()), 1e-6)

    def test_fail_closed_on_nonfinite_wrong_shape_and_no_constraints(self):
        before = torch.zeros(2)
        with self.assertRaises(AssertionError):
            correct_adam_stage2_displacement(before, before, [torch.zeros(2)])
        with self.assertRaises(ValueError):
            correct_adam_stage2_displacement(before, before, [])
        with self.assertRaises(ValueError):
            correct_adam_stage2_displacement(before, before, [torch.zeros(3)])
        with self.assertRaises(ValueError):
            correct_adam_stage2_displacement(
                before, torch.tensor([float("nan"), 0.0]), [torch.ones(2)])


if __name__ == "__main__":
    unittest.main()
