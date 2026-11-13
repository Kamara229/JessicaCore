"""Deterministic S204 first-order unit tests; NOT a native model CPU J7 test."""
import unittest
import torch
from calibri2.s204.first_token_grad_guard import project_protected_nra_step

P = "neuro_representation_area.decoder.weight"
E = "neuro_representation_area.encoder.weight"


class S204ConstrainedUpdateTests(unittest.TestCase):
    def test_actual_adam_direction_projected_through_fp16(self):
        before = {P: torch.tensor([1.0, 1.0], dtype=torch.float32),
                  E: torch.tensor([0.5], dtype=torch.float32)}
        after = {P: torch.tensor([0.6, 1.1]), E: torch.tensor([0.4])}
        # Old-token CE must not increase along actual optimizer displacement.
        old_grads = [{P: torch.tensor([0.0, 1.0]), E: torch.zeros(1)}]
        result, info = project_protected_nra_step(before, after, old_grads)
        self.assertEqual(info["protected_first_token_train_examples"], 1)
        self.assertFalse(info["model_accepted"])
        self.assertTrue(info["must_run_independent_full_456_128_32_cpu_j7"])
        self.assertGreaterEqual(info["fp16_min_protected_gradient_cosine"], -1e-6)
        self.assertLessEqual(float(result[P][1]), 1.0)
        self.assertLess(float(result[P][0]), 1.0)

    def test_rejects_unprotected_state_and_missing_gradients(self):
        p = {P: torch.tensor([1.0])}
        with self.assertRaises(ValueError):
            project_protected_nra_step(p, p, [])
        with self.assertRaises(ValueError):
            project_protected_nra_step(
                {"core.weight": torch.ones(1)},
                {"core.weight": torch.ones(1)},
                [{"core.weight": torch.ones(1)}])
        with self.assertRaises(ValueError):
            project_protected_nra_step(p, p, [{}])
        with self.assertRaises(AssertionError):
            project_protected_nra_step(p, p, [{P: torch.zeros(1)}])

    def test_rejects_nonfinite_and_mismatched_gradient(self):
        b = {P: torch.ones(2)}
        a = {P: torch.ones(2) * 0.9}
        for grad in (torch.tensor([float("nan"), 1.]), torch.ones(3)):
            with self.assertRaises(ValueError):
                project_protected_nra_step(b, a, [{P: grad}])

    def test_does_not_claim_full_answer_acceptance(self):
        before = {P: torch.ones(2)}
        after = {P: torch.ones(2) * 0.95}
        grads = [{P: torch.tensor([1.0, 0.0])}]
        result, audit = project_protected_nra_step(before, after, grads)
        self.assertFalse(audit["model_accepted"])
        self.assertTrue(audit["s204_train_only"])
        self.assertTrue(torch.isfinite(result[P]).all())


if __name__ == "__main__":
    unittest.main()
