"""S204 unit-only real torch Adam rollback and first-token projection contracts."""
import unittest
import torch
from calibri2.s204.guarded_nra_optimizer import guarded_nra_adam_step


class TinyNRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.decoder.weight.fill_(1.0)


class S204ActualAdamStepTests(unittest.TestCase):
    def test_actual_adam_step_requires_independent_cpu_j7(self):
        area = TinyNRA()
        opt = torch.optim.AdamW(area.parameters(), lr=0.05, weight_decay=0.0)
        (area.decoder.weight**2).sum().backward()
        guard = [{"neuro_representation_area.decoder.weight": torch.ones_like(
            area.decoder.weight)}]
        result = guarded_nra_adam_step(
            area=area, optimizer=opt, protected_first_token_grads=guard
        )
        self.assertTrue(result["actual_nra_adam_step"])
        self.assertFalse(result["model_accepted"])
        self.assertTrue(result["independent_original_456_new128_novel32_native_cpu_j7_required"])
        self.assertLess(float(area.decoder.weight.item()), 1.0)
        self.assertTrue(result["fp16_candidate_feasible_first_order_only"])

    def test_conflicting_old_token_rolls_back_optimizer_and_weights(self):
        area = TinyNRA()
        opt = torch.optim.AdamW(area.parameters(), lr=0.1, weight_decay=0.0)
        # New TRAIN wants to increase a weight while protected old loss
        # wants it to decrease. In one dimension projection must be zero.
        (-area.decoder.weight).sum().backward()
        guard = [{"neuro_representation_area.decoder.weight": torch.ones_like(
            area.decoder.weight)}]
        before = area.decoder.weight.detach().clone()
        with self.assertRaises(AssertionError):
            guarded_nra_adam_step(
                area=area, optimizer=opt, protected_first_token_grads=guard
            )
        self.assertTrue(torch.equal(area.decoder.weight.detach(), before))
        self.assertFalse(opt.state)
        self.assertEqual(opt.state_dict()["state"], {})

    def test_rejects_optimizer_with_non_nra_weight(self):
        area = TinyNRA()
        illegal = torch.nn.Parameter(torch.tensor([1.0]))
        opt = torch.optim.AdamW([*area.parameters(), illegal], lr=0.05)
        (area.decoder.weight**2).sum().backward()
        guard = [{"neuro_representation_area.decoder.weight": torch.ones_like(
            area.decoder.weight)}]
        with self.assertRaises(AssertionError):
            guarded_nra_adam_step(
                area=area, optimizer=opt, protected_first_token_grads=guard
            )


if __name__ == "__main__":
    unittest.main()
