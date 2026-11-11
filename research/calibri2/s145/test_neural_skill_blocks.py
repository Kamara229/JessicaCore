"""S145 deterministic CPU contract tests only; NOT language QA or skill proof."""
import unittest

import torch

from .neural_skill_blocks import LayerwiseNeuralSkillBank, NeuralSkillSpec


class NeuralSkillContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(145)
        self.bank = LayerwiseNeuralSkillBank(hidden_size=8, num_layers=12)
        self.x = torch.randn(2, 5, 8)

    def test_zero_init_preserves_all_old_hidden_states_exactly(self):
        self.bank.add_skill(NeuralSkillSpec("math", (3, 9), 2))
        self.bank.add_skill(NeuralSkillSpec("language", (3, 9), 4))
        for layer in range(12):
            self.assertTrue(torch.equal(self.bank.forward_at_layer(layer, self.x), self.x))

    def test_invalid_and_duplicate_registration_fail_before_mutation(self):
        self.bank.add_skill(NeuralSkillSpec("math", (3, 9), 2))
        with self.assertRaises(ValueError):
            self.bank.add_skill(NeuralSkillSpec("math", (3, 9), 2))
        with self.assertRaises(ValueError):
            self.bank.add_skill(NeuralSkillSpec("bad", (12,), 2))
        with self.assertRaises(ValueError):
            NeuralSkillSpec("bad name", (3,), 2)
        with self.assertRaises(ValueError):
            NeuralSkillSpec("bad", (3, 3), 2)
        self.assertEqual(set(self.bank.specs), {"math"})

    def test_train_only_selected_skill_and_freeze_old_router(self):
        self.bank.add_skill(NeuralSkillSpec("old", (3,), 2))
        old = self.bank.layers["3"].blocks["old"]
        with torch.no_grad():
            old.up.weight.fill_(0.01)
        baseline = self.bank.forward_at_layer(3, self.x).detach().clone()
        old_params = [p.detach().clone() for n,p in self.bank.named_parameters()
                      if ".old." in n]
        self.bank.add_skill(NeuralSkillSpec("new", (3,), 2))
        params = self.bank.select_new_skill_for_training("new")
        self.assertEqual(len(params), 4)
        optimizer = torch.optim.SGD(params, lr=0.2)
        for _ in range(2):
            optimizer.zero_grad()
            output = self.bank.forward_at_layer(3, self.x)
            (output - (self.x + 0.1)).square().mean().backward()
            optimizer.step()
        frozen = [p for n,p in self.bank.named_parameters() if ".old." in n]
        for old_copy, now in zip(old_params, frozen):
            self.assertTrue(torch.equal(old_copy, now))
        self.assertTrue(torch.equal(
            baseline, self.bank.forward_at_layer(3, self.x, disabled=frozenset({"new"}))
        ))

    def test_down_and_gate_get_gradient_after_up_has_moved(self):
        self.bank.add_skill(NeuralSkillSpec("math", (3,), 2))
        layer = self.bank.layers["3"]
        block = layer.blocks["math"]
        router = layer.routers["math"]
        optimizer = torch.optim.SGD(self.bank.parameters(), lr=0.2)
        optimizer.zero_grad()
        (self.bank.forward_at_layer(3,self.x)-(self.x+1)).square().mean().backward()
        self.assertEqual(block.down.weight.grad.abs().sum().item(), 0)
        self.assertGreater(block.up.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        (self.bank.forward_at_layer(3,self.x)-(self.x+1)).square().mean().backward()
        self.assertGreater(block.down.weight.grad.abs().sum().item(), 0)
        self.assertGreater(router.bias.grad.abs().sum().item(), 0)

    def test_router_has_no_future_sequence_leakage(self):
        self.bank.add_skill(NeuralSkillSpec("math", (3,), 2))
        with torch.no_grad():
            self.bank.layers["3"].blocks["math"].up.weight.fill_(0.02)
            self.bank.layers["3"].routers["math"].weight.fill_(0.01)
        a = self.bank.forward_at_layer(3, self.x)
        b = self.bank.forward_at_layer(3, self.x[:, :2, :])
        self.assertTrue(torch.equal(a[:, :2], b))

    def test_two_skills_compose_without_mutating_one_another(self):
        self.bank.add_skill(NeuralSkillSpec("a", (3,), 2))
        self.bank.add_skill(NeuralSkillSpec("b", (3,), 2))
        layer = self.bank.layers["3"]
        with torch.no_grad():
            layer.blocks["a"].up.weight.fill_(0.01)
            layer.blocks["b"].up.weight.fill_(-0.02)
        both = self.bank.forward_at_layer(3, self.x)
        only_a = self.bank.forward_at_layer(3, self.x, disabled=frozenset({"b"}))
        only_b = self.bank.forward_at_layer(3, self.x, disabled=frozenset({"a"}))
        self.assertTrue(torch.allclose(both, only_a + only_b - self.x, atol=1e-7))

    def test_manifest_is_deterministic_versioned_and_hashed(self):
        bank2 = LayerwiseNeuralSkillBank(8,12)
        self.bank.add_skill(NeuralSkillSpec("z", (9,), 2, version=3))
        self.bank.add_skill(NeuralSkillSpec("a", (3,), 4))
        bank2.add_skill(NeuralSkillSpec("a", (3,), 4))
        bank2.add_skill(NeuralSkillSpec("z", (9,), 2, version=3))
        self.assertEqual(self.bank.canonical_manifest(),bank2.canonical_manifest())
        self.assertEqual(self.bank.manifest_sha256(),bank2.manifest_sha256())
        self.assertIn(b'"version":3',self.bank.canonical_manifest())

    def test_physical_cost_and_bpp_include_router_and_metadata(self):
        base = self.bank.provisional_bpp(parent_parameters=43_261_760,
                                         parent_physical_bytes=3_550_000)
        self.bank.add_skill(NeuralSkillSpec("math", (3, 9), 8))
        self.assertEqual(self.bank.parameter_count(), 10_882)
        self.bank.half()
        self.assertEqual(self.bank.physical_bytes_in_current_dtype(), 21_764)
        self.assertGreater(self.bank.provisional_bpp(
            parent_parameters=43_261_760,parent_physical_bytes=3_550_000),base)
        self.assertGreaterEqual(self.bank.fixed_parent_reference_bpp(
            parent_parameters=43_261_760,parent_physical_bytes=3_550_000),
            self.bank.provisional_bpp(
            parent_parameters=43_261_760,parent_physical_bytes=3_550_000))
        with self.assertRaises(ValueError):
            self.bank.provisional_bpp(parent_parameters=0,parent_physical_bytes=1)


if __name__ == "__main__":
    unittest.main()
