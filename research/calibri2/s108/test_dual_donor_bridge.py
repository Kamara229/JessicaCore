"""S108 synthetic-only dual-donor architecture checks. No QA or saved model."""
from __future__ import annotations

import unittest
import torch
from calibri2.s103.test_shared_all_layer_green_bridge import DummyModel, FakeTunnel
from calibri2.s108.dual_donor_interlayer_bridge import DualDonorInterlayerGreenBridge


class TestDualDonorInterlayerGreenBridge(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(108)
        self.model = DummyModel()
        self.original = FakeTunnel().install(self.model)
        self.dual = DualDonorInterlayerGreenBridge(self.original).install(self.model)
        self.x = torch.tensor([[0.2, -0.4, 0.8, 0.1]], dtype=torch.float32)

    def tearDown(self):
        self.dual.uninstall()
        self.original.uninstall()

    def test_exact_numerical_parity_with_original_12_layers_at_zero(self):
        self.assertEqual(self.dual.connected_layers, tuple(range(12)))
        self.assertEqual(len(self.dual._handles), 10)
        self.assertEqual(len(self.original._handles), 2)
        self.assertEqual(self.dual.trainable_new_parameters(), 20)
        self.assertEqual(list(self.dual.state_dict()), ["layer_gains"])
        with torch.inference_mode():
            for route in ("S93", "S87", "S91", None):
                with self.dual.profile(route):
                    actual = self.model(self.x)
                self.dual.uninstall()
                expected = self.model(self.x)
                self.dual.install(self.model)
                self.assertTrue(torch.equal(actual, expected), route)

    def test_both_donors_have_independent_influence_on_following_layer(self):
        with torch.inference_mode():
            with self.dual.profile(None):
                base_output = self.model(self.x)
                old_in1 = self.model.model.layers[1].last_input.clone()
            with torch.no_grad():
                self.dual.layer_gains[0, 0] = 0.42
            with self.dual.profile("S93"):
                first = self.model(self.x)
                first_in1 = self.model.model.layers[1].last_input.clone()
            with torch.no_grad():
                self.dual.layer_gains[0, 0] = 0.
                self.dual.layer_gains[1, 0] = 0.42
            with self.dual.profile("S93"):
                second = self.model(self.x)
                second_in1 = self.model.model.layers[1].last_input.clone()
            self.assertFalse(torch.equal(old_in1, first_in1))
            self.assertFalse(torch.equal(old_in1, second_in1))
            self.assertFalse(torch.equal(first, second))
            self.assertFalse(torch.equal(first_in1, second_in1))
            with self.dual.profile("S87"):
                self.assertTrue(torch.equal(base_output, self.model(self.x)))
            with self.dual.profile("S91"):
                self.assertTrue(torch.equal(base_output, self.model(self.x)))

    def test_gradient_reaches_all_twenty_gain_positions_from_zero(self):
        with self.dual.profile("S93"):
            objective = self.model(self.x).square().mean()
            objective.backward()
        gradient = self.dual.layer_gains.grad
        self.assertEqual(tuple(gradient.shape), (2,10))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(int(torch.count_nonzero(gradient)), 20)
        self.assertEqual(len(list(self.dual.named_parameters())), 1)

    def test_only_gain_sidecar_serialized_and_checked(self):
        with torch.no_grad():
            self.dual.layer_gains[0, 3] = 0.12
            self.dual.layer_gains[1, 3] = -0.08
        data = self.dual.physical_sidecar_payload()
        self.assertEqual(list(data), [self.dual.SIDECAR_KEY])
        self.assertEqual(tuple(data[self.dual.SIDECAR_KEY].shape), (2,10))
        before = data[self.dual.SIDECAR_KEY].clone()
        with torch.no_grad():
            self.dual.layer_gains.zero_()
        self.dual.load_physical_sidecar_payload(data)
        self.assertTrue(torch.equal(before, self.dual.layer_gains))
        with self.assertRaises(ValueError):
            self.dual.load_physical_sidecar_payload({"wrong":before})
        with self.assertRaises(ValueError):
            self.dual.load_physical_sidecar_payload(
                {self.dual.SIDECAR_KEY:torch.zeros(2,9)})
        with self.assertRaises(ValueError):
            self.dual.load_physical_sidecar_payload(
                {self.dual.SIDECAR_KEY:torch.full((2,10),float("nan"))})
        try:
            with self.dual.profile("S93"):
                raise RuntimeError("artificial exception")
        except RuntimeError:
            pass
        self.assertIsNone(self.dual._active_profile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
