"""Isolated S103 CPU unit checks; synthetic tensors only, NEVER QA or model checkpoints."""
from __future__ import annotations
import unittest
from types import SimpleNamespace
import torch
from torch import nn

from calibri2.s103.shared_all_layer_green_bridge import SharedInterLayerGreenBridge


class DecoderLayer(nn.Module):
    def __init__(self, i: int):
        super().__init__()
        self.scale = 1 + i * 0.008
        self.last_input = None

    def forward(self, value, cache=None):
        self.last_input = value.detach().clone()
        return (value * self.scale + 0.02, cache)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(DecoderLayer(i) for i in range(12))

    def forward(self, x):
        cache_marker = object()
        for layer in self.model.layers:
            x, marker = layer(x, cache_marker)
            if marker is not cache_marker:
                raise AssertionError("Cache metadata was changed")
        return x


class FakeTunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = (3, 9)
        self.blocks = nn.ModuleDict({
            "3": nn.Linear(4, 4, bias=False),
            "9": nn.Linear(4, 4, bias=False),
        })
        with torch.no_grad():
            self.blocks["3"].weight.copy_(torch.eye(4) * 0.2)
            self.blocks["9"].weight.copy_(torch.eye(4) * 0.03)
        self._handles = []

    def install(self, model):
        for i in self.layers:
            def adapt(_module, _args, out, layer=i):
                return (out[0] + self.blocks[str(layer)](out[0]), *out[1:])
            self._handles.append(model.model.layers[i].register_forward_hook(adapt))
        return self

    def uninstall(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class TestSharedInterLayerGreenBridge(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(103)
        self.model = DummyModel()
        self.original = FakeTunnel().install(self.model)
        self.bridge = SharedInterLayerGreenBridge(self.original)
        self.bridge.install(self.model)
        self.x = torch.tensor([[0.3, -0.8, 0.6, 1.2]], dtype=torch.float32)

    def tearDown(self):
        self.bridge.uninstall()
        self.original.uninstall()

    def test_all_twelve_connected_no_duplicate_donor_weights(self):
        self.assertEqual(self.bridge.connected_layers, tuple(range(12)))
        self.assertEqual(self.bridge.additional_layers, (0, 1, 2, 4, 5, 6, 7, 8, 10, 11))
        self.assertEqual(self.bridge.trainable_added_parameter_count(), 10)
        self.assertEqual(list(self.bridge.state_dict()), ["layer_gains"])
        self.assertEqual(len(self.bridge._handles) + len(self.original._handles), 12)

    def test_zero_gains_exact_original_for_every_route(self):
        with torch.inference_mode():
            for route in (None, "S87", "S91", "S93"):
                with self.bridge.profile(route):
                    baseline = self.model(self.x)
                    self.bridge.uninstall()
                    try:
                        restored = self.model(self.x)
                    finally:
                        self.bridge.install(self.model)
                    self.assertTrue(torch.equal(baseline, restored), route)

    def test_route_protection_and_actual_cross_layer_propagation(self):
        with torch.no_grad():
            self.bridge.layer_gains[0] = 0.7
        with torch.inference_mode():
            with self.bridge.profile("S87"):
                old = self.model(self.x)
                in1_old = self.model.model.layers[1].last_input.clone()
            with self.bridge.profile("S91"):
                self.assertTrue(torch.equal(old, self.model(self.x)))
            with self.bridge.profile("S93"):
                updated = self.model(self.x)
                in1_new = self.model.model.layers[1].last_input.clone()
        self.assertFalse(torch.equal(in1_new, in1_old))
        self.assertFalse(torch.equal(updated, old))

    def test_gain_receives_gradient_from_full_model_output(self):
        with self.bridge.profile("S93"):
            out = self.model(self.x)
            (out**2).mean().backward()
        self.assertIsNotNone(self.bridge.layer_gains.grad)
        self.assertTrue(torch.isfinite(self.bridge.layer_gains.grad).all())
        self.assertNotEqual(torch.count_nonzero(self.bridge.layer_gains.grad).item(), 0)
        self.assertTrue(torch.equal(self.bridge.layer_gains, torch.zeros_like(self.bridge.layer_gains)))

    def test_state_serialization_only_gain_and_scope_restoration(self):
        with torch.no_grad():
            self.bridge.layer_gains[4] = 0.125
        payload = self.bridge.get_bridge_state()
        self.assertEqual(list(payload), ["all_layer_bridge.layer_gains"])
        original = payload["all_layer_bridge.layer_gains"].clone()
        self.bridge.load_bridge_state(payload)
        self.assertTrue(torch.equal(self.bridge.layer_gains, original))
        try:
            with self.bridge.profile("S93"):
                raise RuntimeError("intentional")
        except RuntimeError:
            pass
        self.assertIsNone(self.bridge._active_profile)
        with self.assertRaises(ValueError):
            self.bridge.load_bridge_state({"all_layer_bridge.layer_gains": torch.ones(9)})
        with self.assertRaises(ValueError):
            self.bridge.load_bridge_state({"all_layer_bridge.layer_gains": torch.full((10,), float("nan"))})


if __name__ == "__main__":
    unittest.main(verbosity=2)
