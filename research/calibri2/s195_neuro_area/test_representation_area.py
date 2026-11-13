"""S195-NRA deterministic CPU tests; NOT a neural quality/QA benchmark."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from calibri2.s195_neuro_area.representation_area import (
    NeuroRepresentationArea, EXPECTED_AREA_PARAMS, PARENT_PARAMS,
    WIDTH, DEPTH, trainable_region_parameters,
)


class ToyDecoder(nn.Module):
    """Minimal exact 12-layer ABI to exercise real forward hooks and gradients."""
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(DEPTH)])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class NeuroAreaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(195013)

    def test_has_distinct_trainable_weights_and_zero_output_parent_parity(self):
        area = NeuroRepresentationArea()
        self.assertEqual(area.parameter_count(), EXPECTED_AREA_PARAMS)
        self.assertEqual(PARENT_PARAMS + area.parameter_count(), 43_283_660)
        self.assertEqual(sum(p.numel() for p in trainable_region_parameters(area)),
                         EXPECTED_AREA_PARAMS)
        m = ToyDecoder()
        area.install(m)
        x = torch.randn(2, 7, WIDTH)
        before = m(x).detach().clone()
        with area.activated():
            after = m(x)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(area.last_coverage, tuple(range(12)))
        self.assertEqual(area.forward_calls, 1)
        area.uninstall()

    def test_finite_causal_tokenwise_injection_after_first_real_gradient(self):
        area = NeuroRepresentationArea()
        m = ToyDecoder()
        area.install(m)
        x = torch.randn(2, 4, WIDTH)
        parent = m(x)
        # Train only the new neural-area decoder (the 43M parent is immutable).
        target = x + .2 * torch.tanh(x)
        opt = torch.optim.Adam(trainable_region_parameters(area), lr=.015)
        losses = []
        for _ in range(8):
            opt.zero_grad(set_to_none=True)
            with area.activated():
                got = m(x)
            loss = F.mse_loss(got, target)
            loss.backward()
            self.assertTrue(torch.isfinite(area.decoder.weight.grad).all())
            opt.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(torch.equal(parent, m(x)))
        with area.activated():
            out = m(x)
        self.assertFalse(torch.equal(out, parent))
        self.assertTrue(torch.isfinite(out).all())
        # Only first-token hidden states are identical in the two streams;
        # changing a future token cannot change its neural area transform.
        future = x.clone()
        future[:, 2:] += 1.0
        with area.activated():
            out_future = m(future)
        self.assertTrue(torch.equal(out[:, :2], out_future[:, :2]))
        area.uninstall()

    def test_fp16_roundtrip_all_parameters_and_reject_extra_or_missing(self):
        from safetensors.torch import load_file, save_file
        area = NeuroRepresentationArea()
        with torch.no_grad():
            area.decoder.weight[0, 0] = 0.3125
        with tempfile.TemporaryDirectory() as td:
            file = area.save_fp16(Path(td)/"region.safetensors")
            self.assertTrue(file.is_file())
            reloaded = NeuroRepresentationArea()
            reloaded.load_fp16(file)
            self.assertEqual(set(reloaded.fp16_state()), set(area.fp16_state()))
            for key in area.fp16_state():
                self.assertTrue(torch.equal(reloaded.fp16_state()[key],
                                            area.fp16_state()[key]))
            with self.assertRaises(FileExistsError):
                area.save_fp16(file)
            missing = load_file(str(file))
            missing.pop(next(iter(missing)))
            bad = Path(td)/"missing.safetensors"
            save_file(missing, str(bad))
            with self.assertRaises(ValueError):
                reloaded.load_fp16(bad)
            extra = load_file(str(file))
            extra["neuro_representation_area.illegal_extra"] = torch.ones(1)
            bad2 = Path(td)/"extra.safetensors"
            save_file(extra, str(bad2))
            with self.assertRaises(ValueError):
                reloaded.load_fp16(bad2)

    def test_budget_does_not_disguise_extra_physical_weights_as_free(self):
        area = NeuroRepresentationArea()
        with tempfile.TemporaryDirectory() as td:
            path = area.save_fp16(Path(td)/"region.safetensors")
            bpp = area.all_in_bpp(
                parent_bpp=.6990419453181559, physical_area_bytes=path.stat().st_size)
            self.assertGreater(bpp, .7)
            self.assertGreater(path.stat().st_size, 2 * area.parameter_count())
            with self.assertRaises(ValueError):
                area.all_in_bpp(parent_bpp=float("nan"), physical_area_bytes=2)

    def test_fail_closed_wrong_layer_shapes_and_no_implicit_activation(self):
        area = NeuroRepresentationArea()
        with self.assertRaises(AssertionError):
            area.inject(torch.randn(1, 3, 321), 0)
        with self.assertRaises(IndexError):
            area.concepts(torch.randn(1, 3, WIDTH), DEPTH)
        m = ToyDecoder()
        area.install(m)
        with area.activated():
            with self.assertRaises(RuntimeError):
                with area.activated():
                    pass
            with self.assertRaises(RuntimeError):
                area.uninstall()
        area.uninstall()
        with self.assertRaises(RuntimeError):
            with area.activated():
                pass


if __name__ == "__main__":
    unittest.main()
