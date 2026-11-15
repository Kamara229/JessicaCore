"""Offline CPU structural tests. NOT Russian-language understanding evidence."""
import unittest
import torch
from torch import nn
from .integration import NRA3MidLayerPort
from .neural_area import NRA3Config, NRA3NeuralArea, MORPH, SEMANTIC_ROLES, UPOS


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(nn.Identity() for _ in range(12))

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class NRA3StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(30001)

    def tiny(self):
        return NRA3NeuralArea(NRA3Config(hidden=32, heads=4,
                                          grammar_blocks=1, semantic_blocks=1,
                                          relation_width=24, relation_steps=2)).eval()

    def test_default_capacity_and_three_levels(self):
        area = NRA3NeuralArea()
        self.assertEqual(len(area.grammar), 4)
        self.assertEqual(len(area.semantic), 3)
        self.assertEqual(area.config.relation_steps, 6)
        self.assertGreater(area.parameter_count(), 5_000_000)
        self.assertLess(area.parameter_count(), 7_000_000)
        self.assertGreater(area.readout.weight.abs().sum().item(), 0.)
        self.assertGreater(area.fp16_lower_bound_bpp(43_261_760, 0.69), 0.7)

    def test_shapes_no_oracle_labels_and_zero_readout(self):
        m = self.tiny()
        x = torch.randn(1, 5, 320)
        with torch.no_grad():
            f = m(x)
        self.assertEqual(tuple(f.upos_logits.shape), (1, 5, len(UPOS)))
        self.assertEqual(tuple(f.morph_logits.shape), (1, 5, len(MORPH)))
        self.assertEqual(tuple(f.semantic_role_logits.shape), (1, 5, len(SEMANTIC_ROLES)))
        self.assertEqual(tuple(f.dependency_logits.shape), (1, 5, 5))
        self.assertEqual(tuple(f.antecedent_logits.shape), (1, 5, 5))
        self.assertEqual(tuple(f.relations.shape), (1, 5, 24))
        self.assertEqual(tuple(f.readout.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(f.readout).all())
        self.assertEqual(float(f.dependency_logits[0, 0, 2]), -1e4)

    def test_prefix_causality_and_padding(self):
        m = self.tiny()
        x = torch.randn(1, 6, 320)
        x2 = x.clone()
        x2[:, 4:] += 21
        with torch.no_grad():
            a = m(x)
            b = m(x2)
        for name in ("grammar", "semantics", "relations", "upos_logits",
                     "morph_logits", "semantic_role_logits"):
            self.assertTrue(torch.allclose(getattr(a, name)[:, :4],
                                           getattr(b, name)[:, :4],
                                           atol=1e-6, rtol=0), name)
        valid = torch.tensor([[True, True, True, False, False, False]])
        with torch.no_grad():
            padded = m(x, valid)
            changed = m(x2, valid)
        self.assertTrue(torch.allclose(padded.relations[:, :3], changed.relations[:, :3]))
        with self.assertRaises(ValueError):
            m(x, torch.tensor([[True, False, True, False, False, False]]))
        with self.assertRaises(ValueError):
            m(x, torch.ones(1, 6, dtype=torch.int64))

    def test_actual_hook_shadows_at_layer8_and_leaves_parent_identical(self):
        model = ToyDecoder()
        area = self.tiny()
        port = NRA3MidLayerPort(area).install(model)
        x = torch.randn(1, 4, 320)
        baseline = model(x)
        self.assertIs(baseline, x)
        self.assertEqual(port.calls, 0)
        with port.activated():
            self.assertIs(model(x), baseline)
            self.assertEqual(port.calls, 1)
        self.assertEqual(port.last_shape, (1, 4, 320))
        self.assertEqual(port.last_features["relation"], (1, 4, 24))
        self.assertFalse(port.candidate_accepted)
        with self.assertRaises(RuntimeError):
            port.set_output_gate(1.0)
        with self.assertRaises(RuntimeError):
            with port.activated():
                with port.activated():
                    pass
        with port.activated():
            with self.assertRaises(ValueError):
                model(torch.randn(2, 4, 320))
        port.uninstall()
        self.assertIs(model(x), x)

    def test_reject_wrong_abi(self):
        with self.assertRaises(ValueError):
            NRA3Config(hidden=30, heads=8)
        m = self.tiny()
        with self.assertRaises(ValueError):
            m(torch.randn(1, 3, 321))
        with self.assertRaises(ValueError):
            m(torch.full((1, 3, 320), float("nan")))
        with self.assertRaises(ValueError):
            NRA3MidLayerPort(m).install(nn.Identity())


if __name__ == "__main__":
    unittest.main()
