"""Validate the structural safety of the research-only hybrid experiment."""
import unittest
import torch
from torch import nn
from v110.mix import swap, compare
from v110.protocol import PROBES, CUTS, validate_geometry

class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(2, 2)
        self.mlp = nn.Linear(2, 2)

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 2)
        self.model.layers = nn.ModuleList(ToyLayer() for _ in range(30))
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

class Tests(unittest.TestCase):
    def test_reverts_all_parts(self):
        a, b = Toy(), Toy()
        layer = a.model.layers[0]
        emb = a.model.embed_tokens
        head = a.lm_head
        with swap(a, b, head=True, block_ids=(0, 1)):
            self.assertIs(a.model.layers[0], b.model.layers[0])
            self.assertIs(a.model.embed_tokens, b.model.embed_tokens)
            self.assertIs(a.lm_head, b.lm_head)
        self.assertIs(a.model.layers[0], layer)
        self.assertIs(a.model.embed_tokens, emb)
        self.assertIs(a.lm_head, head)

    def test_error_reverts(self):
        a, b = Toy(), Toy()
        old = a.model.layers[4].mlp
        with self.assertRaisesRegex(RuntimeError, "toy"):
            with swap(a, b, block_ids=(4,), factor="mlp"):
                raise RuntimeError("toy")
        self.assertIs(a.model.layers[4].mlp, old)
        with self.assertRaises(ValueError):
            with swap(a, b, block_ids=(3, 3)):
                pass

    def test_preregistered_protocol(self):
        self.assertEqual(validate_geometry(30)["layers"], 30)
        self.assertEqual(CUTS[-1], 30)
        self.assertEqual(len(set(p[1] for p in PROBES)), len(PROBES))
        v = torch.arange(12, dtype=torch.float32)
        self.assertEqual(compare(v, v)["top10_overlap"], 10)
        self.assertAlmostEqual(compare(v, v)["kl_from_INT6_nats"], 0, places=5)

if __name__ == "__main__":
    unittest.main()
