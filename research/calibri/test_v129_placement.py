import unittest
from unittest.mock import patch
import numpy as np
import torch
from v118.protocol import VOCAB, HEAD_DIM, factor_data_bytes
from v128.lexical import calculate, emitted_prefix, rank
from v128.scoring import matches
from v129 import archive
from v129.protocol import layout, TRAIN, VALID
from v129.operators import TiedInput


class Tokenizer:
    all_special_ids = [0]

    def decode(self, ids, **kw):
        return {0: "", 1: " ", 2: "3", 3: "4", 4: "x"}[ids[0]]

    def encode(self, text, **kw):
        return {"34": [2, 3], " 34": [1, 2, 3]}[text]


class Tests(unittest.TestCase):
    def test_exact_answer_preserves_semantics(self):
        self.assertFalse(matches("7", "-7"))
        self.assertFalse(matches("+7", "-7"))
        self.assertFalse(matches("2 5", "2.5"))
        self.assertTrue(matches("  −7. ", "-7"))
        self.assertFalse(matches("NI", "Ni", "Give the chemical symbol for nickel."))
        self.assertTrue(matches(" YES. ", "Yes"))

    def test_actual_prefix_distinguishes_forcing_eos_and_generated_space(self):
        tok = Tokenizer()
        self.assertEqual(emitted_prefix(tok, [1, 0, 2]), ([1], None))
        self.assertEqual(emitted_prefix(tok, [1, 1]), ([1, 1], None))
        def model(input_ids, use_cache):
            from types import SimpleNamespace
            logits = [0., 5., 1., 3., 4.] if input_ids.shape[-1]==1 else [0., 1., 5., 3., 4.]
            return SimpleNamespace(logits=torch.tensor([[logits]]))
        probe = lambda ids: calculate(model, tok, torch.tensor([[4]]), "34", ids)
        self.assertEqual(probe([1, 2])["gold_lexical_rank_on_generated_prefix"], 1)
        self.assertEqual(probe([4])["gold_lexical_rank_on_generated_prefix"], 4)
        self.assertIsNone(probe([0])["gold_lexical_rank_on_generated_prefix"])
        self.assertEqual(rank(torch.tensor([1., 1., 1.]), 1), 2)
        with self.assertRaises(ValueError):
            rank(torch.tensor([float("nan")]), 0)

    def test_shared_lookup_uses_output_factors(self):
        from torch import nn
        head = nn.Module()
        head.u = nn.Parameter(torch.arange(8.).reshape(4, 2))
        head.v = nn.Parameter(torch.tensor([[1., 0., -1.], [0., 1., 1.]]))
        base = nn.Embedding(4, 3)
        layer = TiedInput(base, head)
        ids = torch.tensor([3, 1])
        torch.testing.assert_close(layer(ids) - base(ids), (head.u @ head.v)[ids])
        layer(ids).sum().backward()
        self.assertIsNotNone(head.u.grad)
        self.assertIsNotNone(head.v.grad)

    def test_physical_roles_and_archive_roundtrip(self):
        for moved in (False, True):
            roles = layout(moved)
            self.assertEqual(len(set(n for n, *_ in roles)), 60)
            self.assertEqual(sum((r+c)*rank*2 for n,r,c,rank in roles) + 2*(3*VOCAB+2*HEAD_DIM),
                             factor_data_bytes())
        self.assertFalse({q for q,g in TRAIN} & {q for q,g in VALID})
        factors = {n:(np.full((r,k), i+1, np.float16), np.ones((k,c), np.float16))
                   for i,(n,r,c,k) in enumerate(layout(True))}
        head = (np.ones((VOCAB, 2)), np.ones((2, HEAD_DIM)), np.ones(VOCAB))
        base = b"test"
        size = 8 + len(base) + factor_data_bytes() + 36
        with patch.object(archive, "pinned"), patch.object(archive, "SOURCE_BYTES", len(base)), patch.object(archive, "SIZE", size):
            data = archive.pack(base, factors, head)
            actual = archive.Source(data)
            for name, expected in factors.items():
                for got, want in zip(actual.role(name), expected):
                    np.testing.assert_array_equal(got, want)
            self.assertEqual(actual.base, base)
            with self.assertRaises(ValueError):
                archive.Source(data[:-1]+bytes([data[-1]^1]))


if __name__=="__main__":
    unittest.main()
