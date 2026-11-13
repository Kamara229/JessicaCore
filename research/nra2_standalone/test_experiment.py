"""Unit and property tests independent of Calibri / Jessica execution."""

import tempfile
import unittest
from pathlib import Path

import torch

from .experiment import graph_batch, run
from .model import RelationalNRA


class StandaloneNRAContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_balanced_disjoint_graphs_and_reproducible_inputs(self):
        a = graph_batch(12, 9, 96, (1, 2, 3, 4, 5))
        b = graph_batch(12, 9, 96, (1, 2, 3, 4, 5))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(int(a[3].sum()), 48)
        self.assertTrue((a[0].sum((1, 2)) == 8).all())
        # Endpoint-degree shortcut forbidden: queried nodes are both internal.
        for queried in (a[1], a[2]):
            self.assertTrue(((a[0].sum(1) * queried).sum(1) == 1).all())
            self.assertTrue(((a[0].sum(2) * queried).sum(1) == 1).all())
        self.assertTrue((a[1].sum(1) == 1).all())
        self.assertTrue((a[2].sum(1) == 1).all())

    def test_neural_gradients_and_variable_graph_sizes(self):
        torch.manual_seed(77)
        model = RelationalNRA(width=16, hops=5)
        self.assertGreater(model.parameter_count(), 100)
        for nodes, distances in ((9, (1, 2)), (10, (5,))):
            edges, src, tgt, gold = graph_batch(55, nodes, 12, distances)
            logits = model(edges, src, tgt)
            self.assertEqual(tuple(logits.shape), (12,))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, gold)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 5, 4), torch.zeros(2, 5), torch.zeros(2, 5))

    def test_first_independent_train_and_fp16_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(out=Path(directory)/"s1", steps=300)
            self.assertEqual(result["train_distribution"]["total"], 1024)
            self.assertEqual(result["novel_graph_size_five_hop_dev"]["total"], 1024)
            self.assertGreaterEqual(result["train_distribution"]["accuracy"], .98)
            self.assertGreaterEqual(result["novel_graph_size_five_hop_dev"]["accuracy"], .95)
            self.assertEqual(result["candidate_status"],
                             "EXPLORATORY_STANDALONE_PASS_NOT_INTEGRATION")
            self.assertFalse(result["calibri_2_integrated"])
            with self.assertRaises(FileExistsError):
                run(out=Path(directory)/"s1", steps=1)


if __name__ == "__main__":
    unittest.main()
