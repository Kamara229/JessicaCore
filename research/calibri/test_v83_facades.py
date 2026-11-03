#!/usr/bin/env python3
"""Pure CPU regression checks; no original model downloads, network or TEST."""
import math
import unittest
import torch
from v83.measure import MetricBook, Moment
from v83.study.scoring import summarize


class V83Numerics(unittest.TestCase):
    def test_true_pooled_l2_and_signed_bias(self):
        m = Moment()
        m.add(torch.tensor([1., 2.]), torch.tensor([2., 2.]))
        m.add(torch.tensor([3.]), torch.tensor([4.]))
        s = m.to_dict()
        self.assertEqual(s["scalar_count"], 3)
        self.assertAlmostEqual(s["relative_L2"], math.sqrt(2 / 14), places=12)
        self.assertAlmostEqual(s["signed_bias"], 2 / 3, places=12)
        self.assertEqual(s["observations"], 2)

    def test_invalid_shape_and_nonfinite_rejected(self):
        m = Moment()
        with self.assertRaises(AssertionError):
            m.add(torch.zeros(2), torch.ones(3))
        with self.assertRaises(AssertionError):
            m.add(torch.tensor([float("nan")]), torch.tensor([1.]))

    def test_error_ranking_not_claim_of_causal_sensitivity(self):
        b = MetricBook()
        b.observe("teacher_forced_local_projection", "model.layers.0.mlp.up_proj",
                  torch.tensor([1., 1.]), torch.tensor([3., 1.]))
        b.observe("teacher_forced_local_projection", "model.layers.1.mlp.up_proj",
                  torch.tensor([1., 1.]), torch.tensor([1.1, 1.]))
        b.observe("propagated_layer_output", "model.layers.0",
                  torch.tensor([1., 1.]), torch.tensor([2., 1.]))
        b.observe("propagated_layer_output", "model.layers.1",
                  torch.tensor([1., 1.]), torch.tensor([2., 1.]))
        result = summarize(b)
        self.assertEqual(result["measured_operators"], 2)
        self.assertEqual(result["measured_layers"], 2)
        self.assertEqual(result["highest_local_teacher_forced_error_energy"][0]["path"],
                         "model.layers.0.mlp.up_proj")
        self.assertTrue(result["diagnostic_not_causal_bit_allocation"])


if __name__ == "__main__":
    unittest.main()
