"""S3I frozen 64 TRAIN-only loss invariants; no CPU 43M model used."""
import unittest
import torch
from research.nra2_bridge.stage3i_controlled_attention import (
    TRAIN, TOTAL, SEED, STEPS, train_loss,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import GROUP_SIZES


class Stage3ILossBoundaryTests(unittest.TestCase):
    def test_preregistered_constants(self):
        self.assertEqual(SEED, 2026111401)
        self.assertEqual(STEPS, 240)
        self.assertEqual(TRAIN, 64)
        self.assertEqual(TOTAL, 128)
        self.assertEqual(list(GROUP_SIZES.values()), [64,16,16,16,16])

    def test_any_control_row_rejected_before_forward(self):
        # Never call a model (None) when the labels are not exactly
        # the admitted 64 training rows.
        tokens=torch.zeros((TRAIN,3,320),dtype=torch.float32)
        mask=torch.ones((TRAIN,3),dtype=torch.bool)
        label=lambda count: {
            "adjacency":torch.zeros(count,6,6),
            "source":torch.zeros(count,6),
            "target":torch.zeros(count,6),
            "reachability":torch.zeros(count),
        }
        with self.assertRaisesRegex(AssertionError,"cannot compute loss"):
            train_loss(None,None,tokens,mask,label(TRAIN+1))
        with self.assertRaisesRegex(AssertionError,"cannot compute loss"):
            train_loss(None,None,torch.zeros((TRAIN+1,3,320)),
                       torch.ones((TRAIN+1,3),dtype=torch.bool),label(TRAIN))


if __name__=="__main__":
    unittest.main()
