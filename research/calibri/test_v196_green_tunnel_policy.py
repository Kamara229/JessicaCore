"""Green Tunnel policy / no-free-weight source contract, standard libraries only."""
import unittest
from v196.policy import RecoveryPolicy,suggested_cycles
from v196 import protocol as P

class UniversalGreenTunnelPolicy(unittest.TestCase):
    def test_finite_multicycle_is_scalable_planning_not_semantic_claim(self):
        self.assertEqual(suggested_cycles(134_515_008),2)
        self.assertGreaterEqual(suggested_cycles(361_821_120),3)
        self.assertGreaterEqual(suggested_cycles(1_000_000_000),4)
        self.assertGreater(suggested_cycles(2_000_000_000),suggested_cycles(361_821_120))
        self.assertLessEqual(suggested_cycles(1_000_000_000_000),16)
        with self.assertRaises(ValueError):
            suggested_cycles(0)
        with self.assertRaises(ValueError):
            RecoveryPolicy(cycles=0)
        with self.assertRaises(ValueError):
            RecoveryPolicy(learning_rate=-.1)
        with self.assertRaises(ValueError):
            RecoveryPolicy(exact_physical_rate_ceiling_bpp=1)

    def test_196_restarts_ACTUAL_pretrained_135M_186_not_cb59_or_360m(self):
        self.assertEqual(P.INITIAL_135M_SHA256,
            "933daaecf9011f77e3705308faab9d6074b5fb7c4c847d5e1bdb830ad47b965a")
        self.assertEqual(P.INITIAL_TRAINED_ARTIFACT_RUN,42085087419)
        self.assertEqual(P.FROZEN_SOURCE_135M_UNIQUE,134_515_008)
        self.assertEqual(P.SOURCE_BYTES,11_229_303)
        self.assertEqual(P.MAX_BYTES,11_770_063)
        self.assertEqual(P.CODEBOOK_MATRIX_COUNT,211)
        self.assertEqual(P.VOCAB_SIZE,49_152)
        self.assertEqual(P.POLICY.cycles,2)
        self.assertEqual(len(P.ORIGINAL_LAYER_SCHEDULE),96)
        self.assertEqual(P.POLICY.cycles*len(P.ORIGINAL_LAYER_SCHEDULE),192)
        self.assertTrue(P.MODEL.endswith("135M-Instruct"))

    def test_all_level_operator_coverage_and_original_source_no_sidecar(self):
        operators=P.ORIGINAL_LAYER_SCHEDULE
        self.assertEqual(len(operators),len(set(operators)))
        # Must train ALL transformer depths, not only output head.
        for n in range(30):
            self.assertTrue(any(f"model.layers.{n}." in p for p in operators))
        self.assertEqual(len(P.PUBLIC_TRAIN_INDICES),20)
        self.assertEqual(len(set(P.PUBLIC_TRAIN_INDICES)),20)
        self.assertEqual(len(P.DEV_HISTORICAL_IDS),24)
        self.assertTrue(all(0<=x<64 for x in P.PUBLIC_TRAIN_INDICES))

if __name__=="__main__":
    unittest.main()
