"""Static controls for original 360M native/oracle + unpaid FP32 intervention."""
import unittest
from v195 import protocol as P

class SourceFamilySafety(unittest.TestCase):
    def test_exact_original_360m_source_and_rate_unchanged(self):
        self.assertEqual(P.ORIGINAL_UNIQUE_PARAMETERS,361_821_120)
        self.assertEqual(P.SOURCE_BYTES,29_348_539)
        self.assertEqual(P.SOURCE_SHA256,
                         "69e2e0f67c1e642ae468a94fc28047a7b1b77afe91776cb539e58e53aaef56a6")
        self.assertLessEqual(P.SOURCE_BYTES,P.MAX_0P700_PHYSICAL_BYTES)

    def test_family_diagnostics_do_not_mix_with_paid_quality(self):
        self.assertEqual(len(P.ARMS),6)
        self.assertTrue(all(arm.startswith("fp32_") for arm in P.ARMS))
        self.assertEqual(len(set(P.ARMS)),len(P.ARMS))
        self.assertEqual(P.ARMS[0],"fp32_embed_head_only")
        self.assertIn("fp32_transformer_only",P.ARMS)
        self.assertEqual(len(P.QA_IDS),8)
        self.assertEqual(len(set(P.QA_IDS)),8)
        for a,b in ((0,24),(24,48),(48,72),(72,96)):
            self.assertEqual(sum(a<=i<b for i in P.QA_IDS),2)

    def test_native_oracle_uses_real_unmodified_source_only(self):
        self.assertEqual(len(P.NATIVE_LINEAR_PATHS),3)
        self.assertIn("model.layers.0.self_attn.q_proj.weight",P.NATIVE_LINEAR_PATHS)
        self.assertIn("model.layers.31.mlp.gate_proj.weight",P.NATIVE_LINEAR_PATHS)
        self.assertTrue(all(isinstance(x,int) and x>=0 for x in P.NATIVE_EMBED_ROWS))
        self.assertEqual(P.TOKEN_CAP,32)
        self.assertEqual(len(P.FROZEN_GRADER_SHA),64)
        self.assertEqual(P.FROZEN_V194_RUN,42097753160)
        self.assertEqual(P.FROZEN_ORIGINAL_QA_RUN,42095901870)

if __name__=="__main__":
    unittest.main()
