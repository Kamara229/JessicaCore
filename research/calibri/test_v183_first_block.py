"""v183 predeclared single-operation counterfactual invariants, no model TEST."""
import unittest
from v183.protocol import PROJECTIONS,OPERATOR_SET
from v183.run import original_name,summary
from v182.protocol import FROZEN_CASE_IDS,MODEL_SOURCE_SHA256

class FirstBlockSourceRegression(unittest.TestCase):
    def test_all_interventions_frozen_and_nonoverlapping(self):
        self.assertEqual(FROZEN_CASE_IDS,(0,16,64,80))
        self.assertEqual(len(PROJECTIONS),7)
        self.assertEqual(len(OPERATOR_SET),11)
        self.assertEqual(OPERATOR_SET["V1_BODY_ONLY"],())
        self.assertEqual(len(OPERATOR_SET["FP32_ALL_ATTENTION0"]),4)
        self.assertEqual(len(OPERATOR_SET["FP32_ALL_MLP0"]),3)
        self.assertEqual(len(OPERATOR_SET["FP32_ALL_BLOCK0"]),7)
        self.assertEqual(len({original_name(x) for x in PROJECTIONS}),7)
        self.assertEqual(len(MODEL_SOURCE_SHA256),64)
        with self.assertRaises(ValueError):
            original_name("model.layers.1.q_proj")

    def test_real_ranker_keeps_negative_regressions(self):
        result={}
        for name in OPERATOR_SET:
            if name=="V1_BODY_ONLY":err=0.4
            elif name=="FP32_ALL_BLOCK0":err=0.
            elif name=="FP32_ALL_MLP0":err=0.1
            elif name=="FP32_Q0":err=0.6  # A negative intervention
            else:err=0.3
            result[name]=[{"block0":{"relative_L2_to_reference":err},
                "next_logits":{"relative_L2_to_reference":err+.01},
                "same_original_first_token":name=="FP32_ALL_BLOCK0"}
                for _ in FROZEN_CASE_IDS]
        metrics=summary(result)
        self.assertEqual(metrics["ranked_firstblock_recovery"][0][0],"FP32_ALL_BLOCK0")
        self.assertAlmostEqual(metrics["statistics"]["FP32_Q0"]
            ["firstblock_error_reduction_fraction"],-.5)
        self.assertEqual(metrics["statistics"]["FP32_ALL_MLP0"]
            ["first_token_matches_original"],0)
        self.assertAlmostEqual(metrics["baseline_first_block_error"],0.4)

    def test_requires_exact_untampered_full_block_rescue(self):
        result={name:[{"block0":{"relative_L2_to_reference":.4},
            "next_logits":{"relative_L2_to_reference":.4},
            "same_original_first_token":False}
            for _ in FROZEN_CASE_IDS] for name in OPERATOR_SET}
        with self.assertRaises(AssertionError):
            summary(result)

if __name__=="__main__":
    unittest.main()
