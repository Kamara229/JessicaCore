"""S23 actual trained S22d SHA-locked synthetic-TRAIN read-only audit guards."""
import inspect
import unittest

from research.nra3_s20.s23_real_s22d_40fold_failure_audit import (
    S22D_MODEL_SHA, S22D_ORIGINAL_CI,
    f1, exact_real_report_audit,
)
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    FROZEN_S22_TRAIN256_SHA256,build_s22_train256,splits,PATTERNS,
)


class S23ActualS22dFortyFoldAuditTests(unittest.TestCase):
    def test_real_S22d_success_sha_and_synthetic_train_only_40_folds(self):
        self.assertEqual(S22D_ORIGINAL_CI,44192186916)
        self.assertEqual(
            S22D_MODEL_SHA,
            "bd56293aa6ada2411152f0ac528633315731c39de6e74a759704577defbadcbb")
        self.assertEqual(
            FROZEN_S22_TRAIN256_SHA256,
            "8963e2e8758b1c6a6fbd28eb597a77be1a454d732d0ecac93df6f5a753f6a361")
        self.assertEqual(len(build_s22_train256()),256)
        self.assertEqual(len(splits()),40)
        self.assertEqual(len(PATTERNS),32)

    def test_report_must_fail_closed_unless_complete_sha_guarded_40fold(self):
        invalid={
            "stage":"NRA3_S23_FABRICATED",
            "source_256_synthetic_graph_SHA256":FROZEN_S22_TRAIN256_SHA256,
            "physical_FP16_candidate_SHA256":S22D_MODEL_SHA,
            "all40_fixed_TRAIN_internal_folds":[],
            "output_gate":0,
        }
        with self.assertRaisesRegex(ValueError,"wrong S22d stage"):
            exact_real_report_audit(invalid,S22D_MODEL_SHA)
        invalid["stage"]="NRA3_S22D_REAL43M_GT_NEW256_SYNTHETIC_TRAIN_ONLY_GROUPED40_GRAPH"
        with self.assertRaisesRegex(ValueError,"real FP16 candidate SHA differs"):
            exact_real_report_audit(invalid,"0"*64)
        with self.assertRaisesRegex(ValueError,"original frozen 43M/GT"):
            exact_real_report_audit(invalid,S22D_MODEL_SHA)

    def test_micro_f1_arithmetic_and_no_false_heldout_claim(self):
        self.assertAlmostEqual(f1(106,277,86),0.36869565217391304)
        self.assertAlmostEqual(f1(147,161,45),0.588)
        self.assertAlmostEqual(f1(23,7,9),0.7419354838709677)
        self.assertAlmostEqual(23/32,0.71875)
        self.assertAlmostEqual(31/32,0.96875)
        self.assertEqual(f1(0,0,0),0.)

    def test_no_inference_no_optimizer_no_eval_no_live_source_paths(self):
        import research.nra3_s20.s23_real_s22d_40fold_failure_audit as source
        code=inspect.getsource(source)
        for forbidden in ("optimizer.step(", "loss.backward(",
                          "torch.inference_mode(", "torch.nn",
                          "model.forward(", "model.load_state_dict(",
                          "import requests"):
            self.assertNotIn(forbidden,code)
        self.assertIn("label_level_relation_confusion_NOT_AVAILABLE_without_fresh_fold_predictions",code)
        self.assertIn("not_independent_evaluation",code)
        self.assertIn('"output_gate":0',code)
        self.assertIn('"activation_allowed":False',code)


if __name__=="__main__":
    unittest.main()
