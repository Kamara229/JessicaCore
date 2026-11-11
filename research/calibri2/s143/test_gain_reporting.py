"""S143 CPU-only regression for truthful, fail-closed J7 TRAIN reporting."""
from pathlib import Path
import ast
import unittest
from calibri2.s143.gain_reporting import classify_j7_development_gain as classify


def evidence(**override):
    valid = dict(
        committed=True, before_old_full=361, after_old_full=362,
        before_new_full=17, after_new_full=17,
        before_new_pairs=0, after_new_pairs=0,
        lost_old_full=0, lost_old_first=0,
        lost_new_full=0, lost_new_first=0,
        fp16_reloaded=True, immutable_source=True,
        physical_guard=True, j7_guard=True,
        physical_bpp=0.6568244742134911, cpu_ratio=1.002255673333473,
    )
    valid.update(override)
    return classify(**valid)


class S143HistoricalGainReporting(unittest.TestCase):
    def test_actual_s142_one_historical_gain_is_reported(self):
        p = evidence()
        self.assertTrue(p["safe_development_TRAIN_improvement"])
        self.assertEqual(p["historical_old456_full_delta"], 1)
        self.assertEqual(p["S122_new128_full_delta"], 0)
        self.assertTrue(p["research_only_not_model_acceptance"])
        self.assertTrue(p["independent_unseen_QA_not_run"])

    def test_new_s122_only_gain_is_reported(self):
        p = evidence(after_old_full=361, after_new_full=18)
        self.assertTrue(p["safe_development_TRAIN_improvement"])
        self.assertEqual(p["S122_new128_full_delta"], 1)

    def test_two_sided_pair_only_gain_is_reported(self):
        self.assertTrue(evidence(after_old_full=361, after_new_pairs=1)
                        ["safe_development_TRAIN_improvement"])

    def test_no_gain_never_reports_success(self):
        self.assertFalse(evidence(after_old_full=361)
                         ["safe_development_TRAIN_improvement"])

    def test_rollback_is_not_a_gain_even_with_positive_counts(self):
        self.assertFalse(evidence(committed=False)
                         ["safe_development_TRAIN_improvement"])

    def test_any_previous_answer_loss_rejects_claim(self):
        for damaged in ("lost_old_full", "lost_old_first",
                        "lost_new_full", "lost_new_first"):
            with self.subTest(damaged=damaged):
                self.assertFalse(evidence(**{damaged: 1})
                                 ["safe_development_TRAIN_improvement"])

    def test_physical_and_transaction_guards_required(self):
        for missing in ("fp16_reloaded", "immutable_source",
                        "physical_guard", "j7_guard"):
            with self.subTest(missing=missing):
                self.assertFalse(evidence(**{missing: False})
                                 ["safe_development_TRAIN_improvement"])
        self.assertFalse(evidence(physical_bpp=.70001)
                         ["safe_development_TRAIN_improvement"])
        self.assertFalse(evidence(cpu_ratio=.8999)
                         ["safe_development_TRAIN_improvement"])

    def test_lowered_score_rejects_claim(self):
        self.assertFalse(evidence(after_old_full=360, after_new_full=18)
                         ["safe_development_TRAIN_improvement"])

    def test_invalid_evidence_fails_closed(self):
        for bad in (dict(before_old_full=-1), dict(cpu_ratio=float("nan")),
                    dict(after_old_full=457), dict(committed=1)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    evidence(**bad)

    def test_integrated_training_uses_precommit_old_baseline(self):
        script = (Path(__file__).parents[1] / "s142" /
                  "train_one_j7_dual_cone.py").read_text()
        ast.parse(script)
        self.assertIn("pre_cycle_old_base_full=sum(", script)
        self.assertIn("classify_j7_development_gain(", script)
        self.assertIn('"historical_old456_full_delta"', script)
        self.assertIn('"development_TRAIN_gain_proof":gain_proof', script)
        self.assertIn('"best_safe_development_TRAIN_checkpoint":best_safe', script)
        self.assertIn('not gain_proof["safe_development_TRAIN_improvement"]', script)
        self.assertNotIn("load_heldout", script)
        self.assertNotIn("from calibri2.s118", script)


if __name__ == "__main__":
    unittest.main()
