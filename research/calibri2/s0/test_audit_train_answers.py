"""TRAIN-only synthetic regression tests; no model, held-out data, or CI required."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("train_audit", Path(__file__).with_name("audit_train_answers.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def payload(left, right=None):
    gold = [11, 12, 99]
    def row(side, generated):
        return {"domain": "synthetic_train", "id": "pair1", "side": side,
                "generated_ids": generated, "gold_ids": gold,
                "full_with_EOS": generated == gold}
    return {"TRAIN_only_no_QA_VALID_checkpoint_selection": True,
            "milestones": [{"step": 0, "actual_answers":
                            [row("left", left), row("right", right or gold) ]}]}

class TrainAuditTests(unittest.TestCase):
    def test_exact_pair(self):
        r = module.audit(payload([11,12,99]))["milestones"][0]
        self.assertEqual(r["full_EOS"], 2)
        self.assertEqual(r["opposing_pair_retention"]["both"], 1)

    def test_early_eos(self):
        r = module.audit(payload([11,99]))["milestones"][0]
        self.assertEqual(r["nonexclusive_error_flags"]["premature_EOS"], 1)

    def test_trailing_after_correct_eos_is_not_premature(self):
        r = module.audit(payload([11,12,99,7]))["milestones"][0]
        self.assertNotIn("premature_EOS", r["nonexclusive_error_flags"])
        self.assertEqual(r["nonexclusive_error_flags"]["EOS_at_wrong_position"], 0)
        self.assertEqual(r["full_EOS"], 1)

    def test_missing_eos(self):
        r = module.audit(payload([11,12,7]))["milestones"][0]
        self.assertEqual(r["nonexclusive_error_flags"]["expected_EOS_token_absent"], 1)
        self.assertEqual(r["full_EOS"], 1)

    def test_eos_at_wrong_position(self):
        r = module.audit(payload([11,99,7]))["milestones"][0]
        self.assertEqual(r["nonexclusive_error_flags"]["premature_EOS"], 1)
        self.assertEqual(r["nonexclusive_error_flags"]["EOS_at_wrong_position"], 1)

    def test_adjacent_bigram_repetition(self):
        r = module.audit(payload([11,12,11,12,99]))["milestones"][0]
        self.assertEqual(r["nonexclusive_error_flags"]["adjacent_bigram_repetition"], 1)

    def test_missing_opposite_side_rejected(self):
        p = payload([11,12,99])
        p["milestones"][0]["actual_answers"].pop()
        with self.assertRaises(ValueError):
            module.audit(p)

    def test_inconsistent_flag_rejected(self):
        p = payload([11,12,99])
        p["milestones"][0]["actual_answers"][0]["full_with_EOS"] = False
        with self.assertRaises(ValueError):
            module.audit(p)

    def test_duplicate_side_rejected(self):
        p = payload([11,12,99])
        p["milestones"][0]["actual_answers"][1]["side"] = "left"
        with self.assertRaises(ValueError):
            module.audit(p)

if __name__ == "__main__":
    unittest.main()
