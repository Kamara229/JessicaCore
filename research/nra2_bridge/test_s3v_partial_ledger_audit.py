"""Synthetic-only tests for S3V partial-ledger fail-closed gates.

No model, artifacts, labels, network or GitHub Actions required.
Run locally: python -m unittest research.nra2_bridge.test_s3v_partial_ledger_audit
"""
import hashlib
import json
import unittest

from research.nra2_bridge.s3v_partial_ledger_audit import audit, STAGE, S3T_SHA, S3O_SHA


def fixture():
    result = {
        "stage": STAGE,
        "S3T_prior_commitment_sha256": S3T_SHA,
        "S3O_frozen_adapter_sha256": S3O_SHA,
        "original_616_S194_J7_hashes_preserved_by_S3U": True,
        "original_584_retained_full_EOS_unchanged": True,
        "candidate_accepted": False,
        "output_gate": 0,
        "new_training_or_parameter_tuning": False,
        "all_and_family_yes_no_accuracy": {
            "all": {"count": 64, "yes": 32, "no": 32,
                    "soft_correct": 64, "hard_correct": 64,
                    "correct_source": 64, "correct_destination": 64},
            "true_yes": {"count": 32, "yes": 32, "no": 0, "soft_correct": 32, "hard_correct": 32,
                         "correct_source": 32, "correct_destination": 32},
            "true_no": {"count": 32, "yes": 0, "no": 32, "soft_correct": 32, "hard_correct": 32,
                        "correct_source": 32, "correct_destination": 32},
        },
        "disagreement_or_error_cases": [],
    }
    # Exact S3T four-family balance: 16 per family, 8 yes and 8 no.
    m = result["all_and_family_yes_no_accuracy"]
    for family in ("chain", "fork", "cycle_sink", "diamond"):
        m["family_" + family] = {
            "count": 16, "yes": 8, "no": 8,
            "soft_correct": 16, "hard_correct": 16,
            "correct_source": 16, "correct_destination": 16}
        for truth in ("yes", "no"):
            m["family_" + family + "_" + truth] = {
                "count": 8, "yes": 8 if truth == "yes" else 0,
                "no": 8 if truth == "no" else 0,
                "soft_correct": 8, "hard_correct": 8,
                "correct_source": 8, "correct_destination": 8}
    return result


def checked(payload):
    raw = json.dumps(payload, allow_nan=False).encode("utf-8")
    return audit(raw, hashlib.sha256(raw).hexdigest())


class PartialLedgerTests(unittest.TestCase):
    def test_missing_family_partition_rejected(self):
        payload = fixture()
        del payload["all_and_family_yes_no_accuracy"]["family_diamond_no"]
        with self.assertRaisesRegex(ValueError, "family/truth partition"):
            checked(payload)

    def test_family_aggregate_drift_rejected(self):
        payload = fixture()
        payload["all_and_family_yes_no_accuracy"]["family_chain"]["correct_source"] = 15
        with self.assertRaisesRegex(ValueError, "family yes/no aggregates"):
            checked(payload)

    def test_family_truth_partition_drift_rejected(self):
        payload = fixture()
        payload["all_and_family_yes_no_accuracy"]["family_fork_yes"]["yes"] = 7
        with self.assertRaisesRegex(ValueError, "family truth balance"):
            checked(payload)

    def test_empty_sparse_rows_remain_unknown(self):
        result = checked(fixture())
        self.assertEqual(result["omitted"], 64)
        self.assertEqual(result["source_bounds"]["lower"], 0)
        self.assertEqual(result["source_bounds"]["upper"], 1)
        self.assertFalse(result["full_64_row_ledger_verified"])
        self.assertEqual(result["output_gate"], 0)

    def test_sparse_role_aggregate_outside_bounds_rejected(self):
        payload = fixture()
        payload["disagreement_or_error_cases"] = [{
            "index": 0, "soft_correct": False, "hard_correct": True,
            "source_correct": False, "target_correct": False,
            "prompt_sha256": "a" * 64, "family": "chain", "gold_yes": True,
        }]
        metrics = payload["all_and_family_yes_no_accuracy"]["all"]
        metrics["soft_correct"] = 63
        payload["all_and_family_yes_no_accuracy"]["true_yes"]["soft_correct"] = 31
        # With one observed incorrect source role, aggregate 64 is impossible.
        with self.assertRaises(ValueError):
            checked(payload)

    def test_duplicate_sparse_index_rejected(self):
        payload = fixture()
        row = {"index": 0, "soft_correct": False, "hard_correct": False,
               "source_correct": False, "target_correct": False,
               "prompt_sha256": "b" * 64, "family": "chain", "gold_yes": True}
        payload["disagreement_or_error_cases"] = [row, dict(row)]
        with self.assertRaises(ValueError):
            checked(payload)

    def test_both_correct_sparse_row_rejected(self):
        # S3V only records incorrect answers or soft/hard disagreement.
        # With a fixed binary truth, two correct answers cannot disagree.
        payload = fixture()
        payload["disagreement_or_error_cases"] = [{
            "index": 0, "soft_correct": True, "hard_correct": True,
            "source_correct": True, "target_correct": True,
            "prompt_sha256": "c" * 64, "family": "chain", "gold_yes": True,
        }]
        with self.assertRaises(ValueError):
            checked(payload)

    def test_producer_float_zero_gate_accepted(self):
        payload = fixture()
        payload["output_gate"] = 0.0
        result = checked(payload)
        self.assertEqual(result["output_gate"], 0)
        self.assertFalse(result["candidate_accepted"])

    def test_boolean_gate_rejected(self):
        payload = fixture()
        payload["output_gate"] = False
        with self.assertRaises(ValueError):
            checked(payload)

    def test_nonfinite_rejected(self):
        raw = json.dumps(fixture()).replace('"output_gate": 0', '"output_gate": NaN').encode()
        with self.assertRaises(ValueError):
            audit(raw, hashlib.sha256(raw).hexdigest())

    def test_exponent_overflow_rejected(self):
        # Valid JSON numeric syntax can overflow Python float to Infinity.
        raw = json.dumps(fixture()).replace('"output_gate": 0',
                                            '"output_gate": 1e999').encode()
        with self.assertRaises(ValueError):
            audit(raw, hashlib.sha256(raw).hexdigest())

    def test_duplicate_key_rejected(self):
        raw = json.dumps(fixture()).replace('"output_gate": 0',
                                            '"output_gate": 0, "output_gate": 0').encode()
        with self.assertRaises(ValueError):
            audit(raw, hashlib.sha256(raw).hexdigest())

    def test_physical_sha_required(self):
        raw = json.dumps(fixture()).encode()
        with self.assertRaises(ValueError):
            audit(raw, "0" * 64)

    def test_inconsistent_partition_rejected(self):
        payload = fixture()
        payload["all_and_family_yes_no_accuracy"]["true_yes"]["correct_source"] = 31
        with self.assertRaises(ValueError):
            checked(payload)


    def test_sparse_wrong_family_truth_aggregate_rejected(self):
        payload = fixture()
        payload["disagreement_or_error_cases"] = [{
            "index": 0, "family": "fork", "gold_yes": False,
            "soft_correct": False, "hard_correct": True,
            "source_correct": True, "target_correct": True,
            "prompt_sha256": "e" * 64,
        }]
        slices = payload["all_and_family_yes_no_accuracy"]
        # Aggregate says the sole soft failure belongs to chain/yes.
        for key in ("all", "true_yes", "family_chain", "family_chain_yes"):
            slices[key]["soft_correct"] -= 1
        with self.assertRaisesRegex(ValueError, "sparse family/truth aggregate mismatch"):
            checked(payload)

    def test_coherent_sparse_error_keeps_omitted_roles_unknown(self):
        payload = fixture()
        payload["disagreement_or_error_cases"] = [{
            "index": 0, "soft_correct": False, "hard_correct": True,
            "source_correct": False, "target_correct": True,
            "prompt_sha256": "d" * 64, "family": "chain", "gold_yes": True,
        }]
        slices = payload["all_and_family_yes_no_accuracy"]
        # One yes/chain example is wrong for soft and source only.
        for key in ("all", "true_yes", "family_chain", "family_chain_yes"):
            slices[key]["soft_correct"] -= 1
            slices[key]["correct_source"] -= 1
        result = checked(payload)
        self.assertEqual(result["recorded"], 1)
        self.assertEqual(result["omitted"], 63)
        self.assertEqual(result["source_bounds"]["lower"], 0)
        self.assertEqual(result["source_bounds"]["upper"], 63 / 64)
        self.assertEqual(result["target_bounds"]["lower"], 1 / 64)
        self.assertEqual(result["target_bounds"]["upper"], 1)
        self.assertFalse(result["full_64_row_ledger_verified"])
        self.assertFalse(result["independent_source_j7_digest_recomputed"])
        self.assertEqual(result["output_gate"], 0)


if __name__ == "__main__":
    unittest.main()
