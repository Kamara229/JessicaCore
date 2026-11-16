"""Synthetic-only tests for S3V partial-ledger fail-closed gates.

No model, artifacts, labels, network or GitHub Actions required.
Run locally: python -m unittest research.nra2_bridge.test_s3v_partial_ledger_audit
"""
import hashlib
import json
import unittest

from research.nra2_bridge.s3v_partial_ledger_audit import audit, STAGE, S3T_SHA, S3O_SHA


def fixture():
    return {
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
            "true_yes": {"count": 32, "soft_correct": 32, "hard_correct": 32,
                         "correct_source": 32, "correct_destination": 32},
            "true_no": {"count": 32, "soft_correct": 32, "hard_correct": 32,
                        "correct_source": 32, "correct_destination": 32},
        },
        "disagreement_or_error_cases": [],
    }


def checked(payload):
    raw = json.dumps(payload, allow_nan=False).encode("utf-8")
    return audit(raw, hashlib.sha256(raw).hexdigest())


class PartialLedgerTests(unittest.TestCase):
    def test_empty_sparse_rows_remain_unknown(self):
        result = checked(fixture())
        self.assertEqual(result["omitted"], 64)
        self.assertEqual(result["source_bounds"]["lower"], 0)
        self.assertEqual(result["source_bounds"]["upper"], 1)
        self.assertFalse(result["full_64_row_ledger_verified"])
        self.assertEqual(result["output_gate"], 0)

    def test_boolean_gate_rejected(self):
        payload = fixture()
        payload["output_gate"] = False
        with self.assertRaises(ValueError):
            checked(payload)

    def test_nonfinite_rejected(self):
        raw = json.dumps(fixture()).replace('"output_gate": 0', '"output_gate": NaN').encode()
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


if __name__ == "__main__":
    unittest.main()
