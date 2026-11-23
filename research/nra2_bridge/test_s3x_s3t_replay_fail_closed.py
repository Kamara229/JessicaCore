"""Regression: S3T replay entry points remain fail-closed without fresh native J7.

Pure synthetic inputs only. No GitHub artifacts, weights, oracle, or model calls.
"""
import unittest
from unittest.mock import patch

from research.nra2_bridge import s3x_s3t_commitment_replay_gate as commitment
from research.nra2_bridge import s3x_s3t_zip_replay_gate as archive


class TestS3TReplayFailClosed(unittest.TestCase):
    def test_commitment_replay_rejects_unattested_bytes(self):
        with self.assertRaises(ValueError):
            commitment.verify(b"not an original S3T artifact")

    def test_zip_replay_rejects_unattested_bytes(self):
        with self.assertRaises(ValueError):
            archive.verify_zip(b"not an original S3T ZIP")

    def test_commitment_replay_stays_blocked_even_after_physical_preflight(self):
        with patch.object(commitment, "verify_physical_only", return_value={
            "physical_json_sha_verified": True,
            "original_616_j7_verified": False,
            "output_gate": 0,
        }), patch.object(commitment.importlib, "import_module",
                         side_effect=AssertionError("generator must not be imported")):
            with self.assertRaisesRegex(RuntimeError, "616 J7 attestation"):
                commitment.verify(b"synthetic placeholder")

    def test_zip_replay_stays_blocked_even_after_physical_preflight(self):
        with patch.object(archive, "verify_zip_preflight", return_value={
            "physical_zip_sha256_verified": True,
            "original_616_j7_verified": False,
            "output_gate": 0,
        }):
            with self.assertRaisesRegex(RuntimeError, "616 J7 attestation"):
                archive.verify_zip(b"synthetic placeholder")


if __name__ == "__main__":
    unittest.main()
