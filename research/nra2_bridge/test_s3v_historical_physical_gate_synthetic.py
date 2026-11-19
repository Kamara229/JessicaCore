"""Synthetic, research-only regression tests; no model, gold labels, or external artifacts."""
import hashlib
import io
import json
import unittest
import zipfile

from research.nra2_bridge.s3v_historical_physical_gate import verify_physical_gate


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(raw=b'{"output_gate":0.0,"candidate_accepted":false}', member="result.json"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(member, raw)
    zipped = buffer.getvalue()
    manifest = {
        "manifest_reviewed": True,  # synthetic flag, NOT independent attestation
        "run_id": 1, "run_attempt": 1, "workflow_path": ".github/workflows/research-only.yml",
        "head_sha": "a" * 40, "artifact_id": 1, "artifact_name": "synthetic",
        "zip_sha256": sha(zipped), "json_member_path": member,
        "json_sha256": sha(raw),
    }
    return raw, json.dumps(manifest).encode(), zipped


class HistoricalPhysicalGateTests(unittest.TestCase):
    def test_synthetic_valid_still_blocked(self):
        result = verify_physical_gate(*fixture())
        self.assertEqual(result["output_gate"], 0)
        self.assertFalse(result["full_64_row_ledger_verified"])
        self.assertFalse(result["independent_source_j7_digest_recomputed"])
        self.assertFalse(result["candidate_accepted"])
        self.assertTrue(result["status"].startswith("BLOCKED_"))

    def test_float_overflow_rejected(self):
        with self.assertRaises(ValueError):
            verify_physical_gate(*fixture(b'{"output_gate":0.0,"candidate_accepted":false,"x":1e999}'))

    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(ValueError):
            verify_physical_gate(*fixture(b'{"output_gate":0.0,"output_gate":0.0,"candidate_accepted":false}'))

    def test_single_backslash_member_rejected(self):
        with self.assertRaises(ValueError):
            verify_physical_gate(*fixture(member="folder\\result.json"))

    def test_nonpositive_identifiers_rejected(self):
        for key in ("run_id", "run_attempt", "artifact_id"):
            with self.subTest(key=key):
                raw, manifest_raw, zipped = fixture()
                manifest = json.loads(manifest_raw)
                manifest[key] = 0
                with self.assertRaises(ValueError):
                    verify_physical_gate(raw, json.dumps(manifest).encode(), zipped)

    def test_zip_member_must_equal_original_bytes(self):
        raw, manifest, zipped = fixture()
        modified = raw + b" "
        with self.assertRaises(ValueError):
            verify_physical_gate(modified, manifest, zipped)

    def test_integer_historical_gate_rejected(self):
        with self.assertRaises(ValueError):
            verify_physical_gate(*fixture(b'{"output_gate":0,"candidate_accepted":false}'))


    def test_manifest_review_flag_cannot_be_omitted(self):
        raw, manifest_raw, zipped = fixture()
        manifest = json.loads(manifest_raw)
        for value in (False, None, "true", 1):
            with self.subTest(value=value):
                manifest["manifest_reviewed"] = value
                with self.assertRaises(ValueError):
                    verify_physical_gate(raw, json.dumps(manifest).encode(), zipped)

    def test_boolean_identifiers_rejected(self):
        raw, manifest_raw, zipped = fixture()
        for key in ("run_id", "run_attempt", "artifact_id"):
            with self.subTest(key=key):
                manifest = json.loads(manifest_raw)
                manifest[key] = True
                with self.assertRaises(ValueError):
                    verify_physical_gate(raw, json.dumps(manifest).encode(), zipped)

    def test_zip_hash_mismatch_rejected(self):
        raw, manifest, zipped = fixture()
        with self.assertRaises(ValueError):
            verify_physical_gate(raw, manifest, zipped + b"tampered")

    def test_missing_or_wrong_zip_member_rejected(self):
        raw, manifest_raw, zipped = fixture()
        manifest = json.loads(manifest_raw)
        manifest["json_member_path"] = "other.json"
        with self.assertRaises(ValueError):
            verify_physical_gate(raw, json.dumps(manifest).encode(), zipped)

    def test_candidate_accepted_true_rejected(self):
        raw = b'{"output_gate":0.0,"candidate_accepted":true}'
        with self.assertRaises(ValueError):
            verify_physical_gate(*fixture(raw))

    def test_unsafe_auxiliary_zip_entries_rejected(self):
        for auxiliary in ("../escape.txt", "/absolute.txt", "folder\\\\escape.txt"):
            with self.subTest(auxiliary=auxiliary):
                raw = b'{"output_gate":0.0,"candidate_accepted":false}'
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("result.json", raw)
                    archive.writestr(auxiliary, b"unused")
                zipped = buf.getvalue()
                manifest = json.loads(fixture()[1])
                manifest["zip_sha256"] = sha(zipped)
                with self.assertRaises(ValueError):
                    verify_physical_gate(raw, json.dumps(manifest).encode(), zipped)

    def test_windows_drive_rooted_paths_rejected(self):
        for path in ("C:/result.json", "D:relative.json"):
            with self.subTest(member=path):
                with self.assertRaises(ValueError):
                    verify_physical_gate(*fixture(member=path))
            with self.subTest(auxiliary=path):
                raw, manifest_raw, _ = fixture()
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("result.json", raw)
                    archive.writestr(path, b"unused")
                zipped = buf.getvalue()
                manifest = json.loads(manifest_raw)
                manifest["zip_sha256"] = sha(zipped)
                with self.assertRaises(ValueError):
                    verify_physical_gate(raw, json.dumps(manifest).encode(), zipped)

    def test_nonfinite_json_constant_rejected(self):
        raw = b'{"output_gate":0.0,"candidate_accepted":false,"x":NaN}'
        with self.assertRaises(ValueError):
            verify_physical_gate(*fixture(raw))


if __name__ == "__main__":
    unittest.main()
