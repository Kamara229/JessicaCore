"""S144 CI: actual S142 source artifact pin must verify and reject a mutated ZIP."""
from pathlib import Path
import json
import os
import unittest

from calibri2.s144.source_pin import SourcePinError, fetch_artifact, verify_archive


class S144ReadOnlySourcePin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            Path(__file__).with_name("source_manifest.json").read_text()
        )
        cls.archive = fetch_artifact(cls.manifest, os.environ.get("GITHUB_TOKEN", ""))

    def test_actual_s142_fp16_source(self):
        proof = verify_archive(self.archive, self.manifest)
        self.assertTrue(proof["valid"])
        self.assertEqual(proof["old_full"], 362)
        self.assertEqual(proof["new_full"], 17)
        self.assertEqual(proof["physical_456_128_replay"],
                         "attested_by_original_S142_CI_only")
        self.assertFalse(proof["model_accepted"])

    def test_zip_digest_tampering_fails_closed(self):
        with self.assertRaisesRegex(SourcePinError, "ZIP digest"):
            verify_archive(self.archive + b"tampered", self.manifest)

    def test_false_checkpoint_digest_fails_closed(self):
        bad = dict(self.manifest, checkpoint_sha256="0" * 64)
        with self.assertRaisesRegex(SourcePinError, "Checkpoint SHA256"):
            verify_archive(self.archive, bad)


if __name__ == "__main__":
    unittest.main()
