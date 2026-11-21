"""S119c synthetic contract tests: no teacher, checkpoint or CPU needed."""
import json
import unittest
import zlib

from research.calibri_nra3.test_s119b_S194_S115_bpp import synthetic_safetensors
from research.calibri_nra3.s119c_real_S115_FP16_reversible_weight_entropy import (
    envelope,decode_envelope,tensor_inventory,
)
from research.calibri_nra3.s119a_hard_four_goals import GateFailure


class WeightEntropyResearchContracts(unittest.TestCase):
    def test_roundtrip_multiple_lossless_codecs_full_original_bytes(self):
        payload=synthetic_safetensors()
        for codec in ("zlib1","zlib6","zlib9","lzma6"):
            with self.subTest(codec=codec):
                packed=envelope(payload,codec)
                self.assertEqual(decode_envelope(packed),payload)

    def test_tensor_partitions_are_complete(self):
        payload=synthetic_safetensors()
        rows=tensor_inventory(payload)
        self.assertEqual(sum(x["FP16_count"] for x in rows),53751)
        self.assertEqual(len(rows),8)
        self.assertTrue(any("structural_trust_gate" in x["name"] for x in rows))

    def test_truncated_or_extra_bytes_fail(self):
        blob=envelope(synthetic_safetensors(),"zlib9")
        with self.assertRaises((GateFailure,zlib.error,EOFError)):
            decode_envelope(blob[:-4])
        with self.assertRaises(GateFailure):
            decode_envelope(blob+b"garbage")

    def test_wrong_encoded_sha_fails(self):
        blob=bytearray(envelope(synthetic_safetensors(),"zlib6"))
        # SHA lies immediately after 9+1+5 kind bytes and 8 length bytes
        off=9+1+5+8
        blob[off]^=1
        with self.assertRaises(GateFailure):
            decode_envelope(bytes(blob))

    def test_no_inference_or_model_architecture_mutations(self):
        import inspect
        from research.calibri_nra3 import s119c_real_S115_FP16_reversible_weight_entropy as stage
        source=inspect.getsource(stage.evaluate)
        self.assertIn("candidate_accepted",source)
        self.assertIn("SynTagRusB_sealed_GOLD_NEVER_IMPORTED",source)
        self.assertNotIn("torch.",source)
        self.assertNotIn("optimizer",source)


if __name__=="__main__":
    unittest.main()
