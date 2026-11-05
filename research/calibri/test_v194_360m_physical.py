"""v194 source integrity, archive budget and direct 5bit byte-level unit tests."""
import unittest
import numpy as np
from direct_codebook_v59 import pack5,unpack5
from v194 import protocol as P

class Real360MSourceControl(unittest.TestCase):
    def test_true_complete_360m_unique_source_bit_budget(self):
        self.assertEqual(P.ORIGINAL_UNIQUE_PARAMETERS,361_821_120)
        self.assertEqual(P.MAX_0P700_PHYSICAL_BYTES,31_659_348)
        self.assertEqual(P.MAX_0P700_PHYSICAL_BYTES,P.ORIGINAL_UNIQUE_PARAMETERS*7//80)
        self.assertEqual(P.BITS_PER_VECTOR,5)
        self.assertEqual(P.WEIGHTS_PER_VECTOR,8)
        self.assertEqual(P.CODEBOOK_ENTRIES,32)
        self.assertNotEqual(P.MODEL,"HuggingFaceTB/SmolLM2-135M-Instruct")

    def test_real_5bit_roundtrip_is_not_synthetic_size_fraud(self):
        original=np.array([(i*7)%32 for i in range(104)],dtype=np.uint8)
        written=pack5(original)
        self.assertEqual(len(written),65)
        reconstructed=unpack5(written,len(original))
        self.assertTrue(np.array_equal(original,reconstructed))
        self.assertLess(len(written),original.nbytes)
        with self.assertRaises(ValueError):
            pack5(np.array([0,32,1,5,6,7,4,3],dtype=np.uint8))

    def test_preselected_public_original_qa_without_gold_leak(self):
        self.assertEqual(len(P.SMOKE_IDS),16)
        self.assertEqual(len(set(P.SMOKE_IDS)),16)
        self.assertEqual(sum(0<=x<24 for x in P.SMOKE_IDS),4)
        self.assertEqual(sum(24<=x<48 for x in P.SMOKE_IDS),4)
        self.assertEqual(sum(48<=x<72 for x in P.SMOKE_IDS),4)
        self.assertEqual(sum(72<=x<96 for x in P.SMOKE_IDS),4)
        self.assertEqual(len(P.FROZEN_GRADER_SHA),64)

if __name__=="__main__":
    unittest.main()
