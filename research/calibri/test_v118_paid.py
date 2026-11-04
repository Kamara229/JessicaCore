"""No 20.87-bpp toy replacement: pin genuine source and pay every byte."""
import hashlib,unittest
from unittest.mock import patch
import numpy as np
from v118.archive import pack,Paid
from v118.protocol import (SOURCE_BYTES,TOTAL_UNIQUE,MAX_SOURCE_BYTES,
                           roles,factor_data_bytes,FIT,VALID)
class Guard(unittest.TestCase):
    def test_structural_source_budget_and_corruption(self):
        dummy=bytes(SOURCE_BYTES)
        lock=hashlib.sha256(dummy).hexdigest()
        with patch("v117.archive.ORIGINAL_CB59_SHA",lock):
            raw=pack(dummy)
            a=Paid(raw)
            self.assertEqual(a.base,dummy)
            self.assertEqual(len(a.layout),60)
            self.assertEqual(len(a.data),factor_data_bytes())
            self.assertLessEqual(len(raw),MAX_SOURCE_BYTES)
            self.assertLess(8*len(raw)/TOTAL_UNIQUE,.7)
            self.assertEqual(a.head()[0].shape,(49152,2))
            broken=bytearray(raw);broken[-60]^=17
            with self.assertRaises(ValueError):Paid(broken)
    def test_unseen_gold_answers_disjoint_and_schema(self):
        self.assertEqual(len(FIT),34)
        self.assertEqual(len(VALID),12)
        self.assertFalse({x for x,_ in FIT}&{x for x,_ in VALID})
        self.assertEqual(len(roles()),60)
if __name__=="__main__":unittest.main()
