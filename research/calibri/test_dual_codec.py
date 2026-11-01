"""Deterministic correctness and corruption tests for 2-stage CALIBRI."""
import unittest
import numpy as np
from dual_codec import (
    StaticPairDictionary, encode_grouped, decode_grouped, read_one_row,
    stable_fp64_ffn_equivalence, original_qmoe_like_bytes)

class DualCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dictionary=StaticPairDictionary()

    def test_exact_all_modes_and_seek(self):
        rng=np.random.default_rng(20261101)
        for nrows,cols in [(1,32),(3,96),(9,128),(17,256)]:
            for pz in [0.,.5,.885,1.]:
                q=rng.choice([-1,0,1],size=(nrows,cols),
                             p=[(1-pz)/2,pz,(1-pz)/2]).astype(np.int8)
                words,off=self.dictionary.encode(q)
                self.assertTrue(np.array_equal(q,self.dictionary.decode(words,off,cols)))
                for group in (1,4,16,64):
                    for selector in ("raw","zlib","planes","tier8","adaptive"):
                        blob,_=encode_grouped(words,off,cols,group,selector)
                        restored,ro,co=decode_grouped(blob)
                        self.assertEqual(co,cols)
                        self.assertTrue(np.array_equal(words,restored))
                        self.assertTrue(np.array_equal(off,ro))
                        self.assertTrue(np.array_equal(q,self.dictionary.decode(restored,ro,co)))
                        for r in (0,nrows-1):
                            self.assertTrue(np.array_equal(
                                words[off[r]:off[r+1]],read_one_row(blob,r)))

    def test_corruption_and_input_validation(self):
        q=np.zeros((4,64),dtype=np.int8)
        words,off=self.dictionary.encode(q)
        blob,_=encode_grouped(words,off,64,2)
        with self.assertRaises(ValueError):
            decode_grouped(blob[:-1])
        with self.assertRaises(ValueError):
            decode_grouped(b"NOPE"+blob[4:])
        with self.assertRaises(ValueError):
            self.dictionary.encode(np.zeros((3,65),dtype=np.int8))
        with self.assertRaises(IndexError):
            read_one_row(blob,5)
        self.assertGreater(original_qmoe_like_bytes(words,off,4),0)

    def test_paired_relu_permutation_equivalence(self):
        rng=np.random.default_rng(444)
        wi=rng.normal(size=(72,24))
        wo=rng.normal(size=(14,72))
        permutation=rng.permutation(72)
        rel,ma=stable_fp64_ffn_equivalence(wi,wo,permutation)
        self.assertLess(rel,1e-12)
        self.assertLess(ma,1e-11)

if __name__=="__main__":unittest.main()
