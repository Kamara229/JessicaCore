"""No full pretrained source in unit tests: deterministic toy fit and paid codes."""
import unittest
import numpy as np
from v149.codec import (pack_signed_int4,unpack_signed_int4,
    EXTRA_BYTES,PROJECTION_BYTES,VECTOR_BYTES)
from v149.fit import fit_rank_one,fit_strict_135m
from v149.protocol import (TRAIN_HIDDEN,VOCAB,CB147_SHA,
                           WIKI_VALID_START,NEWS_VALID_START,
                           VALID_TOKENS,OUTLIER_QUANTILE,RIDGE_FRACTION,
                           POWER_ITERS)
from v117.protocol import MAX_SOURCE_BYTES,TOTAL_UNIQUE


class ContextPaidModel(unittest.TestCase):
    def test_int4_bitpacking_every_vocab_symbol_and_rate_cap(self):
        codes=(np.arange(VOCAB,dtype=np.int32)%15-7).astype(np.int8)
        raw=pack_signed_int4(codes)
        self.assertEqual(len(raw),24576)
        self.assertTrue(np.array_equal(unpack_signed_int4(raw),codes))
        self.assertEqual(PROJECTION_BYTES,1152)
        self.assertLessEqual(11721959+EXTRA_BYTES,MAX_SOURCE_BYTES)
        self.assertLessEqual(8*(11721959+EXTRA_BYTES)/TOTAL_UNIQUE,.7)
        with self.assertRaises(ValueError):
            pack_signed_int4(np.full(VOCAB,-8,dtype=np.int8))
        with self.assertRaises(ValueError):
            unpack_signed_int4(b"truncated")
        with self.assertRaises(ValueError):
            unpack_signed_int4(bytes(24576))

    def test_fit_rank1_tracks_hidden_conditioned_error(self):
        rng=np.random.default_rng(149)
        # Variability across samples, no gold answers, no pretrained model.
        hidden=rng.normal(size=(64,8)).astype(np.float32)
        outvec=rng.normal(size=32).astype(np.float32)
        truth=(hidden[:,0,None]*outvec[None,:]).astype(np.float32)
        ref=truth+rng.normal(scale=.001,size=truth.shape).astype(np.float32)
        approximation=np.zeros_like(ref)
        u,intercept,q,scale,measurement=fit_rank_one(
            hidden,ref,approximation)
        self.assertEqual(u.shape,(8,))
        self.assertEqual(q.shape,(32,))
        self.assertEqual(q.dtype,np.int8)
        self.assertEqual(u.dtype,np.float16)
        self.assertTrue(np.isfinite(u).all())
        self.assertTrue(-7<=int(q.min())<=int(q.max())<=7)
        self.assertTrue(measurement["source_FIT_context_MSE_decreased"])
        r2=fit_rank_one(hidden,ref,approximation)
        self.assertTrue(np.array_equal(u,r2[0]))
        self.assertTrue(np.array_equal(q,r2[2]))
        self.assertEqual(float(scale),float(r2[3]))
        self.assertEqual(float(intercept),float(r2[1]))

    def test_bad_teacher_geometry_and_bad_source_guarded(self):
        with self.assertRaises(ValueError):
            fit_rank_one(np.zeros((8,8)),np.zeros((9,32)),np.zeros((9,32)))
        with self.assertRaises(ValueError):
            fit_strict_135m(np.zeros((32,8)),np.zeros((32,32)),np.zeros((32,32)))
        self.assertEqual((POWER_ITERS,RIDGE_FRACTION,OUTLIER_QUANTILE),
                         (8,.2,.997))
        self.assertGreaterEqual(WIKI_VALID_START,30720)
        self.assertGreaterEqual(NEWS_VALID_START,10300)
        self.assertEqual(VALID_TOKENS,512)
        self.assertEqual(len(CB147_SHA),64)


if __name__=="__main__":
    unittest.main()
