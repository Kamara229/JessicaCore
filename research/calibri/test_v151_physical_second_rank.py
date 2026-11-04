"""Synthetic coded-bit invariants and FIT rank2 no-test-leak checks."""
import unittest
import numpy as np

from v151.codec import (pack3,unpack3,EXTRA_BYTES,PROJECTION_LEN,BITPLANE_LEN)
from v151.protocol import (check_protocol,PINNED_CB149_SHA,NEW_VALID_DOMAINS,
    WIKI_NEW_START,NEWS_NEW_START,NEWS_NEW_END,
    RANK2_POWER_ITERS,RANK2_RIDGE,RANK2_OUTLIER_QUANTILE)
from v151.fit import fit_second_rank,fit_strict
from v117.protocol import MAX_SOURCE_BYTES,TOTAL_UNIQUE


class CompletePaidRank2Tests(unittest.TestCase):
    def test_real_entire_prefix_plus_int3_suffix_below_limit(self):
        self.assertTrue(check_protocol())
        self.assertEqual(PROJECTION_LEN,1152)
        self.assertEqual(BITPLANE_LEN,18432)
        self.assertEqual(EXTRA_BYTES,19642)
        size=11747745+EXTRA_BYTES
        self.assertEqual(size,11767387)
        self.assertLess(size,MAX_SOURCE_BYTES)
        self.assertLess(8*size/TOTAL_UNIQUE,.7)
        self.assertEqual(len(PINNED_CB149_SHA),64)
        self.assertEqual(NEW_VALID_DOMAINS,
                         ("wiki_second_rank_valid","news_second_rank_valid"))
        self.assertGreater(WIKI_NEW_START,31232)
        self.assertGreater(NEWS_NEW_START,10400)
        self.assertGreater(NEWS_NEW_END,NEWS_NEW_START)

    def test_nibble_vs_true_three_bit_output_49152_codes(self):
        codes=((np.arange(49152,dtype=np.int32)%7)-3).astype(np.int8)
        data=pack3(codes)
        self.assertEqual(len(data),18432)
        np.testing.assert_array_equal(unpack3(data),codes)
        with self.assertRaises(ValueError):
            pack3(np.full(49152,-4,dtype=np.int8))
        with self.assertRaises(ValueError):
            pack3(np.full(49152,4,dtype=np.int8))
        with self.assertRaises(ValueError):
            unpack3(bytes(10))
        with self.assertRaises(ValueError):
            unpack3(bytes([0xff])*18432)

    def test_second_rank_regressor_fits_only_synthetic_training_pairs(self):
        rng=np.random.default_rng(151)
        hidden=rng.normal(size=(96,8)).astype(np.float32)
        out=rng.normal(size=32).astype(np.float32)
        student=rng.normal(size=(96,32)).astype(np.float32)*.1
        true=student+hidden[:,2,None]*out[None,:]
        u,b,q,scale,report=fit_second_rank(hidden,true,student)
        self.assertEqual(u.dtype,np.float16)
        self.assertEqual(q.dtype,np.int8)
        self.assertTrue(-3<=int(q.min())<=int(q.max())<=3)
        self.assertEqual(q.shape,(32,))
        self.assertGreater(float(scale),0.)
        self.assertTrue(report["FIT_MSE_better"])
        self.assertEqual(report["fit_tokens"],96)
        u2,b2,q2,step2,_=fit_second_rank(hidden,true,student)
        np.testing.assert_array_equal(u,u2)
        np.testing.assert_array_equal(q,q2)
        self.assertEqual(b,b2)
        self.assertEqual(step2,scale)

    def test_bad_training_geometry_and_guarded_fit_configuration(self):
        with self.assertRaises(ValueError):
            fit_second_rank(np.ones((20,8)),
                            np.ones((19,32)),np.ones((19,32)))
        with self.assertRaises(ValueError):
            fit_strict(np.ones((64,8)),
                       np.ones((64,32)),np.ones((64,32)))
        self.assertEqual(
            (RANK2_POWER_ITERS,RANK2_RIDGE,RANK2_OUTLIER_QUANTILE),
            (8,.2,.997))


if __name__=="__main__":
    unittest.main()
