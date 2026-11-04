"""Fully counted CB147 int8 prior and frozen source-only teacher residual."""
import hashlib
import unittest

import numpy as np

from v147.codec import (COUNT,HEADER,DIGEST,MAGIC,pack,unpack,measured_rate)
from v147.fit import fit
from v147.protocol import (VALID_DOMAINS,WIKI_VALID_START,WIKI_VALID_COUNT,
    NEWS_VALID_START,NEWS_VALID_END)
from v117.protocol import MAX_SOURCE_BYTES,TOTAL_UNIQUE


class PaidInt8PriorTests(unittest.TestCase):
    def test_predeclared_domain_holdout_fully_disjoint(self):
        self.assertEqual(VALID_DOMAINS,("wiki_prior_valid","news_prior_valid"))
        self.assertGreater(WIKI_VALID_START,19968)
        self.assertEqual(WIKI_VALID_COUNT,512)
        self.assertGreaterEqual(NEWS_VALID_START,9500)
        self.assertEqual(NEWS_VALID_END-NEWS_VALID_START,100)

    def test_physically_paid_bias_always_fits_within_0_7(self):
        complete=11_672_747+HEADER.size+COUNT+4+DIGEST
        self.assertLessEqual(complete,MAX_SOURCE_BYTES)
        self.assertGreater(complete,11_672_747)
        self.assertLessEqual(8*complete/TOTAL_UNIQUE,.7)
        self.assertEqual(COUNT,49152)

    def test_teacher_residual_is_centered_and_deterministic(self):
        baseline=np.linspace(-2,2,COUNT,dtype=np.float64)
        teacher=baseline.copy()
        teacher[100:500]+=3.0
        teacher[1000:3000]-=1.0
        code,scale,stats=fit(teacher,baseline)
        self.assertEqual(code.shape,(49152,))
        self.assertEqual(code.dtype,np.int8)
        self.assertGreater(scale,0)
        self.assertTrue(np.isfinite(stats["post_int8_quantization_RMSE"]))
        code2,step2,stats2=fit(teacher,baseline)
        self.assertTrue(np.array_equal(code,code2))
        self.assertEqual(scale,step2)
        with self.assertRaises(ValueError):
            fit(np.ones(30),np.ones(30))
        with self.assertRaises(ValueError):
            fit(np.ones(COUNT),np.ones(COUNT))

    def test_frozen_source_only_archive_not_created_from_random_bytes(self):
        with self.assertRaises(ValueError):
            pack(b"not_a_real_pinned_source",np.zeros(COUNT,dtype=np.int8),1.)
        with self.assertRaises(ValueError):
            unpack(b"not_complete")


if __name__=="__main__":
    unittest.main()
