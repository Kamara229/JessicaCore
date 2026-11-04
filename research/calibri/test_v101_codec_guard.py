"""No changes to source: validate roundtrip and exact decoder on small toy head."""
from __future__ import annotations
import unittest
import numpy as np
from types import SimpleNamespace
from direct_codebook_v59 import pack5,unpack5,QuantHandle
from v101.codec import decode_chunk,exact_streamed_audit

class T(unittest.TestCase):
    def test_original_5bit_index_boundary_roundtrip(self):
        ids=np.tile(np.arange(32,dtype=np.uint8),9)
        raw=pack5(ids)
        self.assertTrue(np.array_equal(ids,unpack5(raw,len(ids))))
        self.assertEqual(raw,pack5(unpack5(raw,len(ids))))
    def test_exact_toy_vocabulary_head_and_embedding_rows(self):
        rng=np.random.default_rng(101)
        m,n=16,16
        groups=n//8
        codes=rng.integers(0,32,size=m*groups,dtype=np.uint8)
        scales=np.linspace(.02,.08,m,dtype=np.float32)
        centers=rng.normal(0,1,size=(32,8)).astype(np.float32)
        handle=QuantHandle("lm_head.weight",(m,n),
                          np.frombuffer(pack5(codes),dtype=np.uint8),
                          scales,centers)
        decoded,indices=decode_chunk(handle,0,m)
        self.assertTrue(np.array_equal(indices,codes))
        x=rng.normal(size=(2,n)).astype(np.float32)
        logits=x@decoded.T
        import torch
        report=exact_streamed_audit(handle,torch.tensor(decoded.copy()),x,logits)
        self.assertLess(report["native_C_vs_explicit_Python_decoder_max_abs_logit_diff"],1e-5)
        self.assertLess(report["original_FP32_to_CB59_head_weight_relative_RMSE"],1e-7)
    def test_bad_source_row_order_rejected(self):
        h=QuantHandle("bad",(16,576),np.zeros(2,dtype=np.uint8),
                      np.ones(16,dtype=np.float32),np.zeros((32,8),dtype=np.float32))
        with self.assertRaises(AssertionError):decode_chunk(h,0,1)

if __name__=="__main__":
    unittest.main()
