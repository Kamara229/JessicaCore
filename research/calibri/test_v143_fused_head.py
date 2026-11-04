"""Synthetic independent dense oracle for fused 5+1bit CPU tied output."""
import ctypes
import tempfile
import unittest
from pathlib import Path

import numpy as np
from direct_codebook_v59 import pack5
from v140.codec import pack_bits
from v143.runtime import build


class CombinedNativeKernelTests(unittest.TestCase):
    def test_one_pass_equals_full_decoded_6bit_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            lib=ctypes.CDLL(str(build(Path(temporary)/"fused.so")))
            fn=lib.calibri143_fused_output
            ptr=ctypes.c_void_p
            fn.argtypes=[ptr,ctypes.c_size_t,ptr,ctypes.c_size_t,
                         ptr,ptr,ptr,ptr,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ptr]
            fn.restype=ctypes.c_int
            rng=np.random.default_rng(143)
            rows,dim=128,576
            num=rows*dim//8
            code=rng.integers(0,32,size=num,dtype=np.uint8)
            bit=rng.integers(0,2,size=num,dtype=np.uint8)
            packed=np.frombuffer(pack5(code),dtype=np.uint8).copy()
            one=np.frombuffer(pack_bits(bit),dtype=np.uint8).copy()
            base=rng.normal(scale=.5,size=(32,8)).astype(np.float32)
            delta=rng.normal(scale=.03,size=(32,2,8)).astype(np.float32)
            scales=rng.uniform(.1,1,size=rows).astype(np.float32)
            x=rng.normal(size=(3,dim)).astype(np.float32)
            out=np.empty((len(x),rows),dtype=np.float32)
            p=lambda array:ptr(int(array.ctypes.data))
            self.assertEqual(fn(
                p(packed),len(packed),p(one),len(one),
                p(base),p(delta),p(scales),p(x),
                len(x),rows,dim,p(out)),0)
            weight=(base[code]+delta[code,bit]).reshape(rows,dim)*scales[:,None]
            np.testing.assert_allclose(out,x@weight.T,rtol=2e-4,atol=2e-3)
            wrong=fn(
                p(packed),len(packed)-1,p(one),len(one),
                p(base),p(delta),p(scales),p(x),
                len(x),rows,dim,p(out))
            self.assertEqual(wrong,2)

    def test_wrong_arguments_abort_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            lib=ctypes.CDLL(str(build(Path(temporary)/"fused.so")))
            fn=lib.calibri143_fused_output
            fn.restype=ctypes.c_int
            fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t,
                         ctypes.c_void_p,ctypes.c_size_t,
                         ctypes.c_void_p,ctypes.c_void_p,
                         ctypes.c_void_p,ctypes.c_void_p,
                         ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
            self.assertEqual(fn(None,0,None,0,None,None,None,None,1,100,576,None),1)


if __name__=="__main__":
    unittest.main()
