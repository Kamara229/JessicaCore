"""One-bit CPU lookup performance, packed C ABI and numerical equivalence."""
import ctypes
import tempfile
import unittest
from pathlib import Path
import numpy as np
from direct_codebook_v59 import pack5
from v140.codec import pack_bits
from v140.runtime import compile_onebit
from v141.kernel import compile_lut


class LUTNativeSourceTests(unittest.TestCase):
    def test_C_head_is_same_as_paid_v140_and_dense_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            original_lib=ctypes.CDLL(str(compile_onebit(Path(temp)/"scalar.so")))
            improved_lib=ctypes.CDLL(str(compile_lut(Path(temp)/"lut.so")))
            ptr=ctypes.c_void_p
            types=[ptr,ctypes.c_size_t,ptr,ctypes.c_size_t,ptr,ptr,ptr,
                   ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ptr]
            for lib,prefix in ((original_lib,"calibri140"),(improved_lib,"calibri141")):
                for suffix in ("embedding_delta","output_delta"):
                    fn=getattr(lib,prefix+"_"+suffix)
                    fn.argtypes=types
                    fn.restype=ctypes.c_int

            rng=np.random.default_rng(141)
            rows,dim=112,576
            n=rows*dim//8
            five=rng.integers(0,32,size=n,dtype=np.uint8)
            bits=rng.integers(0,2,size=n,dtype=np.uint8)
            packed=np.frombuffer(pack5(five),dtype=np.uint8).copy()
            binary=np.frombuffer(pack_bits(bits),dtype=np.uint8).copy()
            dictionary=rng.normal(scale=.1,size=(32,2,8)).astype(np.float32)
            scales=rng.uniform(.3,2,size=rows).astype(np.float32)
            sample=rng.normal(size=(4,dim)).astype(np.float32)
            ids=np.array([0,23,110],dtype=np.int64)
            p=lambda a:ptr(int(a.ctypes.data))
            common=[p(packed),len(packed),p(binary),len(binary),
                    p(dictionary),p(scales)]
            recovered=dictionary[five,bits].reshape(rows,dim)*scales[:,None]
            answer=sample@recovered.T
            for lib,prefix in ((original_lib,"calibri140"),(improved_lib,"calibri141")):
                out=np.empty_like(answer,dtype=np.float32)
                fn=getattr(lib,prefix+"_output_delta")
                self.assertEqual(fn(*common,p(sample),len(sample),rows,dim,p(out)),0)
                np.testing.assert_allclose(out,answer,rtol=4e-5,atol=3e-4)
                emb=np.empty((len(ids),dim),dtype=np.float32)
                rc=getattr(lib,prefix+"_embedding_delta")(
                    *common,p(ids),len(ids),rows,dim,p(emb))
                self.assertEqual(rc,0)
                np.testing.assert_allclose(emb,recovered[ids],rtol=1e-6,atol=2e-6)

            scalar=np.empty_like(answer)
            lut=np.empty_like(answer)
            self.assertEqual(original_lib.calibri140_output_delta(
                *common,p(sample),len(sample),rows,dim,p(scalar)),0)
            self.assertEqual(improved_lib.calibri141_output_delta(
                *common,p(sample),len(sample),rows,dim,p(lut)),0)
            np.testing.assert_allclose(lut,scalar,rtol=4e-5,atol=3e-4)

    def test_wrong_physical_size_and_input_bounds_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            improved=ctypes.CDLL(str(compile_lut(Path(temp)/"lut.so")))
            ptr=ctypes.c_void_p
            fn=improved.calibri141_embedding_delta
            fn.argtypes=[ptr,ctypes.c_size_t,ptr,ctypes.c_size_t,ptr,
                         ptr,ptr,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ptr]
            fn.restype=ctypes.c_int
            rows,dim=4,16
            codes=np.frombuffer(pack5(np.zeros(rows*dim//8,dtype=np.uint8)),
                                dtype=np.uint8).copy()
            bits=np.frombuffer(pack_bits(np.zeros(rows*dim//8,dtype=np.uint8)),
                               dtype=np.uint8).copy()
            centers=np.zeros((32,2,8),dtype=np.float32)
            scales=np.ones(rows,dtype=np.float32)
            ids=np.array([rows+2],dtype=np.int64)
            output=np.empty((1,dim),dtype=np.float32)
            p=lambda a:ptr(int(a.ctypes.data))
            self.assertEqual(fn(p(codes),len(codes),p(bits),len(bits),
                                p(centers),p(scales),p(ids),1,rows,dim,p(output)),3)
            self.assertEqual(fn(p(codes),len(codes)-1,p(bits),len(bits),
                                p(centers),p(scales),p(ids),1,rows,dim,p(output)),2)


if __name__=="__main__":
    unittest.main()
