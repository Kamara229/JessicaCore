import ctypes
import tempfile
import unittest
from pathlib import Path

import numpy as np

from direct_codebook_v59 import pack5
from v140.codec import pack_bits,unpack_bits,centers_as_physical
from v140.runtime import compile_onebit
from v140.train import assign_full_extra_bits
from v140.scoring import summarize_native


class PaidNativeOneBit(unittest.TestCase):
    def test_onebit_roundtrip_and_strict_center_geometry(self):
        b=np.array([1,0,0,1,1,0,1,0,1,0,1],dtype=np.uint8)
        self.assertTrue(np.array_equal(unpack_bits(pack_bits(b),len(b)),b))
        with self.assertRaises(ValueError):
            pack_bits([0,2])
        with self.assertRaises(ValueError):
            centers_as_physical(np.zeros((64,8),dtype=np.float32))
        with self.assertRaises(ValueError):
            unpack_bits(b"abc",12)

    def test_original_weight_residual_is_paid_for_each_8d_group(self):
        weight=np.ones((4,16),dtype=np.float32)
        scales=np.ones(4,dtype=np.float32)
        table=np.zeros((32,8),dtype=np.float32)
        pair=np.zeros((32,2,8),dtype=np.float32)
        pair[0,1,:]=1.0
        handle=type("Handle",(),{"row_scales":scales,"centroids":table})()
        plane,stats=assign_full_extra_bits(weight,handle,pair,
                                           np.zeros(8,dtype=np.uint8),
                                           batch_groups=8)
        self.assertEqual(plane,b"\xff")
        self.assertEqual(stats["independent_full_source_groups"],8)
        self.assertEqual(stats["full_source_embedding_RMSE_6bit"],0.0)

    def test_natively_compiled_onebit_C_projection_equals_dense_oracle(self):
        c_ptr=ctypes.c_void_p;size=ctypes.c_size_t;i64=ctypes.c_int64
        expected=[c_ptr,size,c_ptr,size,c_ptr,c_ptr,c_ptr,i64,i64,i64,c_ptr]
        with tempfile.TemporaryDirectory() as temp:
            binary=compile_onebit(Path(temp)/"onebit.so")
            lib=ctypes.CDLL(str(binary))
            emb=lib.calibri140_embedding_delta
            outfn=lib.calibri140_output_delta
            emb.argtypes=expected;outfn.argtypes=expected
            emb.restype=ctypes.c_int;outfn.restype=ctypes.c_int

            rows,dim=4,16
            codes=np.arange(8,dtype=np.uint8)
            original=np.frombuffer(pack5(codes),dtype=np.uint8).copy()
            flags=np.array([0,1,1,0,1,0,0,1],dtype=np.uint8)
            extra=np.frombuffer(pack_bits(flags),dtype=np.uint8).copy()
            table=np.zeros((32,2,8),dtype=np.float32)
            for k in range(8):
                for bit in range(2):
                    table[k,bit,:]=np.arange(8,dtype=np.float32)*.01+k*.1+bit
            scales=np.array([1.0,.5,2.0,1.5],dtype=np.float32)
            rows_selected=np.array([0,3,1],dtype=np.int64)
            output=np.zeros((len(rows_selected),dim),dtype=np.float32)
            ptr=lambda arr:c_ptr(int(arr.ctypes.data))
            common=[ptr(original),len(original),ptr(extra),len(extra),
                    ptr(table),ptr(scales)]
            self.assertEqual(emb(*common,ptr(rows_selected),len(rows_selected),
                                 rows,dim,ptr(output)),0)

            expected_rows=np.zeros((rows,dim),dtype=np.float32)
            for row in range(rows):
                for g in range(2):
                    k=row*2+g
                    expected_rows[row,g*8:(g+1)*8]=scales[row]*table[codes[k],flags[k]]
            np.testing.assert_allclose(output,expected_rows[rows_selected],atol=1e-6)

            x=np.arange(32,dtype=np.float32).reshape(2,dim)*.01
            mat=np.zeros((2,rows),dtype=np.float32)
            self.assertEqual(outfn(*common,ptr(x),len(x),rows,dim,ptr(mat)),0)
            # Numerically allowed different summation order in compiled C.
            np.testing.assert_allclose(mat,x@expected_rows.T,rtol=1e-5,atol=1e-5)

    def test_reject_incomplete_native_quality_score(self):
        with self.assertRaises(ValueError):
            summarize_native({"original_CB59":[]})


if __name__=="__main__":
    unittest.main()
