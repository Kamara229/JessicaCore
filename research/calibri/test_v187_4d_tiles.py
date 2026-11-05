"""Real in-process 4D geometry, complete physical bytes and C CPU correctness."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from direct_codebook_v59 import compress_matrix
from direct_runtime_v59 import NativeKernel,compile_kernel
from v187.codec import quantize,reconstruct
from v187.native import Kernel187
from v187 import protocol as p

class CrossRowMultidimensional(unittest.TestCase):
    def test_same_group_bits_and_exact_original_source_row_scales(self):
        rng=np.random.default_rng(187)
        original=rng.standard_normal(size=(16,32)).astype(np.float32)
        packed,manifest=quantize("toy.2x4",original)
        control=compress_matrix(original)
        self.assertEqual(manifest["physical_5bit_indices_bytes"],len(control["packed"]))
        self.assertEqual(manifest["centroids_fp16_bytes"],len(control["centroids"]))
        self.assertEqual(manifest["scales_fp16_bytes"],len(control["row_scales"]))
        self.assertEqual(manifest["complete_matrix_payload_bytes"],
                         sum(len(control[k]) for k in ("packed","centroids","row_scales")))
        self.assertEqual(packed.shape,original.shape)
        self.assertEqual(len(packed.packed),40)
        self.assertEqual(packed.payload_bytes(),584)
        np.testing.assert_array_equal(packed.row_scales,
            np.frombuffer(control["row_scales"],dtype="<f2").astype(np.float32))
        self.assertEqual(reconstruct(packed).shape,original.shape)

    def test_native_packed_C_4D_rowpair_GEMV_and_odd_even_embedding(self):
        rng=np.random.default_rng(191)
        original=rng.normal(size=(32,64)).astype(np.float32)
        source,details=quantize("source.4D",original)
        reference=reconstruct(source)
        with TemporaryDirectory() as temp:
            so=compile_kernel(Path(__file__).parent/"v187"/"kernel.c",
                              Path(temp)/"calibri_4d.so")
            kernel=Kernel187(so)
            x=rng.normal(size=(4,64)).astype(np.float32)
            expected=x@reference.T
            actual=kernel.linear(source,x)
            np.testing.assert_allclose(actual,expected,atol=7.e-5,rtol=7.e-5)
            ids=np.array([0,1,2,3,17,30,31],dtype=np.int64)
            embs=kernel.embedding(source,ids)
            np.testing.assert_allclose(embs,reference[ids],atol=3.e-6,rtol=3.e-6)
            with self.assertRaises(ValueError):
                kernel.linear(source,x[:,:5])
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue(np.isfinite(embs).all())

    def test_frozen_geometry_and_invalid_source_rejection(self):
        self.assertEqual(p.ROW_AXIS*p.COL_AXIS,8)
        self.assertEqual(p.K,32)
        self.assertEqual(len(p.MODEL_PILOT),8)
        self.assertEqual(p.BITS_PER_TILE,5)
        for shape in ((15,64),(16,62),(0,64)):
            with self.assertRaises(ValueError):
                quantize("wrong",np.zeros(shape,dtype=np.float32))
        with self.assertRaises(ValueError):
            quantize("nonfinite",np.full((16,64),float("nan"),dtype=np.float32))

if __name__=="__main__":
    unittest.main()
