"""Focused v180 orthogonal-source source->native regression tests."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from direct_codebook_v59 import QuantHandle, compress_matrix, unpack5
from direct_runtime_v59 import compile_kernel
from v180.rotation import apply, signs
from v180.native import RotatedNative
from v180.protocol import BLOCKS, ARMS, MAX_BYTES, TOTAL_UNIQUE

class Rotations(unittest.TestCase):
    def test_involutive_full_precision_and_energy(self):
        rng=np.random.default_rng(180)
        for block in (32,64):
            a=rng.normal(size=(17,192)).astype(np.float32)
            transformed=apply(a,block)
            decoded=apply(transformed,block,inverse=True)
            np.testing.assert_allclose(decoded,a,rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose((transformed*transformed).sum(axis=1),
                (a*a).sum(axis=1),rtol=3e-6,atol=3e-5)
            self.assertTrue(set(np.unique(signs(192)))=={-1.,1.})
            with self.assertRaises(ValueError): apply(a[:,:20],block)
        self.assertLess(MAX_BYTES,12000000)
        self.assertEqual(TOTAL_UNIQUE,134515008)

    def test_transformed_exact_linear_identity(self):
        r=np.random.default_rng(180)
        for block in (32,64):
            weight=r.normal(size=(13,128)).astype(np.float32)
            x=r.normal(size=(4,128)).astype(np.float32)
            y_original=x@weight.T
            y_transformed=apply(x,block)@apply(weight,block).T
            np.testing.assert_allclose(y_original,y_transformed,rtol=1e-5,atol=1e-5)

    def test_compiled_native_correct_basis_and_embedding(self):
        rng=np.random.default_rng(181)
        with TemporaryDirectory() as tmp:
            library=compile_kernel(Path(__file__).parent/"v180"/"kernel.c",
                                   Path(tmp)/"cb180.so")
            for block in (32,64):
                native=RotatedNative(library,block)
                # Matrix quantities deliberately not multiples of 8 in rows;
                # groups across rows DO align with pack5 if ncols=128.
                matrix=rng.normal(size=(13,128)).astype(np.float32)
                packed=compress_matrix(apply(matrix,block))
                h=QuantHandle("toy",(13,128),
                    np.frombuffer(packed["packed"],dtype=np.uint8),
                    np.frombuffer(packed["row_scales"],dtype="<f2").astype(np.float32),
                    np.frombuffer(packed["centroids"],dtype="<f2").astype(np.float32).reshape(32,8))
                ids=unpack5(h.packed.tobytes(),13*128//8)
                qrot=h.centroids[ids].reshape(13,128)*h.row_scales[:,None]
                reference=apply(qrot,block,inverse=True)
                x=rng.normal(size=(5,128)).astype(np.float32)
                np.testing.assert_allclose(native.linear(h,x),
                    x@reference.T,atol=4e-5,rtol=4e-5)
                selected=np.array([0,3,12,7,3],dtype=np.int64)
                np.testing.assert_allclose(native.embedding(h,selected),
                    reference[selected],atol=4e-6,rtol=4e-6)

if __name__=="__main__":
    unittest.main()
