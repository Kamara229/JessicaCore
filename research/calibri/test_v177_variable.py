"""Focused v177 index, source and compiled-native executor tests."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from direct_runtime_v59 import compile_kernel
from v177.codec import pack_indices, unpack_indices, quantize_matrix
from v177.native import MixedNativeKernel
from v177.protocol import MAX_BYTES, TOTAL_UNIQUE, allocation


class VariableBits(unittest.TestCase):
    def test_all_frozen_classification(self):
        names = (
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight")
        self.assertEqual([allocation("ADAPT_A",n) for n in names],[6,6,4,4,5])
        self.assertEqual([allocation("ADAPT_B",n) for n in names],[6,5,4,4,6])
        self.assertLess(MAX_BYTES,12_000_000)
        self.assertEqual(TOTAL_UNIQUE,134515008)
        with self.assertRaises(ValueError):
            allocation("ADAPT_C",names[0])

    def test_bitpacked_roundtrip_all_crossbyte_offsets(self):
        for bits in (4,5,6):
            for count in (1,2,3,7,8,9,13,16,63,71,513):
                values=np.random.default_rng(bits*10000+count).integers(
                    0,1<<bits,size=count,dtype=np.uint8)
                blob=pack_indices(values,bits)
                self.assertEqual(len(blob),(count*bits+7)//8)
                np.testing.assert_array_equal(unpack_indices(blob,count,bits),values)
            with self.assertRaises(ValueError):
                pack_indices(np.array([1<<bits],dtype=np.uint8),bits)
            with self.assertRaises(ValueError):
                unpack_indices(b"\x00",100,bits)

    def test_quantize_finite_and_paid_codebooks(self):
        w=np.random.default_rng(177).normal(size=(64,16)).astype(np.float32)
        for bits in (4,6):
            fields,diagnostics=quantize_matrix(w,bits)
            self.assertEqual(len(fields["centroids"]),(1<<bits)*8*2)
            self.assertEqual(len(fields["row_scales"]),64*2)
            self.assertEqual(len(fields["packed"]),(64*2*bits+7)//8)
            self.assertTrue(np.isfinite(diagnostics["relative_weight_RMSE"]))
            self.assertTrue(0<diagnostics["relative_weight_RMSE"]<2.0)

    def test_gcc_native_all_bits_against_exact_physical_decoded(self):
        with TemporaryDirectory() as tmp:
            lib=compile_kernel(Path(__file__).parent/"v177"/"kernel.c",
                               Path(tmp)/"mixed.so")
            native=MixedNativeKernel(lib)
            rng=np.random.default_rng(177)
            for bits in (4,5,6):
                m,n=11,24
                groups=m*n//8
                ids=rng.integers(0,1<<bits,size=groups,dtype=np.uint8)
                pack=np.frombuffer(pack_indices(ids,bits),dtype=np.uint8)
                scales=rng.uniform(0.1,2,size=m).astype(np.float32)
                centers=rng.normal(size=(1<<bits,8)).astype(np.float32)
                handle=SimpleNamespace(shape=(m,n),packed=pack,
                                       row_scales=scales,centroids=centers,bits=bits)
                dense=centers[ids].reshape(m,n)*scales[:,None]
                x=rng.normal(size=(3,n)).astype(np.float32)
                actual=native.linear(handle,x)
                oracle=x@dense.T
                np.testing.assert_allclose(actual,oracle,rtol=2e-5,atol=2e-5)
                requested=np.array([0,1,10,9,0],dtype=np.int64)
                np.testing.assert_allclose(native.embedding(handle,requested),
                                           dense[requested],rtol=2e-6,atol=2e-6)
                bad=SimpleNamespace(**vars(handle))
                bad.packed=pack[:1]
                with self.assertRaises(ValueError):
                    native.linear(bad,x)


if __name__=="__main__":
    unittest.main()
