"""Fast v184 correctness checks: source byte budget, gradients, native C.

Do not invoke Jessica or use unseen private TEST. Full-source SHA/QA
integrity is tested in the separate real pretrained GitHub Actions job.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from direct_codebook_v59 import QuantHandle,compress_matrix,unpack5
from direct_runtime_v59 import compile_kernel
from v184 import protocol as p
from v184.codec import _matrix_alpha
from v184.native import Joint184Kernel
from v184.trainable import ste_half,ste_int8

class SharedRank2(unittest.TestCase):
    def test_all_joint_source_bytes_are_physically_payable(self):
        self.assertEqual(p.RANK,2)
        self.assertEqual(p.SOURCE_BYTES,11229303)
        self.assertEqual(p.MAX_BYTES,11770063)
        self.assertEqual(p.HEADROOM,540760)
        # 211 paid per-matrix scalar gains + 409,344 A codes,
        # two shared rank2 widths and bounded JSON/CRC/header overhead.
        self.assertEqual(409344 + 211*4 + (576+1536)*p.RANK*2,418636)
        self.assertLess(418636+50000,p.HEADROOM)

    def test_quantization_aware_compact_gradient_is_finite(self):
        torch.manual_seed(p.SEED)
        x=torch.randn(4,32)
        alpha=torch.nn.Parameter(torch.zeros((11,2)))
        beta=torch.nn.Parameter(torch.randn(2,32)/32**.5)
        y=(x@ste_half(beta).T)@ste_int8(alpha).T
        target=torch.randn_like(y)
        loss=(y-target).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertGreater(float(alpha.grad.abs().sum()),0)
        with torch.no_grad():alpha-=0.01*alpha.grad
        self.assertGreater(float(alpha.abs().sum()),0)
        raw,gain=_matrix_alpha(alpha.detach().numpy())
        self.assertEqual(len(raw),22)
        self.assertEqual(len(gain),4)
        self.assertTrue(np.frombuffer(raw,dtype=np.int8).any())

    def test_true_packed_source_plus_native_rank2_matches_decoded(self):
        rng=np.random.default_rng(184)
        w=rng.normal(size=(13,32)).astype(np.float32)
        encoded=compress_matrix(w)
        handle=QuantHandle("toy.weight",(13,32),
            np.frombuffer(encoded["packed"],dtype=np.uint8),
            np.frombuffer(encoded["row_scales"],dtype="<f2").astype(np.float32),
            np.frombuffer(encoded["centroids"],dtype="<f2").astype(np.float32).reshape(32,8))
        ids=unpack5(handle.packed.tobytes(),13*32//8)
        base=handle.centroids[ids].reshape(handle.shape)*handle.row_scales[:,None]
        a=rng.integers(-40,41,size=(13,2),dtype=np.int8)
        gain=.0014
        beta=rng.normal(size=(2,32)).astype(np.float32)
        with TemporaryDirectory() as folder:
            path=compile_kernel(Path(__file__).parent/"v184"/"kernel.c",
                                Path(folder)/"native184.so")
            fake=SimpleNamespace(adapter_for=lambda h:(a,gain,beta))
            kernel=Joint184Kernel(path,fake)
            x=rng.normal(size=(5,32)).astype(np.float32)
            expected=x@base.T+(x@beta.T)@(a.astype(np.float32)*gain).T
            actual=kernel.linear(handle,x)
            np.testing.assert_allclose(actual,expected,atol=4e-5,rtol=4e-5)
            rowids=np.array([0,12,1,5,12],dtype=np.int64)
            emb=kernel.embedding(handle,rowids)
            reference=base[rowids]+(a[rowids].astype(np.float32)*gain)@beta
            np.testing.assert_allclose(emb,reference,atol=4e-6,rtol=4e-6)
            with self.assertRaises(ValueError):
                kernel.linear(handle,x[:,:16])

if __name__=="__main__":
    unittest.main()
