"""Check source-quality recovery controls and exact native V1/V2 outputs."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
from direct_codebook_v59 import unpack5
from direct_runtime_v59 import compile_kernel
from v181.codec import quantize
from v181.native import Direct181
from v181.protocol import GROUPS,TOTAL_UNIQUE

class RecoveryOracle(unittest.TestCase):
    def test_high_rate_distinction(self):
        self.assertEqual(GROUPS,{"ORACLE_V2":2,"ORACLE_V1":1})
        self.assertEqual(TOTAL_UNIQUE,134515008)
    def test_finite_physical_matrix_and_group_cost(self):
        rng=np.random.default_rng(181)
        w=rng.normal(size=(48,32)).astype(np.float32)
        scale=np.sqrt(np.mean(w.astype(np.float64)**2,axis=1)).astype("<f2")
        for width in (1,2):
            parts,info=quantize(w,scale,width)
            groups=48*32//width
            self.assertEqual(len(parts["packed"]),(groups*5+7)//8)
            self.assertEqual(len(parts["centroids"]),32*width*2)
            self.assertEqual(len(parts["row_scales"]),48*2)
            ix=unpack5(parts["packed"],groups)
            self.assertTrue(np.all(ix<32))
            self.assertTrue(np.isfinite(info["relative_weight_RMSE"]))
    def test_actual_C_native_vs_exact_decoded_weight(self):
        rng=np.random.default_rng(182)
        with TemporaryDirectory() as tmp:
            lib=compile_kernel(Path(__file__).parent/"v181"/"kernel.c",
                               Path(tmp)/"v181_native.so")
            for width in (1,2):
                native=Direct181(lib,width)
                m,n=11,32
                w=rng.normal(size=(m,n)).astype(np.float32)
                scale=np.sqrt(np.mean(w.astype(np.float64)**2,axis=1)).astype("<f2")
                parts,_=quantize(w,scale,width)
                ids=unpack5(parts["packed"],m*n//width)
                centers=np.frombuffer(parts["centroids"],dtype="<f2").astype(np.float32).reshape(32,width)
                scales=np.frombuffer(parts["row_scales"],dtype="<f2").astype(np.float32)
                h=SimpleNamespace(shape=(m,n),
                    packed=np.frombuffer(parts["packed"],dtype=np.uint8),
                    row_scales=scales,centroids=centers,group=width)
                decoded=centers[ids].reshape(m,n)*scales[:,None]
                x=rng.normal(size=(3,n)).astype(np.float32)
                np.testing.assert_allclose(native.linear(h,x),x@decoded.T,
                                            rtol=3e-5,atol=3e-5)
                selected=np.array([0,2,10,0],dtype=np.int64)
                np.testing.assert_allclose(native.embedding(h,selected),
                                            decoded[selected],rtol=0,atol=3e-6)
                bad=SimpleNamespace(**vars(h));bad.packed=h.packed[:1]
                with self.assertRaises(ValueError):native.linear(bad,x)
if __name__=="__main__":
    unittest.main()
