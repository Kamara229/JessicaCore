"""Column-normalized physical source and native CPU numerical regression."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from direct_runtime_v59 import compile_kernel
from v177.codec import unpack_indices
from v178 import protocol as p
from v178.codec import calculate_columns, quantize
from v178.native import ColumnNative


class PaidColumns(unittest.TestCase):
    def test_frozen_byte_budget(self):
        self.assertEqual(p.GROUP,8)
        self.assertEqual(p.BITS,5)
        self.assertEqual(p.ENTRIES,32)
        self.assertEqual(p.TOTAL_UNIQUE,134515008)
        self.assertLessEqual(p.SOURCE_BYTES,p.MAX_BYTES)
        self.assertTrue(p.MAX_BYTES<12000000)

    def test_source_row_and_column_factorization(self):
        r=np.random.default_rng(178)
        w=r.normal(size=(96,32)).astype(np.float32)
        w[:,0]*=4.0
        w[:,1]*=0.2
        rows=np.sqrt(np.mean(w.astype(np.float64)**2,axis=1)).astype("<f2")
        c=calculate_columns(w,rows)
        self.assertEqual(c.shape,(32,))
        self.assertTrue(np.all((c>=0.25)&(c<=4.0)))
        self.assertGreater(c[0],c[1])
        fields,quality=quantize(w,rows)
        self.assertEqual(len(fields["packed"]),(96*32//8*5+7)//8)
        self.assertEqual(len(fields["row_scales"]),96*2)
        self.assertEqual(len(fields["col_scales"]),32*2)
        self.assertEqual(len(fields["centroids"]),32*8*2)
        self.assertTrue(np.isfinite(quality["source_weight_RMSE"]))
        ids=unpack_indices(fields["packed"],96*32//8,5)
        self.assertEqual(ids.shape,(384,))
        self.assertTrue(np.all(ids<32))

    def test_real_compiled_packed_linear_embedding(self):
        r=np.random.default_rng(179)
        with TemporaryDirectory() as tmp:
            native=ColumnNative(compile_kernel(
                Path(__file__).parent/"v178"/"kernel.c", Path(tmp)/"test.so"))
            for rows,cols in ((11,24),(32,16),(7,40)):
                codes=r.integers(0,32,size=rows*cols//8,dtype=np.uint8)
                from v177.codec import pack_indices
                h=SimpleNamespace(shape=(rows,cols),
                    packed=np.frombuffer(pack_indices(codes,5),dtype=np.uint8),
                    row_scales=r.uniform(.1,2,size=rows).astype(np.float32),
                    col_scales=r.uniform(.25,4,size=cols).astype(np.float32),
                    centroids=r.normal(size=(32,8)).astype(np.float32))
                dense=(h.centroids[codes].reshape(rows,cols)*h.row_scales[:,None]*
                       h.col_scales[None,:])
                inputs=r.normal(size=(3,cols)).astype(np.float32)
                np.testing.assert_allclose(native.linear(h,inputs),
                                           inputs@dense.T,atol=3e-5,rtol=3e-5)
                ix=np.array([0,rows-1,0],dtype=np.int64)
                np.testing.assert_allclose(native.embedding(h,ix),
                                           dense[ix],atol=3e-6,rtol=3e-6)
                bad=SimpleNamespace(**vars(h));bad.packed=h.packed[:1]
                with self.assertRaises(ValueError):
                    native.linear(bad,inputs)

    def test_invalid_source_columns_rejected(self):
        with self.assertRaises(ValueError):
            calculate_columns(np.ones((10,16),np.float32), np.zeros(10,dtype=np.float32))


if __name__=="__main__":
    unittest.main()
