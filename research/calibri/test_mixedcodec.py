import unittest
import numpy as np
from mixedcodec import pack_symmetric,unpack_symmetric

class MixedTest(unittest.TestCase):
    def test_roundtrip(self):
        rng=np.random.default_rng(20261101)
        for bits,limit in ((4,7),(8,127)):
            for shape in ((1,1),(2,3),(5,17),(128,256)):
                q=rng.integers(-limit,limit+1,size=shape,dtype=np.int16).astype(np.int8)
                s=rng.uniform(1e-3,1.,size=shape[0]).astype('<f2')
                b=pack_symmetric(q,s,bits)
                back,s2=unpack_symmetric(b)
                self.assertTrue(np.array_equal(q,back))
                self.assertTrue(np.array_equal(s.astype(np.float32),s2))
                self.assertEqual(len(b),12+shape[0]*2+
                    ((q.size+1)//2 if bits==4 else q.size))
                with self.assertRaises(ValueError):unpack_symmetric(b[:-1])
if __name__=='__main__':unittest.main()
