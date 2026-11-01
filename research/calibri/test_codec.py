"""Deterministic local checks for codec and exact operator execution."""
import unittest
import numpy as np
from codec import encode,decode
from index import encode as compact_encode,decode as compact_decode
from lift import measure,fused_matvec

class TestCodec(unittest.TestCase):
    def test_roundtrip_and_matvec(self):
        rng=np.random.default_rng(7)
        for r,c in ((2,32),(6,128),(16,256)):
            for zero in (0.,.5,.885,1.):
                q=rng.choice([-1,0,1],size=(r,c),
                             p=[(1-zero)/2,zero,(1-zero)/2]).astype(np.int8)
                self.assertTrue(np.array_equal(q,decode(encode(q))))
                self.assertTrue(np.array_equal(q,compact_decode(compact_encode(q))))
                n,_,(a,s,d,ps)=measure(q)
                self.assertGreater(n,0)
                lo=rng.uniform(-2,-.1,r);hi=rng.uniform(.1,2,r);x=rng.normal(size=c)
                got=fused_matvec(a,s,d,ps,x,lo,hi)
                expected=np.where(q<0,lo[:,None],np.where(q>0,hi[:,None],0.))@x
                self.assertTrue(np.allclose(got,expected,atol=1e-11,rtol=1e-11))

if __name__=='__main__':unittest.main()
