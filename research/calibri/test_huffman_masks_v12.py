"""Huffman 8-mask decoder tests independent of PFOR and QMoE dictionary."""
import unittest
import numpy as np
from huffman_masks_v12 import CODE,REVERSE,MAX_BITS,row_encode,row_decode,encode,decode,read_row,direct_matvec

class HuffmanMasksTests(unittest.TestCase):
    def test_prefix_properties(self):
        self.assertEqual(len(CODE),94)
        self.assertLessEqual(MAX_BITS,10)
        seen=set()
        for symbol,(value,bits) in CODE.items():
            self.assertNotIn((value,bits),seen)
            seen.add((value,bits))
            self.assertEqual(REVERSE[(bits,value)],symbol)
    def test_roundtrip_and_direct_matvec(self):
        rng=np.random.default_rng(32)
        for shape in [(1,8),(3,32),(9,96),(18,256)]:
            for pzero in (0.,.5,.885,1.):
                q=rng.choice([-1,0,1],size=shape,
                   p=[(1-pzero)/2,pzero,(1-pzero)/2]).astype(np.int8)
                for r in q:
                    self.assertTrue(np.array_equal(r,row_decode(row_encode(r),len(r))))
                for group in (1,4,16):
                    for mode in ('auto','u16','u8_escape'):
                        self._roundtrip_group(q,group,mode,rng)

    def _roundtrip_group(self,q,group,mode,rng):
        shape=q.shape
        blob=encode(q,group,index_mode=mode)
        self.assertTrue(np.array_equal(q,decode(blob)))
        self.assertTrue(np.array_equal(q[0],read_row(blob,0)))
        self.assertTrue(np.array_equal(q[-1],read_row(blob,len(q)-1)))
        scales=np.column_stack((rng.uniform(-2,-.1,shape[0]),rng.uniform(.1,2,shape[0])))
        x=rng.normal(size=shape[1])
        gold=np.where(q<0,scales[:,0,None],np.where(q>0,scales[:,1,None],0.))@x
        got=direct_matvec(blob,x,scales)
        self.assertTrue(np.allclose(gold,got,rtol=1e-12,atol=1e-12))
    def test_corruption(self):
        q=np.zeros((3,32),np.int8)
        blob=encode(q)
        for damaged in [blob[:8],blob[:-1],b'BAD!'+blob[4:]]:
            with self.assertRaises(ValueError):decode(damaged)
        with self.assertRaises(IndexError):read_row(blob,3)
        with self.assertRaises(ValueError):row_decode(b'',32)

if __name__=='__main__':unittest.main()
