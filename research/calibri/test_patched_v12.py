"""Deterministic CL12 exception-codec tests, including pathological cases."""
import unittest
import numpy as np
from patched_words_v12 import (
    _pack,_unpack,_block_encode,_block_decode,
    encode_patched,decode_patched,read_row,codeword_matvec)
from dual_codec import StaticPairDictionary

class CL12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.d=StaticPairDictionary()
    def test_packed_integer_primitive(self):
        rng=np.random.default_rng(1202)
        for bits in [0,1,2,4,8,11,15,16]:
            for n in [0,1,5,16,128,257]:
                limit=1<<bits
                v=rng.integers(0,limit,size=n,dtype=np.uint16)
                p=_pack(v,bits)
                self.assertTrue(np.array_equal(v,_unpack(p,n,bits)))
    def test_patched_block_all_widths(self):
        rng=np.random.default_rng(34)
        for n in (1,2,9,32,64,128,256):
            for k in (0,1,5,15,55,127,255,511,16383,65535):
                v=rng.integers(0,k+1,size=n,dtype=np.uint32).astype(np.uint16)
                z,b=_block_encode(v)
                got,used=_block_decode(z,n)
                self.assertEqual(used,len(z))
                self.assertTrue(np.array_equal(v,got))
    def test_rows_random_access_and_direct_fused_matvec(self):
        rng=np.random.default_rng(78)
        for rows,cols in [(1,32),(3,96),(9,128),(18,256)]:
            for pzero in [0.,.5,.885,1.]:
                q=rng.choice([-1,0,1],size=(rows,cols),
                    p=[(1-pzero)/2,pzero,(1-pzero)/2]).astype(np.int8)
                words,offs=self.d.encode(q)
                scales=np.column_stack((rng.uniform(-2,-.1,rows),
                                        rng.uniform(.1,2,rows)))
                x=rng.standard_normal(cols)
                dense=np.where(q<0,scales[:,0,None],
                         np.where(q>0,scales[:,1,None],0.))@x
                for grp,block in [(1,64),(4,128),(16,256)]:
                    b,_=encode_patched(words,offs,cols,grp,block)
                    w2,off2,width=decode_patched(b)
                    self.assertTrue(np.array_equal(w2,words))
                    self.assertTrue(np.array_equal(off2,offs))
                    self.assertEqual(width,cols)
                    self.assertTrue(np.array_equal(self.d.decode(w2,off2,width),q))
                    for i in [0,rows-1]:
                        self.assertTrue(np.array_equal(words[offs[i]:offs[i+1]],read_row(b,i)))
                    got=codeword_matvec(b,self.d,x,scales)
                    self.assertTrue(np.allclose(got,dense,rtol=1e-11,atol=1e-11))
    def test_invalid_stream_fail_closed(self):
        q=np.zeros((4,64),dtype=np.int8)
        w,off=self.d.encode(q)
        b,_=encode_patched(w,off,64,2,128)
        for trunc in [b[:6],b[:-1],b[:12]]:
            with self.assertRaises(ValueError):decode_patched(trunc)
        with self.assertRaises(IndexError):read_row(b,7)
        with self.assertRaises(ValueError):decode_patched(b'BAD!'+b[4:])
if __name__=='__main__':unittest.main()
