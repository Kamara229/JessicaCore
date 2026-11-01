"""CLASS-RANK exactness, real mathematical count/rank and independent rows."""
import unittest
import numpy as np
from hybrid_v13 import encode as hybrid_encode, decode as hybrid_decode
from class_rank_v13 import (
    BLOCK_CHOICES,P_NONZERO_PRIORS,TABLE,_row_encode,_row_decode,
    encode,decode,read_row,direct_matvec)

class ClassRankTests(unittest.TestCase):
    def test_huffman_prefix_free(self):
        for (n,p),(forward,reverse,longest) in TABLE.items():
            self.assertEqual(len(forward),n+1)
            for k,(code,bits) in forward.items():
                self.assertEqual(reverse[(bits,code)],k)
            self.assertGreater(longest,0)
            # No canonical Huffman code may prefix another.
            for k,(code,bits) in forward.items():
                for l,(other,size) in forward.items():
                    if k==l:continue
                    if bits<=size:
                        self.assertNotEqual(code,other>>(size-bits))
    def test_roundtrip_all_zero_levels_and_block_shapes(self):
        rng=np.random.default_rng(313)
        for block in BLOCK_CHOICES:
            for rows in (1,3,12):
                for pz in (0.,.5,.75,.885,1.):
                    q=rng.choice([-1,0,1],size=(rows,block*4),
                        p=[(1-pz)/2,pz,(1-pz)/2]).astype(np.int8)
                    for prior in range(4):
                        p=_row_encode(q[0],block,prior)
                        self.assertTrue(np.array_equal(q[0],_row_decode(p,block*4,block,prior)))
                    for idxmode in ("auto","u16","u8_escape"):
                        for group in (1,4,64):
                            b=encode(q,block,group,index_mode=idxmode)
                            self.assertTrue(np.array_equal(q,decode(b)))
                            for r in (0,rows-1):
                                self.assertTrue(np.array_equal(read_row(b,r),q[r]))
    def test_operator_matches_dense_quantized(self):
        rng=np.random.default_rng(4113)
        for block in (8,16,32,64):
            q=rng.choice([-1,0,1],size=(5,block*4),p=[.06,.88,.06]).astype(np.int8)
            s=np.column_stack((rng.uniform(-1.5,-.1,5),rng.uniform(.1,1.5,5)))
            x=rng.normal(size=q.shape[1])
            b=encode(q,block=block)
            got=direct_matvec(b,x,s)
            exp=np.where(q<0,s[:,0,None],np.where(q>0,s[:,1,None],0))@x
            self.assertTrue(np.allclose(got,exp,rtol=1e-11,atol=1e-11))
    def test_hybrid_correctness_multiple_sparsity(self):
        rng=np.random.default_rng(13)
        for pzero in (0.0,.5,.75,.885,1.0):
            q=rng.choice([-1,0,1],size=(3,128),
                p=[(1-pzero)/2,pzero,(1-pzero)/2]).astype(np.int8)
            blob,choice=hybrid_encode(q,group=2)
            self.assertTrue(np.array_equal(q,hybrid_decode(blob)))
            self.assertIn(choice["selection"],(
                "CL8","MASK_HUFF","CLASS_RANK16","CLASS_RANK32","CLASS_RANK64"))
            self.assertEqual(
                choice["physical_bytes_with_wrapper"],
                5+min(choice["bytes_by_codec"].values()))
        with self.assertRaises(ValueError):hybrid_decode(b"bad!")

    def test_truncation_and_bad_row(self):
        q=np.zeros((4,128),dtype=np.int8)
        b=encode(q,block=16,group=2)
        with self.assertRaises(ValueError):decode(b[:-1])
        with self.assertRaises(ValueError):decode(b[:11])
        with self.assertRaises(ValueError):decode(b"bad!"+b[4:])
        with self.assertRaises(IndexError):read_row(b,7)
        with self.assertRaises(ValueError):encode(np.ones((3,29),dtype=np.int8),block=16)
if __name__=="__main__":unittest.main()
