"""Verify actual 8-bank native C kernel head/embedding equality and source budget."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from direct_codebook_v59 import QuantHandle
from v104.runtime import BankedKernel,BankedPackedLinear,BankedPackedEmbedding
from v104.spec import BANKS,K,V,VOCAB,DIM,ROW_PART
from v104.physical import BPW,TOTAL_BYTES,EXTRA_BYTES

def fixture():
    all_zero_indices=np.zeros(VOCAB*(DIM//V)*5//8,dtype=np.uint8)
    scales=np.ones(VOCAB,dtype=np.float32)
    unused=np.zeros((K,V),dtype=np.float32)
    h=QuantHandle("test.output_and_input_tied",(VOCAB,DIM),
                  all_zero_indices,scales,unused)
    banks=np.zeros((BANKS,K,V),dtype=np.float32)
    for i in range(BANKS):
        banks[i,0,:]=i+1
    return h,banks

class Tests(unittest.TestCase):
    def test_source_paid_complete_archive_budget(self):
        self.assertLessEqual(BPW,.700)
        self.assertEqual(EXTRA_BYTES,4096)
        self.assertEqual(TOTAL_BYTES,11229303+24+4096+36)
    def test_direct_native_output_and_embedding_choose_same_bank(self):
        h,banks=fixture()
        with TemporaryDirectory() as work:
            source=Path(__file__).parent/"v104"/"native104.c"
            kernel=BankedKernel(source,Path(work)/"libv104_test.so")
            head=BankedPackedLinear(h,banks,kernel)
            emb=BankedPackedEmbedding(h,banks,kernel)
            ones=torch.ones((1,1,DIM),dtype=torch.float32)
            score=head(ones).detach().numpy()[0,0]
            for bank in range(BANKS):
                a=bank*ROW_PART
                b=(bank+1)*ROW_PART
                np.testing.assert_allclose(score[a:b],DIM*(bank+1),rtol=0,atol=.1)
            ids=torch.tensor([0,ROW_PART-1,ROW_PART,2*ROW_PART-1,2*ROW_PART,
                              VOCAB-1],dtype=torch.long)
            actual=emb(ids).detach().numpy()
            expected=np.array([1,1,2,2,3,8],dtype=np.float32)
            np.testing.assert_allclose(actual,expected[:,None],rtol=0,atol=1e-5)
            for i,token in enumerate(ids.tolist()):
                self.assertAlmostEqual(float(score[token]),
                    float(actual[i].sum()),places=3)
    def test_native_embedding_rejects_bad_token(self):
        h,banks=fixture()
        with TemporaryDirectory() as work:
            kernel=BankedKernel(Path(__file__).parent/"v104"/"native104.c",
                                Path(work)/"libv104_test.so")
            embedding=BankedPackedEmbedding(h,banks,kernel)
            with self.assertRaises(RuntimeError):
                embedding(torch.tensor([-1]))

if __name__=="__main__":unittest.main()
