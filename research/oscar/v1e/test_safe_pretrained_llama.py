"""Oscar independent ~42M original pretrained LLaMA reference ABI smoke."""
from __future__ import annotations

import unittest
import torch

from oscar.v1e.safe_pretrained_llama import (
    Shape, LocalOriginalPretrained42M, RMS,
)


class NoRemotePythonModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.m=LocalOriginalPretrained42M(
            Shape(vocab=96,dim=64,layers=2,heads=4,hidden=128,max_ctx=32)
        ).eval()

    def test_tied_unique_input_output_source(self):
        self.assertIs(self.m.tok_embeddings.weight,self.m.output.weight)
        sd=self.m.state_dict()
        self.assertEqual(sd["tok_embeddings.weight"].data_ptr(),
                         sd["output.weight"].data_ptr())
        out,states=self.m(torch.tensor([[1,7,32,9]]),hidden=True)
        self.assertEqual(out.shape,(1,4,96))
        self.assertEqual(len(states),4)
        self.assertTrue(torch.isfinite(out).all())

    def test_original_pretrained_transformer_is_causal(self):
        a=torch.tensor([[1,7,32,9,12]])
        b=torch.tensor([[1,7,32,93,87]])
        x,_=self.m(a)
        y,_=self.m(b)
        self.assertTrue(torch.allclose(x[:,:3],y[:,:3],
                                       rtol=2e-5,atol=2e-5))

    def test_rope_pair_interleaving_full_shape(self):
        x=torch.tensor([[1,7,2,15]])
        out,_=self.m(x)
        self.assertEqual(self.m.cos.shape,(32,8))
        self.assertEqual(self.m.sin.shape,(32,8))
        self.assertTrue(torch.isfinite(out).all())

    def test_wrong_large_position_rejected(self):
        with self.assertRaises(ValueError):
            self.m(torch.ones((1,33),dtype=torch.long))


if __name__=="__main__":
    unittest.main()
