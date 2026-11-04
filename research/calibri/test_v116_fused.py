"""Fast parity and cost accounting for temporary generated GEMV blocks."""
import unittest
import torch
from v114.operators import TinyLanguageModel
from v114.protocol import TRAIN,example
from v116.runtime import (
    fuse_compact,hot_generated_weight_bytes,
    hot_generated_all_buffers_bytes)

class Tests(unittest.TestCase):
    def test_generated_operators_are_original_function(self):
        torch.manual_seed(116)
        for mode in ("shared","shared_adapter"):
            compact=TinyLanguageModel(mode).eval()
            fused=fuse_compact(compact)
            self.assertGreater(hot_generated_weight_bytes(fused),0)
            self.assertGreaterEqual(
                hot_generated_all_buffers_bytes(fused),
                hot_generated_weight_bytes(fused))
            self.assertLess(
                hot_generated_weight_bytes(fused),
                3*2*32*32*4+1)
            case=example(*TRAIN[0])
            x=torch.tensor([case["context"]],dtype=torch.long)
            previous=torch.tensor([[2,4,18,5,13,6]],dtype=torch.long)
            original=compact(x,previous)
            joined=fused(x,previous)
            self.assertLess(
                float(torch.max(torch.abs(original-joined))),5e-5)
            before=compact.generate(x[0])
            after=fused.generate(x[0])
            self.assertEqual(before,after)

    def test_original_is_not_mutated_and_wrong_source_rejected(self):
        model=TinyLanguageModel("shared_adapter")
        original={k:v.detach().clone() for k,v in model.state_dict().items()}
        transformed=fuse_compact(model)
        self.assertTrue(all(
            torch.equal(v,model.state_dict()[k])
            for k,v in original.items()))
        self.assertFalse(any(k.startswith("book.") for k in transformed.state_dict()))
        with self.assertRaises(ValueError):
            fuse_compact(TinyLanguageModel("dense"))

if __name__=="__main__":
    unittest.main()
