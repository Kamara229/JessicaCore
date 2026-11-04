"""Fast smoke checks, including real autoregressive output and compact parameter format."""
import unittest
import torch
from v114.protocol import TRAIN,VALID,example,TOKENS,MODES,STEPS
from v114.operators import TinyLanguageModel

class BehavioralGuards(unittest.TestCase):
    def test_no_train_valid_case_overlap(self):
        self.assertEqual(len(TRAIN),28)
        self.assertEqual(len(VALID),8)
        self.assertFalse(set(TRAIN).intersection(VALID))
        self.assertEqual(len(TOKENS),23)

    def test_all_three_direct_models(self):
        for mode in MODES:
            model=TinyLanguageModel(mode)
            case=example(*TRAIN[0])
            context=torch.tensor([case["context"]],dtype=torch.long)
            prev=torch.tensor([[2,4,17,5,11,6]],dtype=torch.long)
            scores=model(context,prev)
            self.assertEqual(tuple(scores.shape),(1,6,len(TOKENS)))
            self.assertTrue(torch.isfinite(scores).all())
            generated=model.generate(context[0])
            self.assertTrue(0<len(generated)<=6)
            self.assertTrue(all(0<=idx<len(TOKENS) for idx in generated))

    def test_compact_core_uses_only_factors_not_full_dense_matrices(self):
        dense=TinyLanguageModel("dense")
        shared=TinyLanguageModel("shared")
        adapted=TinyLanguageModel("shared_adapter")
        n=lambda model:sum(p.numel() for p in model.parameters())
        self.assertLess(n(shared),n(adapted))
        self.assertLess(n(adapted),n(dense))
        # No trainable D x D role weight stored in the compact core.
        self.assertFalse(any(
            tuple(p.shape)==(32,32) for name,p in shared.named_parameters()))

if __name__=="__main__":
    unittest.main()
