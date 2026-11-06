"""Synthetic no-network v220 invariants: original TRAIN split and feature gradients."""
import unittest
from types import SimpleNamespace

import torch

from v196 import protocol as p
from v220.run import (TRAIN_INDICES, HELDOUT_TRAIN_INDICES, check_protocol,
                       feature_objective)


class FakeStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features=torch.nn.Parameter(torch.tensor(0.5))
        self.base=torch.nn.Parameter(torch.randn(1,3,p.VOCAB_SIZE)*0.0001)

    def forward(self,input_ids,use_cache=False,output_hidden_states=False):
        h=torch.ones((1,input_ids.shape[1],4))*self.features
        logits=self.base[:,:input_ids.shape[1],:] + self.features
        return SimpleNamespace(hidden_states=(h,h*0.9,h*1.1),logits=logits)


class V220SyntheticTest(unittest.TestCase):
    def test_precommitted_split(self):
        check_protocol()
        self.assertEqual(len(TRAIN_INDICES),16)
        self.assertEqual(HELDOUT_TRAIN_INDICES,(4,9,14,19))

    def test_feature_alignment_backpropagates(self):
        torch.manual_seed(1)
        model=FakeStudent()
        probs=torch.ones((2,p.VOCAB_SIZE))/p.VOCAB_SIZE
        reference={"split":"TRAIN","ids":torch.tensor([[3,4,5]]),
                   "positions":torch.tensor([0,2]),
                   "layer_states":tuple(torch.ones(2,4) for _ in range(3)),
                   "vocab_probs":probs.half()}
        a,b=feature_objective(model,reference)
        (a+0.15*b).backward()
        self.assertIsNotNone(model.features.grad)
        self.assertTrue(torch.isfinite(model.features.grad))
        self.assertNotEqual(float(model.features.grad),0.)

    def test_no_heldout_or_dev_as_optimizer_reference(self):
        with self.assertRaises(AssertionError):
            feature_objective(FakeStudent(),{"split":"DEV"})


if __name__=="__main__":
    unittest.main()
