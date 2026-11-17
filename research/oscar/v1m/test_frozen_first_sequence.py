"""Oscar v1M original-FP32 causal sequence and frozen FIRST GT assertions.

Synthetic test only. Authentic original 42M pretrained FP32, full source,
dual real FP16 sidecars and actual greedy are verified by GitHub CPU CI.
"""
from __future__ import annotations

import unittest
import torch
from torch import nn
from types import SimpleNamespace

from oscar.v1f.green_tunnel import FullLayerGreenTunnel,DEPTH,WIDTH
from oscar.v1m.real_frozen_first_sequence_later import (
    TRAIN,NEW_DEBUG,COHORTS,STEPS,
    context_from_continuation,weight_anchor,
)


class TinyOriginalEightDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList(
           [nn.Linear(WIDTH,WIDTH,bias=False) for _ in range(DEPTH)])
        self.norm=nn.LayerNorm(WIDTH)

    def forward(self,hidden):
        for layer in self.layers:
            hidden=layer(hidden)
        return self.norm(hidden)


class FrozenFirstSequenceContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)
        torch.set_num_threads(2)
        self.model=TinyOriginalEightDecoder().eval()
        self.model.requires_grad_(False)
        self.first=FullLayerGreenTunnel().install(self.model)
        self.later=FullLayerGreenTunnel().install(self.model)
        self.anchor=FullLayerGreenTunnel().install(self.model)
        self.anchor.requires_grad_(False)
        self.first.requires_grad_(False)
        with torch.no_grad():
            self.first.up[0,0,0]=.19
            self.later.up[4,0,0]=.09
            self.anchor.up[4,0,0]=.07

    def tearDown(self):
        for route in (self.anchor,self.later,self.first):
            route.uninstall()

    def test_precommitted_92_unique_training_and_44_distinct_public(self):
        self.assertEqual(len(TRAIN),92)
        self.assertEqual(len(set(TRAIN)),92)
        debug=sum((tuple(v) for v in COHORTS.values()),())
        self.assertEqual(len(debug),44)
        self.assertEqual(len(set(debug)),44)
        self.assertFalse(set(TRAIN)&set(debug))
        self.assertEqual(len(NEW_DEBUG),8)
        self.assertEqual(STEPS,112)
        self.assertEqual(len(set((i*37)%92 for i in range(92))),92)

    def test_correct_causal_prefix_includes_own_generated_before_last(self):
        prompt=torch.tensor([[1,9,7]],dtype=torch.long)
        tok=SimpleNamespace(eos_token_id=2)
        c=context_from_continuation(prompt,[11,5,88,2],tok)
        self.assertEqual(c.tolist(),[[1,9,7,11,5,88]])
        self.assertNotIn(2,c.tolist()[0])
        self.assertEqual(
            context_from_continuation(prompt,[2],tok).tolist(),
            prompt.tolist())
        with self.assertRaises(AssertionError):
            context_from_continuation(prompt,[3,2,6],tok)

    def test_FIRST_GT_unchanged_even_when_LATER_updated(self):
        x=torch.randn(1,5,WIDTH)
        with self.first.scope(True),self.later.scope(False),self.anchor.scope(False):
            before=self.model(x).detach().clone()
        with torch.no_grad():
            self.later.up[0,1,2]+=.021
            self.later.down[2,3,4]+=.018
        with self.first.scope(True),self.later.scope(False),self.anchor.scope(False):
            after=self.model(x).detach().clone()
        self.assertTrue(torch.equal(before,after))
        self.assertEqual(self.first.last_coverage,tuple(range(DEPTH)))
        self.assertIsNone(self.first.up.grad)

    def test_trust_anchor_flows_to_LATER_only(self):
        original=self.later.source_paid_tensors()
        with torch.no_grad():
            self.later.up[4,0,0]+=.03
            self.later.down[3,0,0]+=.02
        loss=weight_anchor(self.later,original)
        loss.backward()
        self.assertGreater(float(self.later.up.grad.norm()),0)
        self.assertGreater(float(self.later.down.grad.norm()),0)
        self.assertIsNone(self.first.up.grad)
        self.assertIsNone(self.anchor.up.grad)


if __name__=="__main__":
    unittest.main()
