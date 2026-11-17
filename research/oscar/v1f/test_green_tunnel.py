"""Oscar v1F full original eight-layer *causal inter-layer* GT ABI tests.

Pure synthetic CPU matrices. No pretrained teacher/QA/test source is loaded.
"""
from __future__ import annotations
import unittest
import torch
from torch import nn
from oscar.v1f.green_tunnel import (
    DEPTH,WIDTH,RANK,HEAD_RANK,VOCAB,
    FullLayerGreenTunnel,PaidHeadCorrection,
    merge_paid,restore_paid
)


class SimpleOriginalGeometryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList(
            [nn.Linear(WIDTH,WIDTH,bias=False) for _ in range(DEPTH)])
        self.norm=nn.LayerNorm(WIDTH)

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return self.norm(x)


class RecurrentOriginal8GTContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(154)
        torch.set_num_threads(2)
        self.model=SimpleOriginalGeometryModel()
        self.model.requires_grad_(False)
        self.gt=FullLayerGreenTunnel().install(self.model)
        self.head=PaidHeadCorrection()
        self.x=torch.randn(1,5,WIDTH)

    def tearDown(self):
        self.gt.uninstall()

    def test_exact_identity_when_new_residual_up_zero(self):
        y=self.model(self.x)
        self.assertEqual(self.gt.last_coverage,tuple(range(DEPTH)))
        with self.gt.scope(False):
            z=self.model(self.x)
        self.assertTrue(torch.equal(y,z))
        self.assertTrue(torch.count_nonzero(self.head(y))==0)

    def test_genuine_carry_from_prior_layer_activates_next(self):
        with torch.no_grad():
            self.gt.up[0,0,0]=.2
            self.gt.up[1,0,0]=.2
        a=self.model(self.x)
        with self.gt.scope(False):
            b=self.model(self.x)
        self.assertFalse(torch.equal(a,b))
        self.assertEqual(self.gt.last_coverage,tuple(range(DEPTH)))

    def test_real_all8_trainable_and_head_gradient_nonzero(self):
        out=self.model(self.x)
        loss=out.square().sum()+self.head(self.gt.latest_norm).square().sum()
        # Exact head up zero produces zero loss derivative; a teacher
        # CE-like linear loss explicitly tests all 32k paid output rows.
        loss=loss+self.head(self.gt.latest_norm)[...,:8].sum()
        loss.backward()
        self.assertGreater(float(self.gt.up.grad.norm()),0)
        self.assertGreater(float(self.head.up.grad.norm()),0)
        self.assertIsNone(self.model.layers[0].weight.grad)

    def test_physical_6_fp16_tensors_shapes_and_fresh_restore(self):
        with torch.no_grad():
            self.gt.up[3,0,0]=.03
            self.head.up[47,0]=.04
        raw=merge_paid(self.gt,self.head)
        self.assertEqual(len(raw),6)
        self.assertEqual(sum(x.numel()*x.element_size() for x in raw.values()),
                         8*2*512*32*2+8*2*2+
                         (32000*8+512*8)*2)
        self.assertTrue(all(x.dtype==torch.float16 for x in raw.values()))
        new_gt=FullLayerGreenTunnel()
        new_head=PaidHeadCorrection()
        restore_paid(new_gt,new_head,raw)
        self.assertEqual(new_gt.up[3,0,0].half(),self.gt.up[3,0,0].half())
        self.assertEqual(new_head.up[47,0].half(),self.head.up[47,0].half())
        wrong=dict(raw)
        wrong.pop(next(iter(wrong)))
        with self.assertRaises(AssertionError):
            restore_paid(new_gt,new_head,wrong)

    def test_per_forward_carry_and_context_not_reused(self):
        first=self.model(self.x).detach()
        second=self.model(self.x).detach()
        self.assertTrue(torch.equal(first,second))
        self.assertIsNone(self.gt._carry)


if __name__=="__main__":
    unittest.main()
