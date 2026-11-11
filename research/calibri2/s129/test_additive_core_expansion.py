"""S129 isolated architecture and physical seed tests; no QA, no model files."""
import unittest
import torch
from torch import nn
from calibri2.s129.additive_core_expansion import (
    AdditiveCoreExpansion,ORIGINAL_CORE_PARAMS,EXTRA_CORE_PARAMS,
    EXPANDED_CORE_PARAMS,ACTIVE_RANK,WIDTH,DEPTH,UP_KEY,
)


class MockLayer(nn.Module):
    def forward(self,h):
        return (h+0.01,)


class TinyOriginalCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([MockLayer() for _ in range(DEPTH)])

    def forward(self,input_ids=None,past_key_values=None,use_cache=True):
        h=input_ids
        for layer in self.model.layers:
            h=layer(h)[0]
        return h


class TestS129Expansion(unittest.TestCase):
    def test_39m_explicit_additive_parameter_count_and_zero_parity(self):
        self.assertEqual(ORIGINAL_CORE_PARAMS,36_380_480)
        self.assertEqual(EXTRA_CORE_PARAMS,3_440_640)
        self.assertEqual(EXPANDED_CORE_PARAMS,39_821_120)
        model=TinyOriginalCore()
        x=torch.randn((1,5,WIDTH),generator=torch.Generator().manual_seed(137))
        original=model(input_ids=x,past_key_values=None)
        expansion=AdditiveCoreExpansion().install(model)
        try:
            self.assertEqual(expansion.architectural_new_parameter_count(),EXTRA_CORE_PARAMS)
            self.assertEqual(len(expansion.trainable_initial_parameters()),1)
            self.assertFalse(expansion.down.requires_grad)
            self.assertTrue(expansion.up.requires_grad)
            with expansion.prefill_only(model):
                zero=model(input_ids=x,past_key_values=None)
            self.assertTrue(torch.equal(original,zero))
            self.assertEqual(expansion.last_prefill_coverage,tuple(range(DEPTH)))
            first=expansion.export_active_fp16()
            self.assertEqual(first[UP_KEY].shape,(DEPTH,WIDTH,ACTIVE_RANK))
            replica=AdditiveCoreExpansion()
            replica.load_active_fp16(first)
            self.assertTrue(torch.equal(replica.down,expansion.down))
            self.assertTrue(torch.equal(replica.up,expansion.up))
        finally:
            expansion.uninstall()

    def test_new_weights_train_and_roundtrip_without_modifying_old(self):
        model=TinyOriginalCore()
        x=torch.randn((1,3,WIDTH),generator=torch.Generator().manual_seed(138))
        baseline=model(input_ids=x,past_key_values=None)
        exp=AdditiveCoreExpansion().install(model)
        try:
            optimizer=torch.optim.Adam(exp.trainable_initial_parameters(),lr=.001)
            optimizer.zero_grad()
            with exp.prefill_only(model):
                result=model(input_ids=x,past_key_values=None)
            result.square().mean().backward()
            self.assertGreater(int(torch.count_nonzero(exp.up.grad)),0)
            optimizer.step()
            exp.assert_inactive_zero()
            snapshot=exp.export_active_fp16()
            second_model=TinyOriginalCore()
            loaded=AdditiveCoreExpansion().install(second_model)
            try:
                loaded.load_active_fp16(snapshot)
                self.assertTrue(torch.equal(loaded.down,exp.down))
                self.assertTrue(torch.equal(loaded.export_active_fp16()[UP_KEY],snapshot[UP_KEY]))
                with loaded.prefill_only(second_model):
                    candidate=second_model(input_ids=x,past_key_values=None)
            finally:
                loaded.uninstall()
            with exp.prefill_only(model):
                changed=model(input_ids=x,past_key_values=None)
            self.assertFalse(torch.equal(baseline,changed))
            with exp.prefill_only(model):
                continued=model(input_ids=x,past_key_values=((torch.zeros(1),),))
            self.assertTrue(torch.equal(baseline,continued))
        finally:
            exp.uninstall()


if __name__=="__main__":
    unittest.main()
