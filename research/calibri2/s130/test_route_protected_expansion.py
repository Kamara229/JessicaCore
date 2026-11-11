"""S130 route-isolated expansion and actual J7 policy regression tests."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from calibri2.s130.route_protected_expansion import RouteIsolatedAdditiveCore
from calibri2.s129.additive_core_expansion import WIDTH,DEPTH,EXTRA_CORE_PARAMS


class MockLayer(nn.Module):
    def forward(self,h):
        return (h+0.01,)


class MockCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList(MockLayer() for _ in range(DEPTH))
    def forward(self,input_ids=None,past_key_values=None,use_cache=True):
        h=input_ids
        for layer in self.model.layers:
            h=layer(h)[0]
        return h


class TestRouteIsolation(unittest.TestCase):
    def test_39m_s93_only_and_cache_bypass(self):
        model=MockCore()
        x=torch.randn((1,5,WIDTH),generator=torch.Generator().manual_seed(130))
        base=model(input_ids=x,past_key_values=None)
        bridge=SimpleNamespace(_active_profile=None)
        expansion=RouteIsolatedAdditiveCore().bind_original_route_bridge(bridge).install(model)
        try:
            self.assertEqual(expansion.architectural_new_parameter_count(),EXTRA_CORE_PARAMS)
            with expansion.prefill_only(model):
                zero=model(input_ids=x,past_key_values=None)
            self.assertTrue(torch.equal(base,zero))
            self.assertEqual(expansion.last_prefill_coverage,tuple(range(DEPTH)))
            with torch.no_grad():
                expansion.up[0,0,0]=0.125
            bridge._active_profile="S87"
            with expansion.prefill_only(model):
                old_route=model(input_ids=x,past_key_values=None)
            self.assertTrue(torch.equal(base,old_route))
            bridge._active_profile="S91"
            with expansion.prefill_only(model):
                other_route=model(input_ids=x,past_key_values=None)
            self.assertTrue(torch.equal(base,other_route))
            bridge._active_profile="S93"
            with expansion.prefill_only(model):
                specialized=model(input_ids=x,past_key_values=None)
            self.assertFalse(torch.equal(base,specialized))
            with expansion.prefill_only(model):
                continuation=model(input_ids=x,past_key_values=((torch.zeros(1),),))
            self.assertTrue(torch.equal(base,continuation))
        finally:
            expansion.uninstall()

    def test_inactive_zero_and_physical_roundtrip(self):
        x=RouteIsolatedAdditiveCore()
        with torch.no_grad():
            x.up[0,0,0]=0.01
        side=x.export_active_fp16()
        y=RouteIsolatedAdditiveCore()
        y.load_active_fp16(side)
        self.assertTrue(torch.equal(y.down,x.down))
        self.assertTrue(torch.equal(y.export_active_fp16()[next(iter(side))],side[next(iter(side))]))


if __name__=="__main__":
    unittest.main()
