"""Synthetic-only causal KV-prefill and decoded-token GT isolation tests."""
from __future__ import annotations

import unittest
import torch
from torch import nn

from calibri2.s103.test_shared_all_layer_green_bridge import (
    DecoderLayer, FakeTunnel,
)
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s115.cached_phase_gate import (
    VALID_PHASES,phase_enabled,frozen_cpu_phase_scope,
)


class ToyCached(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList(DecoderLayer(i) for i in range(12))
    def forward(self,input_ids,*,past_key_values=None,use_cache=True):
        hidden=input_ids
        for layer in self.model.layers:
            hidden,_=layer(hidden,past_key_values)
        return hidden


class TestCausalCachedPhaseMasks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(115)
        self.model=ToyCached()
        self.old=FakeTunnel().install(self.model)
        self.bridge=PromptConditionedDualTunnel(self.old).install(self.model)
        self.prompt=torch.tensor([[0.3,-0.8,1.2,0.5]],dtype=torch.float32)
        self.features=torch.linspace(-1,1,512)
        with torch.no_grad():
            self.bridge.coefficients[0,0,0]=0.2
            self.bridge.coefficients[1,0,0]=0.5
        self.cache=object()

    def tearDown(self):
        self.bridge.uninstall()
        self.old.uninstall()

    def test_prefill_vs_second_vs_all_masks_are_causally_isolated(self):
        with torch.inference_mode():
            with self.bridge.prompt_scope(None,None):
                base_0=self.model(input_ids=self.prompt,past_key_values=None,
                                  use_cache=True)
                base_1=self.model(input_ids=self.prompt,past_key_values=self.cache,
                                  use_cache=True)
            observed={}
            for phase in VALID_PHASES:
                with frozen_cpu_phase_scope(
                        self.model,self.bridge,"S93",self.features,phase=phase) as trace:
                    y0=self.model(input_ids=self.prompt,past_key_values=None,
                                  use_cache=True)
                    y1=self.model(input_ids=self.prompt,past_key_values=self.cache,
                                  use_cache=True)
                    self.assertEqual(trace["calls"],2)
                observed[phase]=(not torch.equal(y0,base_0),
                                 not torch.equal(y1,base_1))
        self.assertEqual(observed["all_steps"],(True,True))
        self.assertEqual(observed["prefill_only"],(True,False))
        self.assertEqual(observed["prefill_and_second"],(True,True))
        self.assertEqual(observed["decode_only"],(False,True))
        self.assertEqual(observed["second_only"],(False,True))
        self.assertIsNone(self.bridge._active_profile)
        self.assertEqual(len(self.model._forward_pre_hooks),0)

    def test_alt_profile_never_receives_s93_conditional_gains(self):
        with torch.inference_mode():
            with self.bridge.prompt_scope(None,None):
                baseline=self.model(
                    input_ids=self.prompt,past_key_values=None,use_cache=True)
            for route in ("S87","S91",None):
                with frozen_cpu_phase_scope(
                        self.model,self.bridge,route,None,
                        phase="all_steps"):
                    output=self.model(input_ids=self.prompt,
                                      past_key_values=None,use_cache=True)
                    self.assertTrue(torch.equal(baseline,output),route)

    def test_invalid_cache_behavior_and_exception_cleanup(self):
        with torch.inference_mode():
            for phase in ("unknown",):
                with self.assertRaises(ValueError):
                    with frozen_cpu_phase_scope(
                            self.model,self.bridge,"S93",self.features,phase=phase):
                        pass
            with self.assertRaises(AssertionError):
                with frozen_cpu_phase_scope(
                        self.model,self.bridge,"S93",self.features,
                        phase="prefill_only"):
                    self.model(input_ids=self.prompt,
                               past_key_values=self.cache,use_cache=True)
            self.assertEqual(len(self.model._forward_pre_hooks),0)
            self.assertIsNone(self.bridge._active_profile)
            with self.assertRaises(AssertionError):
                with frozen_cpu_phase_scope(
                        self.model,self.bridge,"S93",self.features,
                        phase="prefill_only"):
                    self.model(input_ids=self.prompt,
                               past_key_values=None,use_cache=False)
        self.assertEqual(len(self.model._forward_pre_hooks),0)

    def test_phase_mask_predeclared_semantics(self):
        self.assertEqual(
            [phase_enabled("prefill_only",i) for i in range(3)],
            [True,False,False])
        self.assertEqual(
            [phase_enabled("prefill_and_second",i) for i in range(3)],
            [True,True,False])
        self.assertEqual(
            [phase_enabled("decode_only",i) for i in range(3)],
            [False,True,True])
        self.assertEqual(
            [phase_enabled("second_only",i) for i in range(3)],
            [False,True,False])
        with self.assertRaises(ValueError):
            phase_enabled("all_steps",-1)


if __name__=="__main__":
    unittest.main(verbosity=2)
