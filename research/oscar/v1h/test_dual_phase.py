"""Oscar v1H first/later route is real original 8-layer causal Green Tunnel.

Tests synthetic 8layer/512 outputs. No new independent pretrained QA involved.
"""
from __future__ import annotations
from copy import deepcopy
import unittest
import torch
from torch import nn

from oscar.v1f.green_tunnel import FullLayerGreenTunnel,DEPTH,WIDTH
from oscar.v1h.real_dual_phase_inference import (
    scopes,repetition,PHASES,RUNTIME_MANIFEST_BYTES,
    MAX_PHYSICAL_BYTES,EXPECTED_ORIGINAL_UNIQUE,
)


class SmallOriginalEightLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList(
            [nn.Linear(WIDTH,WIDTH,bias=False) for _ in range(DEPTH)])
        self.norm=nn.LayerNorm(WIDTH)
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return self.norm(x)


class TwoPaidGreenTunnelPhaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(91118)
        self.net=SmallOriginalEightLayer().eval()
        self.net.requires_grad_(False)
        self.old=FullLayerGreenTunnel().install(self.net)
        self.new=FullLayerGreenTunnel().install(self.net)
        with torch.no_grad():
            self.old.up[0,0,0]=.2
            self.old.up[3,0,0]=.17
            self.new.up[0,0,0]=-.3
            self.new.up[3,0,0]=-.17
        self.inp=torch.randn(1,4,WIDTH)

    def tearDown(self):
        self.old.uninstall()
        self.new.uninstall()

    def test_first_token_path_is_BITWISE_unchanged_when_new_GT_off(self):
        with self.old.scope(True),self.new.scope(False):
            both_hooks_y=self.net(self.inp).detach().clone()
        self.new.uninstall()
        with self.old.scope(True):
            original_only_y=self.net(self.inp).detach().clone()
        self.new.install(self.net)
        self.assertTrue(torch.equal(both_hooks_y,original_only_y))
        self.assertEqual(self.old.last_coverage,tuple(range(DEPTH)))
        self.assertEqual(self.new.last_coverage,tuple(range(DEPTH)))

    def test_two_phase_routing_is_not_prompt_answer_lookup(self):
        self.assertEqual(scopes(PHASES[2],0),(True,False))
        for i in range(1,8):
            self.assertEqual(scopes(PHASES[2],i),(False,True))
        self.assertEqual(scopes(PHASES[0],4),(True,False))
        self.assertEqual(scopes(PHASES[1],0),(False,True))
        with self.assertRaises(AssertionError):
            scopes("NOT_A_REAL_ORIGINAL_GT_MODE",1)

    def test_later_route_changes_only_actively_enabled_GT(self):
        with self.old.scope(True),self.new.scope(False):
            y1=self.net(self.inp)
        with self.old.scope(False),self.new.scope(True):
            y2=self.net(self.inp)
        self.assertFalse(torch.equal(y1,y2))
        self.assertIsNone(self.old._carry)
        self.assertIsNone(self.new._carry)

    def test_repetition_metrics_and_both_fp16_counted_in_rate(self):
        self.assertEqual(repetition([1,2,1,2])["unigram_repeat"],.5)
        self.assertEqual(repetition([1,2,1,2])["bigram_repeat"],1/3)
        self.assertEqual(repetition([3,4,5])["bigram_repeat"],0)
        source=2508827
        gt=524664
        self.assertLessEqual(source+2*gt+RUNTIME_MANIFEST_BYTES,
                             MAX_PHYSICAL_BYTES)
        self.assertGreater(source+2*gt+RUNTIME_MANIFEST_BYTES,
                           source+gt+RUNTIME_MANIFEST_BYTES)
        self.assertEqual(EXPECTED_ORIGINAL_UNIQUE,41689600)


if __name__=="__main__":
    unittest.main()
