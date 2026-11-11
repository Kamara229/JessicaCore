"""S132 CI: 5 J7 cycles then bounded rank growth, exact zero parity."""
from __future__ import annotations
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from calibri2.s132.growth_schedule import (
    ActualJ7Cycle,authorize_growth,next_rank,planned_stages,
    architecture_params,START_RANK,MAX_RANK,may_grow_green_tunnel)
from calibri2.s132.progressive_core import (
    ProgressiveRouteCore,expand_preserving_previous,fixed_down_chunks,
    IDS_KEY,WEIGHTS_KEY)
from calibri2.s129.additive_core_expansion import AdditiveCoreExpansion,WIDTH,DEPTH

class Layer(nn.Module):
    def forward(self,h):return h+0.01,

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList(Layer() for _ in range(DEPTH))
    def forward(self,input_ids=None,past_key_values=None,use_cache=True):
        h=input_ids
        for l in self.model.layers:h=l(h)[0]
        return h

def evidence(n):
    return ActualJ7Cycle(
        cycle=n,old_456_checked=456,lost_previously_correct_full=0,
        lost_previously_correct_first=0,physical_bpp=.68,
        paired_cpu_ratio=.98,physical_reloaded=True,source_immutable=True,
        old_full_count=353,old_first_count=387,new_full_count=15,
        new_pairs_both_count=0,transaction_committed=True,
        transaction_rollback_verified=False)

class TestS132Progression(unittest.TestCase):
    def test_five_cycles_only_then_135m(self):
        stages=planned_stages()
        self.assertEqual(len(stages),28)
        self.assertEqual(stages[0]['training_core_params'],39_821_120)
        self.assertEqual(stages[0]['next_core_params'],43_261_760)
        self.assertEqual(stages[-1]['next_core_params'],134_999_360)
        self.assertEqual(stages[-1]['next_rank'],MAX_RANK)
        self.assertEqual(authorize_growth([evidence(i) for i in range(1,6)],448),896)
        with self.assertRaises(AssertionError):
            authorize_growth([evidence(i) for i in range(1,5)],448)
        with self.assertRaises(ValueError):
            next_rank(448,completed_cycles=4)
        self.assertFalse(may_grow_green_tunnel(
            new_correct_full=0,forgotten=0,extra_serialized_bytes=100,
            projected_bpp=.69,paired_cpu_ratio=1.0))
        self.assertFalse(may_grow_green_tunnel(
            new_correct_full=1,forgotten=0,extra_serialized_bytes=100,
            projected_bpp=.701,paired_cpu_ratio=1.0))

    def test_no_growth_after_invalid_guard(self):
        v=[evidence(i) for i in range(1,6)]
        v[2]=ActualJ7Cycle(**{**v[2].__dict__,'lost_previously_correct_full':1})
        with self.assertRaises(AssertionError):
            authorize_growth(v,448)

    def test_preserve_exact_original_seed_down_and_old_up(self):
        historical=AdditiveCoreExpansion()
        old=ProgressiveRouteCore(rank=448)
        self.assertTrue(torch.equal(old.down,historical.down))
        self.assertEqual(old.architectural_new_parameter_count(),3_440_640)
        with torch.no_grad():old.up[0,0,0]=0.0625
        new=expand_preserving_previous(old,896)
        self.assertEqual(new.architectural_new_parameter_count(),6_881_280)
        self.assertTrue(torch.equal(new.up[:,:,:448],old.up))
        self.assertTrue(torch.equal(new.down[:,:448],old.down))
        self.assertEqual(len(new.active_indices),48)
        self.assertEqual(new.active_indices[24],448)
        data=new.export_active_fp16()
        self.assertEqual(data[WEIGHTS_KEY].shape,(12,320,48))
        self.assertEqual(int(data[IDS_KEY][24]),448)
        replica=ProgressiveRouteCore(rank=896)
        replica.load_active_fp16(data)
        self.assertTrue(torch.equal(replica.export_active_fp16()[WEIGHTS_KEY],data[WEIGHTS_KEY]))
        self.assertTrue(torch.equal(replica.down,new.down))

    def test_growing_new_channels_preserves_previous_answers(self):
        x=torch.randn((1,4,WIDTH),generator=torch.Generator().manual_seed(132))
        model=Core()
        bridge=SimpleNamespace(_active_profile='S93')
        old=ProgressiveRouteCore(rank=448).bind_original_route_bridge(bridge).install(model)
        with torch.no_grad():old.up[0,0,0]=0.03125
        try:
            with old.prefill_only(model):
                before=model(input_ids=x,past_key_values=None)
        finally:old.uninstall()
        next_core=expand_preserving_previous(old,896)
        next_core.bind_original_route_bridge(bridge).install(model)
        try:
            with next_core.prefill_only(model):
                after=model(input_ids=x,past_key_values=None)
            self.assertTrue(torch.allclose(before,after,rtol=0,atol=1e-6))
            with torch.no_grad():next_core.up[0,0,448]=.125
            with next_core.prefill_only(model):
                changed=model(input_ids=x,past_key_values=None)
            self.assertFalse(torch.equal(after,changed))
        finally:next_core.uninstall()

if __name__=="__main__":unittest.main()
