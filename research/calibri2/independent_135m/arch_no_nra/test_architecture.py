"""CPU synthetic contracts for C2 component hooks before expensive 135M run."""
import unittest
import torch
from torch import nn
from types import SimpleNamespace

from calibri2.independent_135m.arch_no_nra.architecture import (
    InternalGreenTunnel, FullDepthGreenTunnel,
    ParallelSourceSlots, PromptOnlyRouter
)


class TinyCausal(nn.Module):
    def __init__(self,width=32,depth=8):
        super().__init__()
        self.config=SimpleNamespace(hidden_size=width,num_hidden_layers=depth)
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Linear(width,width,bias=False)
                                         for _ in range(depth)])
        self.output=nn.Linear(width,16,bias=False)
    def forward(self,x):
        for layer in self.model.layers:
            x=layer(x)
        return self.output(x)


class TransferContracts(unittest.TestCase):
    def test_true_local_and_full_layer_zero_identity(self):
        torch.manual_seed(114)
        model=TinyCausal()
        x=torch.randn(2,4,32)
        baseline=model(x)
        local=InternalGreenTunnel(32,8)
        full=FullDepthGreenTunnel(32,8)
        local.install(model)
        full.install(model)
        self.assertTrue(torch.equal(baseline,model(x)))
        self.assertEqual(full.last_full_coverage,tuple(range(8)))
        self.assertEqual(local.positions,(2,6))
        self.assertEqual(len(list(local.parameters()))>0,True)
        with torch.no_grad():
            full.gates[3]=0.1
        new=model(x)
        self.assertFalse(torch.equal(baseline,new))

    def test_source_bound_parallel_slots(self):
        torch.manual_seed(114)
        head=ParallelSourceSlots(32,rank=8,slots=2)
        states=torch.randn(1,7,32)
        token=torch.tensor([[11,12,13,14,15,16,17]])
        result=head(states,token)
        selected=result["chosen_source_span_ids"][0].tolist()
        self.assertIn(selected,
            [token[0,j:j+2].tolist() for j in range(6)])
        self.assertFalse(head.enabled)

    def test_prompt_router_untrained_default(self):
        net=PromptOnlyRouter()
        r=net.suggest([11,12,13])
        self.assertEqual(r["active_profile"],0)
        self.assertFalse(r["router_trained_and_armed"])


if __name__=="__main__":
    unittest.main()
