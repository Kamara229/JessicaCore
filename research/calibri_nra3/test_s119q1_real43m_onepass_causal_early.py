"""S119q1 ONE-PASS original-decoder hook ABI tests: synthetic ONLY."""
import unittest
import torch
from torch import nn
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    EARLY_LAYERS, PLACEHOLDER_MAPPING,
    build_early_causal_view,single_pass_early_intervention,
)


class FakeLayer(nn.Module):
    def __init__(self,index):
        super().__init__()
        self.index=index
        self.calls=0
        self.received=None
    def forward(self,x):
        self.calls+=1
        self.received=x.detach().clone()
        return (x+float(self.index+1),)


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([FakeLayer(i) for i in range(12)])
    def forward(self,x):
        for layer in self.model.layers:
            x=layer(x)[0]
        return x


class Adapter(nn.Module):
    output_gate=0
    candidate_accepted=False
    activation_allowed=False
    def __init__(self):
        super().__init__()
        self.midlayer_gates=nn.Parameter(torch.tensor([.1,.2,.3]))
        self.previous=None
    def forward(self,h,*,contextual):
        assert contextual
        self.previous=h.detach().clone()
        return h[:,5,-1,:]*torch.tanh(self.midlayer_gates[0])


class S119Q1EarlySinglePassControls(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119129)
        self.core=FakeCore()
        self.adapter=Adapter()
        self.x=torch.ones(1,3,320)

    def test_01_exact_six_real_early_layers_and_predeclared_placeholder_mapping(self):
        d={i:torch.full((1,3,320),float(i)) for i in EARLY_LAYERS}
        h=build_early_causal_view(d)
        self.assertEqual(h.shape,(1,12,3,320))
        self.assertEqual(PLACEHOLDER_MAPPING,
                         (0,1,2,3,4,5,5,5,5,5,5,5))
        for slot,real in enumerate(PLACEHOLDER_MAPPING):
            self.assertTrue(torch.equal(h[:,slot],d[real]))

    def test_02_cannot_ingest_future_layer_6_or_missing_layer(self):
        d={i:torch.zeros(1,3,320) for i in range(7)}
        with self.assertRaisesRegex(GateFailure,"exact early"):
            build_early_causal_view(d)
        del d[6];del d[0]
        with self.assertRaisesRegex(GateFailure,"exact early"):
            build_early_causal_view(d)

    def test_03_one_real_forward_after_layer5_updates_later_states(self):
        baseline=self.core(self.x).detach().clone()
        pre=[layer.calls for layer in self.core.model.layers]
        original_hooks=[set(layer._forward_hooks)
                        for layer in self.core.model.layers[:6]]
        with torch.enable_grad(),single_pass_early_intervention(
            self.core,self.adapter) as witness:
            new=self.core(self.x)
        self.assertTrue(torch.isfinite(new).all())
        self.assertFalse(torch.equal(new.detach(),baseline))
        self.assertTrue(all(layer.calls==pre[i]+1 for i,layer in
                            enumerate(self.core.model.layers)))
        self.assertEqual(witness["candidate_forward_calls"],1)
        self.assertEqual(set(witness["early_layer_calls"]),set(range(6)))
        self.assertTrue(torch.equal(
            self.core.model.layers[6].received,
            self.core.model.layers[5].received+6.+
            torch.cat([torch.zeros(1,2,320),(
                self.adapter.previous[:,5,-1,:]*
                torch.tanh(self.adapter.midlayer_gates[0])).view(1,1,320)],1)))
        self.assertTrue(all(set(layer._forward_hooks)==original_hooks[i]
                            for i,layer in enumerate(self.core.model.layers[:6])))

    def test_04_zero_gate_exact_original_and_genuine_onepass_gradient(self):
        baseline=self.core(self.x)
        original_gate=self.adapter.midlayer_gates.detach().clone()
        with torch.no_grad():self.adapter.midlayer_gates.zero_()
        try:
            with torch.enable_grad(),single_pass_early_intervention(
                self.core,self.adapter):
                zero=self.core(self.x)
            self.assertTrue(torch.equal(zero.detach(),baseline.detach()))
        finally:
            with torch.no_grad():self.adapter.midlayer_gates.copy_(original_gate)
        with torch.enable_grad(),single_pass_early_intervention(
            self.core,self.adapter):
            changed=self.core(self.x)
        grad=torch.autograd.grad(changed[0,-1,0],
                                 self.adapter.midlayer_gates)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertNotEqual(float(grad[0]),0.)
        self.assertEqual(self.core.model.layers[5].calls,3)

    def test_05_hook_cleanup_on_core_error(self):
        before=[set(x._forward_hooks) for x in self.core.model.layers[:6]]
        with self.assertRaisesRegex(RuntimeError,"synthetic"):
            with single_pass_early_intervention(self.core,self.adapter):
                self.core.model.layers[0](self.x)
                raise RuntimeError("synthetic")
        for i,layer in enumerate(self.core.model.layers[:6]):
            self.assertEqual(set(layer._forward_hooks),before[i])

    def test_06_reject_second_original_forward_inside_same_candidate_scope(self):
        with self.assertRaisesRegex(GateFailure,"one real pass"):
            with single_pass_early_intervention(self.core,self.adapter):
                self.core(self.x)
                self.core(self.x)
        self.assertTrue(all(len(layer._forward_hooks)==0
                            for layer in self.core.model.layers[:6]))

    def test_07_fail_closed_on_activation_instead_of_testing_jessica(self):
        self.adapter.candidate_accepted=True
        with self.assertRaisesRegex(GateFailure,"unpromoted"):
            with single_pass_early_intervention(self.core,self.adapter):
                self.core(self.x)
        self.adapter.candidate_accepted=False
        self.adapter.output_gate=1
        with self.assertRaisesRegex(GateFailure,"unpromoted"):
            with single_pass_early_intervention(self.core,self.adapter):
                self.core(self.x)

    def test_08_strict_shape_finite_controls(self):
        cases=[
            {i:torch.zeros(1,1,320) for i in EARLY_LAYERS},
            {i:torch.zeros(1,3,319) for i in EARLY_LAYERS},
            {i:torch.zeros(1,3,320) for i in EARLY_LAYERS},
        ]
        cases[2][4][0,0,0]=float("nan")
        for d in cases:
            with self.subTest(shape=tuple(d[0].shape)):
                with self.assertRaisesRegex(GateFailure,"physical causal"):
                    build_early_causal_view(d)


if __name__=="__main__":
    unittest.main()
