"""S119m: fail-closed synthetic contracts and genuine midlayer gradient path.

Synthetic tests are NOT a substitute for the original 43M CPU training,
which runs separately under its physically SHA-pinned GitHub Actions step.
"""
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    NativeFullHeadShadow,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    LearnedOnPathInterlayer,research_only_midlayer_injection,
    HOOK_LAYER,
)

class FrozenToy12(nn.Module):
    """Only synthetic native 12-layer gradient path; NOT real Calibri."""
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([
            nn.Linear(320,320,bias=False) for _ in range(12)])
        self.embed=nn.Embedding(80,320)
        self.head=nn.Linear(320,1107,bias=False)
        self.eval().requires_grad_(False)
    def forward(self,ids):
        x=self.embed(ids)
        for layer in self.model.layers:
            x=x+0.015*layer(x)
        return self.head(x[:,-1])

class S119MRealOnpathInterlayerResearch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119126)
        torch.set_num_threads(2)
        self.native=FrozenToy12()
        self.adapter=LearnedOnPathInterlayer(
            NativeFullHeadShadow()).eval()

    def mock_original12(self):
        return torch.randn(1,12,8,320)

    def test_01_complete_old_and_new_neural_areas_in_one_module(self):
        self.assertEqual(HOOK_LAYER,5)
        self.assertEqual(sum(p.numel() for p in
            self.adapter.shadow.unified.nra3_neural_weights.parameters()),53751)
        self.assertFalse(any(p.requires_grad for p in
            self.adapter.shadow.unified.nra3_neural_weights.parameters()))
        self.assertEqual(self.adapter.midlayer_gates.shape,(3,))
        self.assertEqual(int(self.adapter.output_gate),0)
        self.assertFalse(self.adapter.candidate_accepted)

    def test_02_zero_interlayer_gate_identical_native_logits(self):
        ids=torch.randint(0,80,(1,8))
        expected=self.native(ids)
        delta=self.adapter(self.mock_original12(),contextual=True)
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        with research_only_midlayer_injection(self.native,self.adapter,delta):
            after=self.native(ids)
        self.assertTrue(torch.equal(expected,after),
                        "real later decoder layers must see identical initial hidden")
        self.assertEqual(len(self.native.model.layers[5]._forward_hooks),0)

    def test_03_nonzero_interlayer_truly_reaches_original_remaining_layers(self):
        ids=torch.randint(0,80,(1,8))
        original=self.native(ids)
        with torch.no_grad():
            self.adapter.midlayer_gates.fill_(0.40)
        delta=self.adapter(self.mock_original12(),contextual=True)
        with research_only_midlayer_injection(self.native,self.adapter,delta):
            changed=self.native(ids)
        self.assertFalse(torch.equal(original,changed))
        self.assertTrue(changed.requires_grad)
        loss=changed.square().mean()
        loss.backward()
        self.assertGreater(float(
            self.adapter.midlayer_gates.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None
                            for p in self.native.parameters()))

    def test_04_same_original_model_with_ctx_and_local_matched_params(self):
        h=self.mock_original12()
        self.adapter.shadow.eval()
        with torch.no_grad():
            c=self.adapter.delta_components(h,contextual=True)
            l=self.adapter.delta_components(h,contextual=False)
        self.assertEqual(c.shape,(1,3,320))
        self.assertEqual(l.shape,(1,3,320))
        self.assertTrue(torch.isfinite(c).all() and torch.isfinite(l).all())

    def test_05_denied_activation_without_research_scope(self):
        self.adapter.output_gate=1
        with self.assertRaises(GateFailure):
            self.adapter(self.mock_original12(),contextual=True)

    def test_06_no_inference_only_or_detach_on_midlayer_delta(self):
        d=self.adapter(self.mock_original12(),contextual=True)
        self.assertTrue(d.requires_grad)
        self.assertTrue(torch.isfinite(d).all())
        with torch.no_grad():
            d0=d.detach()
        with self.assertRaises(GateFailure):
            with research_only_midlayer_injection(self.native,self.adapter,d0):
                self.native(torch.randint(0,80,(1,8)))

    def test_07_frozen_original_weights_after_one_real_head_backward(self):
        h=self.mock_original12()
        ids=torch.randint(0,80,(1,8))
        old={n:p.detach().clone() for n,p in self.native.named_parameters()}
        with torch.no_grad():
            self.adapter.midlayer_gates.fill_(0.002)
        residual=self.adapter(h,contextual=True)
        with research_only_midlayer_injection(self.native,self.adapter,residual):
            out=self.native(ids)
        F.cross_entropy(out,torch.tensor([15])).backward()
        self.assertTrue(all(torch.equal(old[n],p)
                        for n,p in self.native.named_parameters()))
        self.assertTrue(self.adapter.shadow.unified.core_growth.up.grad is not None)

    def test_08_backprop_gate_fail_if_hook_not_invoked(self):
        d=self.adapter(self.mock_original12(),contextual=True)
        with self.assertRaises(GateFailure):
            with research_only_midlayer_injection(self.native,self.adapter,d):
                pass

if __name__=="__main__":
    unittest.main()
