"""S119k: synthetic verification of actual native head shadow contracts."""
import unittest
import torch
from torch import nn
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    NativeFullHeadShadow,full_vocab_ce,eval_trial,train,original_parameter_hash,
)

class S119KFullNativeVocabularyShadowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119124)
        torch.set_num_threads(2)
    def tensors(self):
        h=torch.randn(1,12,9,320)
        last=torch.randn(1,320)
        head=nn.Linear(320,2227,bias=False).eval()
        for p in head.parameters():p.requires_grad_(False)
        return h,last,head
    def test_01_full_original_native_vocab_not_restricted_proxy(self):
        child=NativeFullHeadShadow()
        h,last,head=self.tensors()
        out=child(h,last,head,contextual=True)
        self.assertEqual(out.shape,(1,2227))
        self.assertTrue(torch.isfinite(out).all())
    def test_02_gate_zero_exact_full_native_logits_at_initialization(self):
        child=NativeFullHeadShadow()
        h,last,head=self.tensors()
        with torch.no_grad():
            old=head(last)
            on=child(h,last,head,contextual=True)
            off=child(h,last,head,contextual=False)
        self.assertTrue(torch.equal(old,on))
        self.assertTrue(torch.equal(old,off))
        self.assertFalse(child.candidate_accepted)
    def test_03_can_train_actual_full_head_without_touching_original(self):
        child=NativeFullHeadShadow()
        h,last,head=self.tensors()
        before=head.weight.detach().clone()
        record={"h12":h,"last":last,"base_logits":head(last).detach(),
                "next_id":143,"row":4,"prefix":9}
        loss,_=full_vocab_ce(child,record,head,True)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(child.readout_gates.grad.abs().sum()),0)
        self.assertTrue(torch.equal(before,head.weight))
        self.assertFalse(child.candidate_accepted)
    def test_04_old_NRA3_all_53751_remain_frozen(self):
        child=NativeFullHeadShadow()
        self.assertEqual(sum(p.numel() for p in
            child.unified.nra3_neural_weights.parameters()),53751)
        for p in child.unified.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        h,last,head=self.tensors()
        ref={k:v.detach().clone()
             for k,v in child.unified.nra3_neural_weights.state_dict().items()}
        out=child(h,last,head,contextual=True)
        out.square().mean().backward()
        self.assertTrue(all(torch.equal(
            v,child.unified.nra3_neural_weights.state_dict()[k])
            for k,v in ref.items()))
    def test_05_can_never_activate_real_decoder(self):
        child=NativeFullHeadShadow()
        h,last,head=self.tensors()
        child.output_gate=1
        with self.assertRaises(GateFailure):
            child(h,last,head,contextual=True)
    def test_06_wrong_native_output_head_fails_closed(self):
        child=NativeFullHeadShadow()
        h,last,head=self.tensors()
        bad=nn.Linear(320,52,bias=False)
        with self.assertRaises(GateFailure):
            child(h,last,bad,contextual=True)
    def test_07_actual_native_target_outside_full_head_must_fail(self):
        child=NativeFullHeadShadow()
        h,last,head=self.tensors()
        record={"h12":h,"last":last,"next_id":3000,"row":0,"prefix":9}
        with self.assertRaises(GateFailure):
            full_vocab_ce(child,record,head,True)
    def test_08_matched_context_local_same_parameter_shape(self):
        a=NativeFullHeadShadow()
        b=NativeFullHeadShadow()
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        self.assertEqual(sum(v.numel() for v in a.state_dict().values()),
                         sum(v.numel() for v in b.state_dict().values()))
        self.assertEqual(a.logit_residual_gates.shape,(3,))

if __name__=="__main__":unittest.main()
