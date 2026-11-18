"""Six synthetic S65 contract tests; NEVER use new S65 manual gold."""
import unittest
import torch
from research.nra3_s20.test_s63_quote_outer_strictbpe_contracts import (
    case,TEXTS,
)
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import (
    S63QuoteOuterSafe,quote_overlay,
)
from research.nra3_s20.s62_joint_event_target_beam2166 import S62JointEventRecipient
from research.nra3_s20.s65_factuality_attributed_event259 import (
    S65FactualityJoint,discourse_features,FEATURE,PAID,
)

def mock():
    beam,unsafe=case()
    x=discourse_features(quote_overlay(beam,TEXTS,unsafe),TEXTS)
    s62=S62JointEventRecipient().eval().requires_grad_(False)
    s63=S63QuoteOuterSafe().eval().requires_grad_(False)
    return x,s62,s63

class TestS65Factual(unittest.TestCase):
    def test_259_fp16_abi_and_reload_no_old_mutation(self):
        self.assertEqual((FEATURE,PAID),(36,259))
        m=S65FactualityJoint()
        with torch.no_grad():
            m.event[1]=.125;m.none[4]=-.25;m.classifier[2,3]=.5
        fp=m.fp16()
        self.assertEqual(fp.shape,(259,))
        x=S65FactualityJoint.from_fp16(fp)
        self.assertTrue(torch.equal(x.fp16(),fp))
        self.assertTrue(all(not p.requires_grad for p in x.parameters()))

    def test_source_only_features_original_quote_and_span(self):
        x,_,_=mock()
        self.assertEqual(tuple(x.factual_source.shape),(2,8,36))
        self.assertTrue(bool(torch.isfinite(x.factual_source).all()))
        self.assertFalse(bool(x.strict_native_or_quote_unsafe.any()))
        self.assertEqual(float(x.factual_source[0,1,6]),1.)
        self.assertFalse(x.factual_source.requires_grad)

    def test_strict_original_BPE_failure_prevents_link(self):
        beam,unsafe=case()
        unsafe[1,2]=True
        x=discourse_features(quote_overlay(beam,TEXTS,unsafe),TEXTS)
        p=S65FactualityJoint().eval().requires_grad_(False)
        old2=S62JointEventRecipient().eval().requires_grad_(False)
        old3=S63QuoteOuterSafe().eval().requires_grad_(False)
        result=p.decode(x,old2,old3)
        self.assertEqual(int(result["directed_recipient"][1].sum()),0)
        self.assertFalse(result["streaming_causality_certified"])
        self.assertEqual(result["output_gate"],0)

    def test_global_source_and_target_single_safe_link(self):
        x,a,b=mock()
        m=S65FactualityJoint().eval().requires_grad_(False)
        result=m.decode(x,a,b)
        self.assertLessEqual(int(result["directed_recipient"].sum()),2)
        self.assertFalse(bool((result["directed_rejected"]&
                              ~result["directed_recipient"]).any()))

    def test_three_class_factuality_and_polar_head_grad(self):
        x,a,b=mock()
        m=S65FactualityJoint()
        score,polar,classes=m(x,a,b)
        n,k,w1=score.shape
        self.assertEqual(classes.shape,(2,8,3))
        slots=torch.tensor([1,1])
        ix=torch.tensor([next(j for j in range(w1-1)
                           if bool(x.older.original.target_safe[i,1,j]))
                         for i in range(2)])
        loss=torch.nn.functional.cross_entropy(
            score.reshape(n,-1),slots*w1+ix)
        loss+=torch.nn.functional.cross_entropy(
            classes[torch.arange(2),slots],torch.tensor([1,2]))
        loss+=torch.nn.functional.binary_cross_entropy_with_logits(
            polar[torch.arange(2),slots,ix],torch.tensor([0.,1.]))
        loss.backward()
        self.assertTrue(all(z.grad is not None and
                            bool(torch.isfinite(z.grad).all())
                            for z in m.parameters()))

    def test_activation_fails_closed_before_forward(self):
        x,a,b=mock()
        m=S65FactualityJoint()
        m.activation_allowed=True
        with self.assertRaisesRegex(ValueError,"strict research-only"):
            m.decode(x,a,b)
        m.activation_allowed=False
        m.output_gate=1
        with self.assertRaisesRegex(ValueError,"strict research-only"):
            m.decode(x,a,b)

if __name__=="__main__":unittest.main()
