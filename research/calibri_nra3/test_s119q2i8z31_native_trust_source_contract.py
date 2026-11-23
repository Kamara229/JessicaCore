"""Z31 strict source, actual native49152 KL trust and no global bias train."""
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from collections import Counter
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS
from research.calibri_nra3.s119q2i8z31_84legaltrain_new14dev_bounded_context_source import (
    rows,verify,digest,EPOCHS,NATIVE_HEAD_GAIN,TRAIN_LR,
    NATIVE_REFERENCE_KL,CONTEXT_PREGAIN_RESIDUAL_BOUND,
)
from research.calibri_nra3.s119q2i8z31_bounded_contextual_head_trust_contract import (
    VOCAB,HIDDEN,frozen_new_head_params,
    bounded_frozen_original_native_delta,trust_kl_after_same_paid_forward,
)
class NativeHeadShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Linear(HIDDEN,4,bias=False)
        self.up=nn.Linear(4,VOCAB,bias=False)
        self.output_bias=nn.Parameter(torch.zeros(VOCAB))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
    def forward(self,x):
        return self.up(F.gelu(self.down(F.layer_norm(x,(HIDDEN,)))))+self.output_bias

class TestZ31FrozenTrustRegion(unittest.TestCase):
    def test_precommitted_only_legal_train_and_fresh_dev(self):
        p=verify()
        self.assertEqual((p["prior_legal_TRAIN84"],p["brandNEW_author_synthetic_DEV14"]),(84,14))
        self.assertTrue(p["new_DEV_disjoint_all_prior_spent_including_Z28"])
        self.assertTrue(p["no_previous_SPENT_DEV_in_TRAIN"])
        self.assertEqual(p["TRAIN_SHA256"],digest(rows("TRAIN")))
        self.assertEqual(p["NEW_DEV_SHA256"],digest(rows("DEV")))
        self.assertEqual(Counter(r["kind"] for r in rows("TRAIN")),{k:12 for k in KINDS})
        self.assertEqual(Counter(r["kind"] for r in rows("DEV")),{k:2 for k in KINDS})
        self.assertEqual((EPOCHS,TRAIN_LR,NATIVE_HEAD_GAIN,NATIVE_REFERENCE_KL,
                          CONTEXT_PREGAIN_RESIDUAL_BOUND),(2,.004,.25,1.,2.))
        for invalid in ("GOLD","J7","EVAL128","TEST","TRAINDEV"):
            with self.assertRaises(ValueError):rows(invalid)
    def test_exact_frozen_vocab_bias_only_lowrank_learned(self):
        old=NativeHeadShape()
        new=NativeHeadShape()
        new.load_state_dict(old.state_dict())
        params=frozen_new_head_params(new,old)
        self.assertEqual(sum(p.numel() for p in params),197888)
        self.assertEqual(new.output_bias.numel(),49152)
        self.assertFalse(new.output_bias.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in old.parameters()))
        self.assertTrue(all(p.requires_grad for p in params))
    def test_same_real_head_hidden_single_reference_no_extra_original_call(self):
        torch.manual_seed(43)
        old=NativeHeadShape()
        new=NativeHeadShape()
        new.load_state_dict(old.state_dict())
        params=frozen_new_head_params(new,old)
        h=torch.randn(1,1,HIDDEN)
        with bounded_frozen_original_native_delta(new,old) as cap:
            zero=new(h)
        self.assertEqual(len(cap),1)
        self.assertTrue(torch.allclose(zero,old(h)))
        self.assertEqual(cap[0]["original43M_extra_decoder_calls"],0)
        # Force a different contextual projection with original global bias fixed.
        with torch.no_grad():
            new.up.weight.add_(torch.randn_like(new.up.weight)*0.5)
        with bounded_frozen_original_native_delta(new,old) as cap:
            delta=new(h)
        self.assertEqual(len(cap),1)
        self.assertLessEqual(float(cap[0]["bounded_contextual_difference"].abs().max()),2.000001)
        self.assertTrue(torch.isfinite(delta).all())
        native_input_base=torch.randn(1,1,VOCAB)
        paid_native_logits=(native_input_base+NATIVE_HEAD_GAIN*delta)[0]
        loss,meta=trust_kl_after_same_paid_forward(paid_native_logits,cap[0],old,new)
        self.assertGreater(meta["full49152_exact_reference_KL"],0.)
        self.assertEqual(meta["new_original_S142_model_forwards_for_reference"],0)
        self.assertTrue(meta["all_correct_native_token_candidates_unmasked"])
        # Reconstruction should recover exact old native result from SAME forward.
        reference=paid_native_logits.detach()-.25*cap[0]["bounded_contextual_difference"].detach()[0]
        exact_old=native_input_base[0]+.25*old(h).detach()[0]
        self.assertTrue(torch.allclose(reference,exact_old,atol=2e-6))
        loss.backward()
        self.assertGreater(float(new.up.weight.grad.abs().sum()),0.)
        self.assertIsNone(new.output_bias.grad)
        self.assertIsNone(old.up.weight.grad)
        self.assertIsNone(old.down.weight.grad)
        self.assertTrue(torch.isfinite(new.down.weight.grad).all())
    def test_dont_train_original_head_as_new_head(self):
        old=NativeHeadShape()
        with self.assertRaises(ValueError):
            frozen_new_head_params(old,old)
if __name__=="__main__":
    unittest.main()
