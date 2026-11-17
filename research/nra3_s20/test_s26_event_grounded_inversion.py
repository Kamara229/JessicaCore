"""S26 event-scoped inversion role/polarity rescue: source-only geometry.

Does NOT execute original43M; physical original CPU inputs are CI only.
Exact original S22d model logits at zero residual, three paid parameters,
no annotation/oracle lexical template inside forward.
"""
from __future__ import annotations
import inspect
import unittest
import torch
from research.nra3_s18.s18c3_raw_russian_grammar_cues import FEATURES
from research.nra3_s19.paid_grammar_parser import CompactWordGrammar
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import (
    PaidSemanticEventGraph,full_parameter_count,semantic_parameter_count,
)
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    EventGroundedInversionGraph,PREV_NEG,LIKELY_DATIVE,EXTRA_FP16,
    ROLE_RECIPIENT,
)
from research.nra3_s20.s26_train256_event_grounded_matched40 import (
    PAID_PARENT,PAID_GRAPH,PAID_TOTAL,STEPS,SEED,
)
from research.nra3_s20.test_s22c_real43m_new256_source_separation import (
    fake_geometry_for_tests_only,
)
from research.nra3_s20.s22c_real43m_new256_graph_target_join import build_features


def parent():
    return PaidContextualEventScope(
        CompactWordGrammar().eval().requires_grad_(False)).eval().requires_grad_(False)


class S26EventGroundedInversionRescueContracts(unittest.TestCase):
    def test_exact_original_graph_zero_residual_parity_all_five_heads(self):
        self.assertEqual((PAID_PARENT,PAID_GRAPH,PAID_TOTAL,STEPS),
                         (9332,3052,12384,240))
        torch.manual_seed(SEED)
        baseline=PaidSemanticEventGraph(parent()).eval()
        torch.manual_seed(SEED)
        adjusted=EventGroundedInversionGraph(parent()).eval()
        self.assertEqual(semantic_parameter_count(adjusted),3052)
        self.assertEqual(full_parameter_count(adjusted),12384)
        self.assertEqual(EXTRA_FP16,3)
        self.assertEqual(tuple(adjusted.event_scoped_recipient_weights.tolist()),
                         (0.,0.,0.))
        texts,h,meta=fake_geometry_for_tests_only()
        examples=build_features(texts,h,meta)
        with torch.inference_mode():
            x=examples.raw_word_hidden320_and_raw18[:3]
            mask=examples.valid_word_mask[:3]
            before=baseline(x,mask)
            after=adjusted(x,mask)
        for key in ("event_status","role","argument_polarity",
                    "coreference","event_relation"):
            self.assertTrue(torch.equal(getattr(before,key),getattr(after,key)),key)
        self.assertEqual(adjusted.output_gate,0)
        self.assertFalse(adjusted.candidate_accepted or adjusted.activation_allowed)

    def test_forward_only_observed_russian_raw18_and_model_predicted_event_no_gold(self):
        self.assertEqual(FEATURES[PREV_NEG],"previous_word_negative_particle")
        self.assertEqual(FEATURES[LIKELY_DATIVE],"ending_dative_possible")
        import research.nra3_s20.s26_predicted_event_inversion_rescue as model_module
        code=inspect.getsource(model_module.EventGroundedInversionGraph.forward)
        self.assertIn("event_prob=1.-torch.softmax(base.event_status",code)
        self.assertIn("x[...,320+PREV_NEG]*x[...,320+LIKELY_DATIVE]",code)
        self.assertIn("torch.softmax(base.role",code)
        for no in ("gold", "train_ids", "held_ids", "source_graph",
                   "row_index", "lexical_family", "len(text)", "optimizer"):
            self.assertNotIn(no,code)
        self.assertIn("output_gate==0",inspect.getsource(
            model_module.EventGroundedInversionGraph.__init__))

    def test_nonzero_local_cue_changes_only_recipient_role_and_rejected_polarity(self):
        torch.manual_seed(SEED)
        model=EventGroundedInversionGraph(parent()).eval()
        with torch.no_grad():
            model.event_scoped_recipient_weights.copy_(torch.tensor([2.,2.,1.]))
        texts,h,meta=fake_geometry_for_tests_only()
        data=build_features(texts,h,meta)
        x=data.raw_word_hidden320_and_raw18[:4].clone()
        valid=data.valid_word_mask[:4]
        # Inject *only text-only binary features* to test architecture:
        # do not supply a gold predicate role or human target.
        self.assertTrue(bool(valid[0,1]))
        x[0,1,320+PREV_NEG]=1.
        x[0,1,320+LIKELY_DATIVE]=1.
        with torch.inference_mode():
            baseline=PaidSemanticEventGraph.forward(model,x,valid)
            after=model(x,valid)
        for key in ("event_status","coreference","event_relation"):
            self.assertTrue(torch.equal(getattr(baseline,key),getattr(after,key)),key)
        role_diff=after.role-baseline.role
        polar_diff=after.argument_polarity-baseline.argument_polarity
        self.assertTrue(torch.equal(
            role_diff[...,:ROLE_RECIPIENT],
            torch.zeros_like(role_diff[...,:ROLE_RECIPIENT])))
        self.assertTrue(torch.equal(
            role_diff[...,ROLE_RECIPIENT+1:],
            torch.zeros_like(role_diff[...,ROLE_RECIPIENT+1:])))
        self.assertTrue(torch.equal(polar_diff[...,:2],torch.zeros_like(polar_diff[...,:2])))
        self.assertGreater(float(role_diff[...,ROLE_RECIPIENT].sum()),0.)
        self.assertGreater(float(polar_diff[...,2].sum()),0.)

    def test_three_fp16_residual_weights_physically_paid_roundtrip_and_inactive(self):
        m=EventGroundedInversionGraph(parent()).eval()
        self.assertEqual(semantic_parameter_count(m),3052)
        self.assertTrue(all(not p.requires_grad for p in m.parent.parameters()))
        with torch.no_grad():
            m.event_scoped_recipient_weights.copy_(torch.tensor([.5,-.25,.125]))
        b=m.physical_bytes()
        obj=EventGroundedInversionGraph.from_physical(b)
        self.assertEqual(obj.head_physical(),m.head_physical())
        self.assertTrue(torch.equal(obj.event_scoped_recipient_weights,
                                    m.event_scoped_recipient_weights))
        self.assertFalse(any(p.requires_grad for p in obj.parameters()))
        self.assertEqual(obj.output_gate,0)
        self.assertFalse(obj.candidate_accepted or obj.activation_allowed)


if __name__=="__main__":
    unittest.main()
