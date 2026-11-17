"""S24 original43M gold-blind raw-scope residual: zero-gain parity and guards.

Source-only unit inputs use FAKE geometry; actual original43M+GT physical
training requires the separately verified S22c archive. Not independent EVAL.
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
from research.nra3_s20.s24_neural_role_local_negation_scope import (
    ScopedRejectionGraph,PREV_NEG,LIKELY_DATIVE,EXTRA_SCOPE_FP16,
)
from research.nra3_s20.s24_train256_scoped_rejection_matched40 import (
    PAID_GRAPH,PAID_PARENT,PAID_TOTAL,STEPS,SEED,
)
from research.nra3_s20.test_s22c_real43m_new256_source_separation import (
    fake_geometry_for_tests_only,
)
from research.nra3_s20.s22c_real43m_new256_graph_target_join import build_features


def frozen_test_parent():
    return PaidContextualEventScope(
        CompactWordGrammar().eval().requires_grad_(False)).eval().requires_grad_(False)


class S24ScopedRejectionOnlyContracts(unittest.TestCase):
    def test_baseline_exact_zero_scope_initial_logits_for_all_five_heads(self):
        self.assertEqual((PAID_PARENT,PAID_GRAPH,PAID_TOTAL,STEPS),
                         (9332,3052,12384,240))
        torch.manual_seed(SEED)
        base=PaidSemanticEventGraph(frozen_test_parent()).eval()
        torch.manual_seed(SEED)
        new=ScopedRejectionGraph(frozen_test_parent()).eval()
        self.assertEqual(semantic_parameter_count(base),3049)
        self.assertEqual(semantic_parameter_count(new),3052)
        self.assertEqual(full_parameter_count(new),12384)
        self.assertEqual(EXTRA_SCOPE_FP16,3)
        self.assertEqual(tuple(new.local_scope_weights.tolist()),(0.,0.,0.))
        texts,h,meta=fake_geometry_for_tests_only()
        x=build_features(texts,h,meta)
        with torch.inference_mode():
            old=base(x.raw_word_hidden320_and_raw18[:3],x.valid_word_mask[:3])
            after=new(x.raw_word_hidden320_and_raw18[:3],x.valid_word_mask[:3])
        for task in ("event_status","role","argument_polarity",
                     "coreference","event_relation"):
            self.assertTrue(torch.equal(getattr(old,task),getattr(after,task)),task)
        self.assertEqual(new.output_gate,0)
        self.assertFalse(new.candidate_accepted)
        self.assertFalse(new.activation_allowed)

    def test_genuine_model_uses_only_raw_surfaces_and_predicted_role_no_oracle(self):
        self.assertEqual(FEATURES[PREV_NEG],"previous_word_negative_particle")
        self.assertEqual(FEATURES[LIKELY_DATIVE],"ending_dative_possible")
        import research.nra3_s20.s24_neural_role_local_negation_scope as module
        s=inspect.getsource(module.ScopedRejectionGraph.forward)
        self.assertIn("torch.softmax(base.role",s)
        self.assertIn("prev_negative=x[...,320+PREV_NEG]",s)
        self.assertIn("dative=x[...,320+LIKELY_DATIVE]",s)
        self.assertNotIn("gold[",s)
        self.assertNotIn("graph.",s)
        self.assertNotIn("role_targets",s)
        self.assertNotIn("optimizer.step(",s)
        self.assertNotIn("from_pretrained",s)
        self.assertNotIn("EVAL128",s[s.index("class ScopedRejectionGraph"):])

    def test_trained_residual_changes_only_rejected_logit_and_is_scope_local(self):
        torch.manual_seed(SEED)
        model=ScopedRejectionGraph(frozen_test_parent()).eval()
        with torch.no_grad():
            model.local_scope_weights.copy_(torch.tensor([1.5,.5,1.]))
        texts,h,meta=fake_geometry_for_tests_only()
        x=build_features(texts,h,meta)
        raw=x.raw_word_hidden320_and_raw18[:5].clone()
        valid=x.valid_word_mask[:5]
        # Controlled *input-only* negative particle cue, not a gold role.
        # The first five coordination templates contain no 'не'.
        raw[0,1,320+PREV_NEG]=1.
        # Compare model base logits using same physical weights.
        with torch.inference_mode():
            parent=PaidSemanticEventGraph.forward(model,raw,valid)
            scoped=model(raw,valid)
        for task in ("event_status","role","coreference","event_relation"):
            self.assertTrue(torch.equal(getattr(parent,task),getattr(scoped,task)),task)
        self.assertTrue(torch.equal(parent.argument_polarity[...,:2],
                                    scoped.argument_polarity[...,:2]))
        diff=scoped.argument_polarity[...,2]-parent.argument_polarity[...,2]
        self.assertTrue(bool(torch.isfinite(diff).all()))
        self.assertTrue(bool((diff>=0).all()))
        self.assertTrue(bool((diff[~valid.unsqueeze(2).expand_as(diff)]==0).all()))
        self.assertGreater(float(diff.sum()),0.)

    def test_three_trainable_parameters_physically_roundtrip_and_frozen_parent(self):
        model=ScopedRejectionGraph(frozen_test_parent()).eval()
        self.assertEqual(sum(p.numel() for p in model.semantic_params()),3052)
        self.assertTrue(all(not p.requires_grad for p in model.parent.parameters()))
        with torch.no_grad():
            model.local_scope_weights.copy_(torch.tensor([.5,-.25,.125]))
        blob=model.physical_bytes()
        copy=ScopedRejectionGraph.from_physical(blob)
        self.assertEqual(copy.head_physical(),model.head_physical())
        self.assertTrue(torch.equal(copy.local_scope_weights,
                                    model.local_scope_weights))
        self.assertFalse(any(p.requires_grad for p in copy.parameters()))
        self.assertEqual(copy.output_gate,0)
        self.assertFalse(copy.candidate_accepted)


if __name__=="__main__":
    unittest.main()
