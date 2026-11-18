"""S33 source contract: bitwise event/relation protection and TRAIN-only scope.

Source-only tests never claim genuine frozen43M CPU performance; only
GitHub Actions physical S22c/S19d shard execution can establish it.
"""
from __future__ import annotations

import inspect
import unittest
import torch

from research.nra3_s19.paid_grammar_parser import CompactWordGrammar
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s22c_real43m_new256_graph_target_join import build_features
from research.nra3_s20.test_s22c_real43m_new256_source_separation import fake_geometry_for_tests_only
from research.nra3_s20.s33_head_only_precision_preserve_s31_relation import (
    fixed_train_S33,S33_ONLY_HEAD_STEPS,S33_HEAD_FINETUNE_LR,S33_STAGE,
    FOLDS,PREDECLARED_STEPS,PAID_TOTAL,accepted_recipient_and_hard_false_pair_loss,
)
from research.nra3_s20.s31_train256_rare_pair_margin_probe import fixed_train_S31
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import splits
from research.nra3_s20.s28_real43m_trainonly_determinism_probe import canonical_paid_semantic_fp16_sha


class S33ExactFrozenRelationProtection(unittest.TestCase):
    def test_source_fixed_schedule_and_no_added_fp16_parameters(self):
        self.assertEqual(PAID_TOTAL,12384)
        self.assertEqual(PREDECLARED_STEPS,240)
        self.assertEqual(S33_ONLY_HEAD_STEPS,64)
        self.assertEqual(S33_HEAD_FINETUNE_LR,.003)
        self.assertEqual(S33_STAGE,"S31_240_TRAIN_then_64_head_only_precise")
        self.assertEqual(FOLDS,(32,33,34,35,36,15,24,30))
        for fold in FOLDS:
            axis,group,train,held=splits()[fold]
            self.assertEqual((axis,group,len(train),len(held)),
                             ("CONSTRUCTION",fold-8,248,8))
            self.assertTrue(set(held).isdisjoint(set(train)))

    def test_head_only_allowlist_and_trainable_tensors_in_real_source(self):
        source=inspect.getsource(fixed_train_S33)
        for required in (
            "fixed_train_S31(model,raw,valid,gold)",
            "preserved_s31=EventGroundedInversionGraph.from_physical",
            "event_scoped_recipient_weights",
            '"role.weight","role.bias"',
            '"argument_polarity.weight","argument_polarity.bias"',
            "p.requires_grad_(name in allowed)",
            "all(p.grad is None for p in s31.parameters() if not p.requires_grad)",
            "torch.equal(p.detach(),initial_weights[name])",
            "canonical_paid_semantic_fp16_sha",
        ):
            self.assertIn(required,source)
        self.assertNotIn("candidate_accepted=True",source)
        self.assertNotIn("output_gate=1",source)

    def test_exact_unaffected_graph_heads_unchanged_if_only_role_and_polarity_fit(self):
        torch.manual_seed(320256)
        parent=PaidContextualEventScope(
            CompactWordGrammar().eval().requires_grad_(False)).eval().requires_grad_(False)
        m=EventGroundedInversionGraph(parent).eval()
        texts,hidden,meta=fake_geometry_for_tests_only()
        data=build_features(texts,hidden,meta)
        x=data.raw_word_hidden320_and_raw18[:4]
        valid=data.valid_word_mask[:4]
        with torch.inference_mode():
            prior=m(x,valid)
        # Simulated targeted head update on fake data: isolates role/pol
        # trainability and asserts the relation tensors do not change.
        for name,p in m.named_parameters():
            p.requires_grad_(name in {"role.weight","role.bias",
                  "argument_polarity.weight","argument_polarity.bias",
                  "event_scoped_recipient_weights"})
        with torch.no_grad():
            m.role.bias[1].add_(0.25)
            m.argument_polarity.bias[2].sub_(0.25)
        with torch.inference_mode():
            after=m(x,valid)
        for head in ("event_status","coreference","event_relation"):
            self.assertTrue(torch.equal(getattr(prior,head),getattr(after,head)),head)
        self.assertFalse(torch.equal(prior.role,after.role))
        self.assertFalse(torch.equal(prior.argument_polarity,after.argument_polarity))
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.candidate_accepted or m.activation_allowed)

    def test_no_gold_forward_or_spent_eval_in_s33_precision_objective(self):
        s=inspect.getsource(accepted_recipient_and_hard_false_pair_loss)
        self.assertIn("gold[",s)
        self.assertIn("topk(",s)
        forward=inspect.getsource(EventGroundedInversionGraph.forward)
        for forbidden in ("gold[","train_ids","held_ids","role_targets"):
            self.assertNotIn(forbidden,forward)


if __name__=="__main__":
    unittest.main()
