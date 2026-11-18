"""S40 research-only compositional structural graph, not Calibri/NRA2 LIVE."""
from __future__ import annotations
import inspect
import unittest
import torch

from research.nra3_s20.s39_new80_trainonly_scope_graph_source import (
    build_train80,validate_source,partitions,
)
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import (
    raw_token_pair_features,labels_train_only,
)
from research.nra3_s20.s40_compositional_predicate_negation_graph import (
    CompositionalNRA2S40,plain_scope_features,STRUCTURE_DIM,
    PREDECLARED_BIAS_ALLOWED,PREDECLARED_BIAS_BLOCKED,
)
from research.nra3_s20.s40_fit_train80_compositional_graph_18fold import (
    build_batch,TRAIN_STEPS,TRAIN_LR,
)

class S40CompositionalGraphContracts(unittest.TestCase):
    def test_unchanged_s39_train80_18fold_data_and_no_s38_reuse(self):
        src=validate_source()
        self.assertEqual(src["TRAIN_rows"],80)
        self.assertEqual((src["positive"],src["negative"]),(56,24))
        self.assertTrue(src["S38_challenge_source_not_imported_or_used"])
        self.assertEqual(len(partitions()),18)

    def test_fixed_compositional_graph_detects_source_and_quoted_negative_mentions(self):
        rows=build_train80()
        supported=0
        rejected_false=0
        for row in rows:
            graph=plain_scope_features(row.text)
            n=len(raw_token_pair_features(row.text)[0])
            self.assertEqual(tuple(graph.shape),(n,n,STRUCTURE_DIM))
            self.assertEqual(int(graph.diagonal(dim1=0,dim2=1)[11].sum()),0)
            idx=(row.source_predicate_token_index0,
                 row.potential_recipient_token_index0)
            eligibility=int(graph[idx[0],idx[1],11].item())
            if row.rejected_recipient:
                supported+=eligibility
            else:
                rejected_false+=eligibility
        self.assertEqual(supported,56,
                         "S40 fixed TRAIN-side surface compositional support drift")
        self.assertEqual(rejected_false,0,
                         "S40 quote or event-negation versus rejected participant conflated")
        self.assertEqual((PREDECLARED_BIAS_ALLOWED,PREDECLARED_BIAS_BLOCKED),
                         (4.,-12.))

    def test_gold_blind_forward_still_has_learnable_event_quote_scope_heads(self):
        rows=build_train80()[:10]
        x,q,p,s,valid,legal,ey,qy,ry=build_batch(rows)
        model=CompositionalNRA2S40()
        outputs=model(x,q,p,s)
        self.assertEqual(outputs.rejected_logits.shape,ry.shape)
        self.assertEqual(outputs.event_logits.shape,ey.shape)
        self.assertEqual(outputs.quote_logits.shape,qy.shape)
        loss=outputs.rejected_logits[legal].square().mean()+(
            outputs.event_logits[valid].square().mean()
            +outputs.quote_logits[valid].square().mean())
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertTrue(model.output_gate==0
                        and not model.candidate_accepted
                        and not model.activation_allowed)

    def test_no_gold_or_s38_challenge_in_model_graph_and_training(self):
        import research.nra3_s20.s40_compositional_predicate_negation_graph as graph
        import research.nra3_s20.s40_fit_train80_compositional_graph_18fold as trainer
        s=inspect.getsource(graph.plain_scope_features)
        forward=inspect.getsource(graph.CompositionalNRA2S40.forward)
        for forbidden in ("gold[","source_predicate_token_index0",
                          "rejected_recipient","row.family","PATTERNS[",
                          "ROWS","s38_"):
            self.assertNotIn(forbidden,s)
            self.assertNotIn(forbidden,forward)
        module=inspect.getsource(trainer)
        self.assertNotIn("from research.nra3_s20.s38_",module)
        self.assertIn("model(xp,xq,xy,xstruct)",module)
        self.assertNotIn("model(xp,xq,xy,xstruct,ry)",module)
        self.assertEqual((TRAIN_STEPS,TRAIN_LR),(96,.018))

if __name__=="__main__":
    unittest.main()
