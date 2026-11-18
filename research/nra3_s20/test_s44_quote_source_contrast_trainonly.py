"""S44 frozen TRAIN-only event-source pairwise contrast tests.

These local CPU tests contain no original 43M embedding. CI will
physically train same-run S43 and S44 on S43 NEW128 TRAIN-only folds.
"""
from __future__ import annotations
import inspect
import unittest
import torch
from research.nra3_s20.s43_new128_multi_event_trainonly_source import (
    build_train128,validate_source,train_partitions,
)
from research.nra3_s20.s43_joint_learned_predicate_source_adapter import (
    JointLearnedPredicateSourceS43,
)
from research.nra3_s20.s44_train128_quote_calibrated_source_contrast_24fold import (
    directed_source_recipient_contrast,
    batch_raw_plus_train_labels,
    S44_EVENT_NEGATIVE_WEIGHT,
    S44_QUOTE_NEGATIVE_WEIGHT,
    S44_QUOTE_LOSS_COEFFICIENT,
    S44_PAIR_RANKING_COEFFICIENT,
    S44_TRAIN_LR,S44_TRAIN_STEPS,
)


class S44LossOnlyRUStructure(unittest.TestCase):
    def test_gold_only_training_rank_loss_handles_two_recipients_per_predicate(self):
        logits=torch.full((1,4,4),-4.0,requires_grad=True)
        gold=torch.zeros((1,4,4))
        gold[0,1,2]=1.0
        gold[0,1,3]=1.0
        mask=torch.ones((1,4,4),dtype=torch.bool)
        mask &= ~torch.eye(4,dtype=torch.bool).unsqueeze(0)
        loss=directed_source_recipient_contrast(logits,gold,mask)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        with torch.no_grad():
            improved=logits.detach().clone()
            improved[0,1,2]=3.
            improved[0,1,3]=3.
            loss_after=directed_source_recipient_contrast(improved,gold,mask)
        self.assertLess(float(loss_after),float(loss))
        self.assertEqual(int(gold.sum()),2)

    def test_new128_sealed_provenance_and_24_grouped_folds(self):
        src=validate_source()
        self.assertEqual(src["count"],128)
        self.assertEqual(src["TRAIN_two_rejected_edges_per_sentence"],32)
        self.assertTrue(src["S38_reserved_not_imported_or_used"])
        for axis,group,tr,held in train_partitions():
            self.assertTrue(set(tr).isdisjoint(held))
            self.assertEqual(len(tr),112 if axis=="LEXICAL" else 120)
            self.assertEqual(len(held),16 if axis=="LEXICAL" else 8)

    def test_new_quote_loss_and_learned_head_train_gradient_real(self):
        rows=build_train128()
        sample=[rows[j] for j in (6,7,10,11,15,22,23,27,37)]
        xp,xq,xy,xs,valid,legal,events,quotes,rejected=batch_raw_plus_train_labels(sample)
        self.assertGreater(int(quotes.sum()),0)
        self.assertGreater(int((quotes<.5).logical_and(valid).sum()),0)
        self.assertGreater(int(rejected.sum()),0)
        torch.manual_seed(441118)
        model=JointLearnedPredicateSourceS43()
        out=model(xp,xq,xy,xs)
        pos=(quotes>.5)&valid
        neg=(quotes<.5)&valid
        quote_loss=(torch.nn.functional.softplus(-out.quote_logits[pos]).mean()+
                    S44_QUOTE_NEGATIVE_WEIGHT*
                    torch.nn.functional.softplus(out.quote_logits[neg]).mean())
        pair_loss=directed_source_recipient_contrast(out.rejected_logits,rejected,legal)
        ev_pos=(events>.5)&valid
        full=quote_loss+pair_loss+torch.nn.functional.softplus(-out.event_logits[ev_pos]).mean()
        self.assertTrue(bool(torch.isfinite(full)))
        full.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in model.parameters()))
        self.assertEqual((model.output_gate,model.candidate_accepted,
                          model.activation_allowed),(0,False,False))

    def test_no_gold_used_in_neural_forward_and_no_prior_s38(self):
        import research.nra3_s20.s44_train128_quote_calibrated_source_contrast_24fold as stage
        code=inspect.getsource(stage)
        self.assertNotIn("from research.nra3_s20.s38_",code)
        self.assertNotIn("matching_links_from_s40(",code)
        self.assertNotIn("structurally_complete_REJECTED(",code)
        self.assertIn("out=model(x,q,p,structure)",code)
        self.assertNotIn("model(x,q,p,structure,ry)",code)
        self.assertEqual(S44_QUOTE_NEGATIVE_WEIGHT,2.)
        self.assertEqual(S44_QUOTE_LOSS_COEFFICIENT,.8)
        self.assertEqual(S44_EVENT_NEGATIVE_WEIGHT,1.)
        self.assertEqual(S44_PAIR_RANKING_COEFFICIENT,.6)
        self.assertEqual((S44_TRAIN_STEPS,S44_TRAIN_LR),(120,.012))


if __name__=="__main__":
    unittest.main()
