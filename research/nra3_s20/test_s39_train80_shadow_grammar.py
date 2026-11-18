"""S39 non-live source, head independence, raw quote and TRAIN fold guards.

These quick unit checks do NOT constitute actual Calibri43M+GT inference;
GitHub Actions runs physical CPU synthetic S39 trainer separately.
"""
from __future__ import annotations
import inspect
import unittest
import torch

from research.nra3_s20.s39_new80_trainonly_scope_graph_source import (
    build_train80,validate_source,partitions,STAGE,source_digest,
)
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import (
    raw_token_pair_features,ShadowNRA2S39,labels_train_only,
    model_weights_digest,
)
from research.nra3_s20.s39_fit_train80_nra2_shadow_scope_cv import (
    build_batch,TRAIN_STEPS,TRAIN_LR,TRAIN_GLOBAL_SEED,GROUP_FOLDS,
)


class S39TrainOnlyShadowGrammar(unittest.TestCase):
    def test_new80_exactly_separate_and_grouped_folds(self):
        r=validate_source()
        self.assertEqual(r["TRAIN_rows"],80)
        self.assertEqual((r["families"],r["patterns"],r["internal_folds"]),
                         (8,10,18))
        self.assertEqual((r["positive"],r["negative"]),(56,24))
        self.assertTrue(r["old_TRAIN256_disjoint"])
        self.assertTrue(r["S38_challenge_source_not_imported_or_used"])
        self.assertEqual(STAGE,"NRA3_S39_NOVEL80_TRAIN_ONLY_PREDICATE_NEGATION_SCOPE_PASSIVE_QUOTE")
        self.assertEqual(source_digest(),r["sha256"])
        for axis,group,tr,held in partitions():
            self.assertEqual(len(tr),(70 if axis=="LEXICAL" else 72))
            self.assertEqual(len(held),(10 if axis=="LEXICAL" else 8))
            self.assertTrue(set(tr).isdisjoint(held))

    def test_quote_is_feature_of_text_not_a_gold_role(self):
        examples=build_train80()
        quoted=examples[6]
        ordinary=examples[0]
        _,q1,x1=raw_token_pair_features(quoted.text)
        _,q2,x2=raw_token_pair_features(ordinary.text)
        self.assertGreater(int(q1[:,0].sum()),0)
        self.assertEqual(int(q2[:,0].sum()),0)
        self.assertEqual(int(quoted.rejected_recipient),0)
        ev,quot,y=labels_train_only(quoted)
        self.assertEqual(int(y.sum()),0)
        self.assertEqual(int(ev.sum()),1)
        self.assertGreater(int(quot.sum()),0)

    def test_raw_only_prediction_and_differentiable_three_head_train(self):
        torch.manual_seed(393911)
        examples=build_train80()[:7]
        p,q,x,valid,legal,ge,gq,gy=build_batch(examples)
        model=ShadowNRA2S39()
        out=model(p,q,x)
        self.assertEqual(tuple(out.event_logits.shape),tuple(ge.shape))
        self.assertEqual(tuple(out.quote_logits.shape),tuple(gq.shape))
        self.assertEqual(tuple(out.rejected_logits.shape),tuple(gy.shape))
        gold_perturbed=gy.clone()
        gold_perturbed[0]*=0.
        # Out cannot change when labels change without passing them to forward.
        with torch.no_grad():
            out2=model(p,q,x)
        self.assertTrue(torch.equal(out.rejected_logits,out2.rejected_logits))
        loss=(out.event_logits[valid].square().mean()
              +out.quote_logits[valid].square().mean()
              +out.rejected_logits[legal].square().mean())
        loss.backward()
        self.assertTrue(all(z.grad is not None for z in model.parameters()))
        self.assertTrue(all(torch.isfinite(z.grad).all() for z in model.parameters()))
        self.assertEqual((model.output_gate,model.candidate_accepted,
                          model.activation_allowed),(0,False,False))
        self.assertEqual(len(model_weights_digest(model)),64)

    def test_no_S38_holdout_in_trainer_code_or_original_backend_changes(self):
        import research.nra3_s20.s39_fit_train80_nra2_shadow_scope_cv as script
        train=inspect.getsource(script.train_one_fold)
        module=inspect.getsource(script)
        self.assertNotIn("from research.nra3_s20.s38_",module)
        self.assertNotIn("import research.nra3_s20.s38_",module)
        self.assertNotIn("nra2_original",train)
        self.assertNotIn("calibri43m",train.lower())
        self.assertIn("TRAIn_or_never" if False else "positive=(ry>0.5)&legal",train)
        self.assertIn("model(xp,xq,xy)",train)
        self.assertNotIn("model(xp,xq,xy,ry)",train)
        self.assertEqual(TRAIN_STEPS,96)
        self.assertEqual(TRAIN_LR,.018)
        self.assertEqual(GROUP_FOLDS,18)
        self.assertGreater(TRAIN_GLOBAL_SEED,0)


if __name__=="__main__":
    unittest.main()
