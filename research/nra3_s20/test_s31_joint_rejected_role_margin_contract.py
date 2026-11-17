"""S31 source-only TRAIN loss unit checks (never claimed as actual 43M CPU)."""
from __future__ import annotations
import inspect
import unittest

import torch

from research.nra3_s19.s19f_real43m_train9_graph_targets import ROLE
from research.nra3_s19.s19g_paid_semantic_graph import GraphLogits
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    EventGroundedInversionGraph,PREV_NEG,LIKELY_DATIVE,ROLE_RECIPIENT,
)
from research.nra3_s20.s31_train256_rare_pair_margin_probe import (
    pairwise_hard_negative_positive_margin_loss,
    MARGIN,HARD_NEGATIVE_MARGIN,TRAIN_POSITIVE_COEFF,
    TRAIN_HARD_NEGATIVE_COEFF,FOLDS,PREDECLARED_STEPS,PAID_TOTAL,
)
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import splits


def fake_train_loss_fixture(role_recipient_boost=0.,polarity_rejected_boost=0.,
                            hard_negative_recipient_boost=0.,hard_negative_rejected_boost=0.):
    """TRAIN-only synthetic role arrays, not original frozen43M evidence."""
    b,w=1,4
    valid=torch.ones(b,w,dtype=torch.bool)
    x=torch.zeros((b,w,338),dtype=torch.float32)
    # True recipient j0, distractor j2, both raw negative+dative.
    for j in (0,2):
        x[0,j,320+PREV_NEG]=1.
        x[0,j,320+LIKELY_DATIVE]=1.
    gold={
        "event_status_TRAIN_ONLY":torch.tensor([[0,1,0,0]]),
        "predicate_argument_role_TRAIN_ONLY":torch.zeros((b,w,w),dtype=torch.long),
        "argument_polarity_TRAIN_ONLY":torch.full((b,w,w),-1,dtype=torch.long),
    }
    gold["predicate_argument_role_TRAIN_ONLY"][0,1,0]=ROLE_RECIPIENT
    gold["argument_polarity_TRAIN_ONLY"][0,1,0]=1
    roles=torch.zeros((b,w,w,len(ROLE)),dtype=torch.float32)
    polarity=torch.zeros((b,w,w,3),dtype=torch.float32)
    roles[0,1,0,ROLE_RECIPIENT]=role_recipient_boost
    polarity[0,1,0,2]=polarity_rejected_boost
    roles[0,1,2,ROLE_RECIPIENT]=hard_negative_recipient_boost
    polarity[0,1,2,2]=hard_negative_rejected_boost
    roles.requires_grad_(True)
    polarity.requires_grad_(True)
    out=GraphLogits(
      torch.zeros((b,w,6)),roles,polarity,
      torch.zeros((b,w,w,2)),torch.zeros((b,w,w,5)),
    )
    return out,x,valid,gold


class S31Frozen43mResearchSourceContracts(unittest.TestCase):
    def test_rare_training_polarity_scope_and_hard_negative_are_separate(self):
        out,x,valid,gold=fake_train_loss_fixture()
        pos,neg,details=pairwise_hard_negative_positive_margin_loss(
            out,x,valid,gold)
        self.assertEqual(details["TRAIN_only_true_rejected_recipient"],1)
        self.assertEqual(details["TRAIN_only_hard_confusable_nonrecipient"],1)
        self.assertGreater(float(pos),0.)
        self.assertGreater(float(neg),0.)
        (pos+neg).backward()
        self.assertIsNotNone(out.role.grad)
        self.assertIsNotNone(out.argument_polarity.grad)
        self.assertNotEqual(float(out.role.grad[0,1,0,ROLE_RECIPIENT]),0.)
        self.assertNotEqual(float(out.argument_polarity.grad[0,1,0,2]),0.)

    def test_joint_positive_margin_grows_with_role_and_pol_logits(self):
        base=fake_train_loss_fixture()
        good=fake_train_loss_fixture(role_recipient_boost=4.,
                                     polarity_rejected_boost=4.)
        p0,n0,_=pairwise_hard_negative_positive_margin_loss(*base)
        p1,n1,_=pairwise_hard_negative_positive_margin_loss(*good)
        self.assertLess(float(p1),float(p0))
        self.assertAlmostEqual(float(n0),float(n1),places=5)

    def test_hard_negative_penalty_rejects_spurious_other_candidate_pair(self):
        base=fake_train_loss_fixture()
        wrong=fake_train_loss_fixture(hard_negative_recipient_boost=5.,
                                      hard_negative_rejected_boost=5.)
        p0,n0,_=pairwise_hard_negative_positive_margin_loss(*base)
        p1,n1,_=pairwise_hard_negative_positive_margin_loss(*wrong)
        self.assertAlmostEqual(float(p0),float(p1),places=5)
        self.assertGreater(float(n1),float(n0))

    def test_predeclared_small_eight_construction_fold_geometry_and_no_model_gold(self):
        self.assertEqual(FOLDS,(32,33,34,35,36,15,24,30))
        self.assertEqual(PREDECLARED_STEPS,240)
        self.assertEqual(PAID_TOTAL,12384)
        self.assertEqual((MARGIN,HARD_NEGATIVE_MARGIN), (3.0,0.5))
        self.assertEqual((TRAIN_POSITIVE_COEFF,TRAIN_HARD_NEGATIVE_COEFF),(.7,.1))
        for fold in FOLDS:
            axis,group,train,held=splits()[fold]
            self.assertEqual(axis,"CONSTRUCTION")
            self.assertEqual(group,fold-8)
            self.assertEqual((len(train),len(held)),(248,8))
            self.assertTrue(set(train).isdisjoint(held))
        source=inspect.getsource(EventGroundedInversionGraph.forward)
        for forbidden in ("gold","oracle","label_targets","held_group"):
            self.assertNotIn(forbidden,source)
        train_source=inspect.getsource(pairwise_hard_negative_positive_margin_loss)
        self.assertIn("gold_role",train_source)
        self.assertIn("gold_event",train_source)
        self.assertIn("raw_cue",train_source)


if __name__=="__main__":
    unittest.main()
