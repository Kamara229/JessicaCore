"""S34 gold-blind-forward, TRAIN-only anti-collapse and rare pair ranking unit checks."""
from __future__ import annotations
import inspect
import unittest
import torch
from research.nra3_s20.test_s32_precision_margin_trainonly import controlled_train_fixture
from research.nra3_s20.s34_trainonly_teacher_rejected_anti_collapse import (
    train_only_rejected_anti_collapse,
    S34_REJECTED_ANCHOR_COEFF,
    S34_REJECTED_TEACHER_FLOOR_DELTA,
    S34_TRUE_VS_FALSE_PAIR_RANK_COEFF,
    S34_TRAIN_RANK_MARGIN,
    FOLDS,
    PAID_TOTAL,
    S34_ONLY_HEAD_STEPS,
    S34_HEAD_FINETUNE_LR,
    fixed_train_S34,
)
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph


class S34TrainOnlyRarePositiveAnchorContract(unittest.TestCase):
    def get_case(self, true_rejected_logit=0., false_rejected_logit=0.):
        return controlled_train_fixture(
            true_rejected_logit=true_rejected_logit,
            false_rejected_logit=false_rejected_logit,
        )

    def test_fall_of_gold_positive_margin_strictly_penalized(self):
        base=self.get_case()
        teacher=self.get_case(true_rejected_logit=4.)
        strong=self.get_case(true_rejected_logit=5.)
        poor=self.get_case(true_rejected_logit=-3.)
        pos_high,rank_high,ct=train_only_rejected_anti_collapse(
            strong[0],teacher[0],strong[2],strong[3])
        pos_low,rank_low,_=train_only_rejected_anti_collapse(
            poor[0],teacher[0],poor[2],poor[3])
        self.assertLess(float(pos_high),float(pos_low))
        self.assertLess(float(rank_high),float(rank_low))
        self.assertEqual(ct["TRAIN_only_supervised_rejected_pairs"],1)
        self.assertEqual(ct["TRAIN_only_ranked_texts"],1)

    def test_high_scored_false_pair_worsens_rank_but_not_teacher_floor(self):
        teacher=self.get_case(true_rejected_logit=2.)
        normal=self.get_case(true_rejected_logit=3.,false_rejected_logit=-4.)
        spurious=self.get_case(true_rejected_logit=3.,false_rejected_logit=12.)
        anchor1,r1,_=train_only_rejected_anti_collapse(
            normal[0],teacher[0],normal[2],normal[3])
        anchor2,r2,_=train_only_rejected_anti_collapse(
            spurious[0],teacher[0],spurious[2],spurious[3])
        self.assertAlmostEqual(float(anchor1),float(anchor2),places=6)
        self.assertGreater(float(r2),float(r1))

    def test_gradient_flow_through_student_logits_not_teacher(self):
        teacher=self.get_case(true_rejected_logit=4.)
        student=self.get_case(true_rejected_logit=-3.)
        anchor,rank,_=train_only_rejected_anti_collapse(
            student[0],teacher[0],student[2],student[3])
        (anchor+rank).backward()
        self.assertIsNotNone(student[0].argument_polarity.grad)
        self.assertLess(float(student[0].argument_polarity.grad[0,1,0,2]),0.)
        self.assertIsNone(teacher[0].argument_polarity.grad)

    def test_precommitted_eight_folds_frozen_semantics_and_12384_paid_fp16(self):
        self.assertEqual(FOLDS,(32,33,34,35,36,15,24,30))
        self.assertEqual(PAID_TOTAL,12384)
        self.assertEqual((S34_ONLY_HEAD_STEPS,S34_HEAD_FINETUNE_LR),(64,.003))
        self.assertEqual((S34_REJECTED_ANCHOR_COEFF,
                          S34_REJECTED_TEACHER_FLOOR_DELTA),(2.,.5))
        self.assertEqual((S34_TRUE_VS_FALSE_PAIR_RANK_COEFF,
                          S34_TRAIN_RANK_MARGIN),(.5,1.))
        s=inspect.getsource(fixed_train_S34)
        self.assertIn("teacher_logits=preserved_s31(raw,valid)",s)
        self.assertIn("train_only_rejected_anti_collapse",s)
        self.assertIn("name in allowed",s)
        self.assertIn("torch.equal(p.detach(),initial_weights[name])",s)
        self.assertNotIn("eval128",s.lower())
        f=inspect.getsource(EventGroundedInversionGraph.forward)
        self.assertNotIn("gold[",f)
        self.assertNotIn("held_ids",f)


if __name__=="__main__":
    unittest.main()
