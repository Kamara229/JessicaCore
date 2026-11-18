"""S32 source-only tests: precision contrast is TRAIN-only, NEVER evaluation.

Fake logits here test math and guards only; genuine original43M+GT physical
outcome is established separately by GitHub Actions S32 run.
"""
from __future__ import annotations

import inspect
import unittest

import torch

from research.nra3_s20.s31_train256_rare_pair_margin_probe import (
    pairwise_hard_negative_positive_margin_loss,
)
from research.nra3_s20.test_s31_joint_rejected_role_margin_contract import (
    fake_train_loss_fixture,
)
from research.nra3_s20.s32_train256_precision_rejection_calibration import (
    accepted_recipient_and_hard_false_pair_loss,
    TRAIN_ACCEPTED_RECIPIENT_COEFF,TRAIN_FALSE_PAIR_TOPK_COEFF,
    NOT_REJECTED_MARGIN,FALSE_PAIR_MARGIN,
    FALSE_PAIR_TOPK_PER_TEXT,FOLDS,PREDECLARED_STEPS,
    PAID_TOTAL,
)
from research.nra3_s20.s26_predicted_event_inversion_rescue import ROLE_RECIPIENT


def controlled_train_fixture(accepted_rejected_logit=0.,false_rejected_logit=0.,
                             true_rejected_logit=0.):
    # A test-only fake model output with an actual rejected recipient (1,0),
    # actual accepted recipient (1,3), and fake non-recipient (1,2).
    out,raw,valid,gold=fake_train_loss_fixture(
        polarity_rejected_boost=true_rejected_logit)
    gold["predicate_argument_role_TRAIN_ONLY"][0,1,3]=ROLE_RECIPIENT
    gold["argument_polarity_TRAIN_ONLY"][0,1,3]=0
    with torch.no_grad():
        out.argument_polarity[0,1,3,2]=accepted_rejected_logit
        out.argument_polarity[0,1,2,2]=false_rejected_logit
    return out,raw,valid,gold


class S32Original43mTrainOnlyPrecisionCalibGuards(unittest.TestCase):
    def test_hard_false_pair_and_accepted_recipient_supervision_both_live(self):
        out,raw,valid,gold=controlled_train_fixture()
        accepted,hard,counts=accepted_recipient_and_hard_false_pair_loss(
            out,valid,gold)
        self.assertEqual(counts["TRAIN_accepted_recipient_pairs"],1)
        self.assertGreater(counts["TRAIN_other_nonrecipient_pairs"],0)
        self.assertEqual(counts["TRAIN_per_sentence_top_false_pairs"],2)
        self.assertGreater(float(accepted),0.)
        self.assertGreater(float(hard),0.)
        (accepted+hard).backward()
        self.assertNotEqual(float(out.argument_polarity.grad[0,1,3,2]),0.)
        self.assertTrue(torch.isfinite(out.argument_polarity.grad).all())

    def test_penalizes_wrong_rejection_for_accepted_recipients(self):
        low=controlled_train_fixture(accepted_rejected_logit=-5.)
        high=controlled_train_fixture(accepted_rejected_logit=5.)
        good_loss,_,_=accepted_recipient_and_hard_false_pair_loss(
            low[0],low[2],low[3])
        bad_loss,_,_=accepted_recipient_and_hard_false_pair_loss(
            high[0],high[2],high[3])
        self.assertLess(float(good_loss),float(bad_loss))

    def test_penalizes_top_spurious_wrong_rejections_without_using_held(self):
        normal=controlled_train_fixture(false_rejected_logit=-4.)
        spurious=controlled_train_fixture(false_rejected_logit=12.)
        pos1,hard1,_=accepted_recipient_and_hard_false_pair_loss(
            normal[0],normal[2],normal[3])
        pos2,hard2,_=accepted_recipient_and_hard_false_pair_loss(
            spurious[0],spurious[2],spurious[3])
        self.assertAlmostEqual(float(pos1),float(pos2),places=5)
        self.assertGreater(float(hard2),float(hard1))

    def test_positive_rejected_recipient_still_uses_original_s31_training_margin(self):
        out,raw,valid,gold=controlled_train_fixture()
        positive,hard,stats=pairwise_hard_negative_positive_margin_loss(
            out,raw,valid,gold)
        self.assertEqual(stats["TRAIN_only_true_rejected_recipient"],1)
        self.assertGreater(float(positive),0.)

    def test_precommitted_fold_schedule_paid_parameter_and_no_gold_forward(self):
        self.assertEqual(FOLDS,(32,33,34,35,36,15,24,30))
        self.assertEqual(PREDECLARED_STEPS,240)
        self.assertEqual(PAID_TOTAL,12384)
        self.assertEqual((TRAIN_ACCEPTED_RECIPIENT_COEFF,
                          TRAIN_FALSE_PAIR_TOPK_COEFF),(.45,.05))
        self.assertEqual((NOT_REJECTED_MARGIN,
                          FALSE_PAIR_MARGIN,FALSE_PAIR_TOPK_PER_TEXT),(2.,.5,2))
        code=inspect.getsource(accepted_recipient_and_hard_false_pair_loss)
        self.assertIn("accepted_recipient",code)
        self.assertIn("topk",code)
        self.assertIn("polarity_gold",code)
        import research.nra3_s20.s26_predicted_event_inversion_rescue as mod
        forward=inspect.getsource(mod.EventGroundedInversionGraph.forward)
        self.assertNotIn("gold[",forward)
        self.assertNotIn("held_group",forward)
        self.assertNotIn("train_ids",forward)


if __name__=="__main__":
    unittest.main()
