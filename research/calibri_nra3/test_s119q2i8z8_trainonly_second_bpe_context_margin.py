"""Pure CPU unit tests for source-only S119q2i8z8 objective pre-registration.

These tests train NOTHING, read NO artifacts/data, use NO sealed scores and
do NOT establish any model improvement or safe release readiness.
"""
import unittest
import torch
from torch.nn import functional as F

from research.calibri_nra3.s119q2i8z8_trainonly_second_bpe_context_margin import (
    NATIVE_VOCAB, EOS_NATIVE, COMPETITOR_TOPK,
    MARGIN_LOSS_WEIGHT, SECOND_NATIVE_INDEX,
    trainonly_second_native_margin, training_objective,
)


class SecondNativeContextMarginTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.zeros((1, NATIVE_VOCAB), dtype=torch.float32, requires_grad=True)

    def margin(self, logits=None, target=17, idx=1, split="TRAIN", previous=True):
        return trainonly_second_native_margin(
            self.x if logits is None else logits, target,
            bpe_index=idx, source_split=split, human_previous_gold=previous,
        )

    def test_exact_native_vocab_and_prespecified_hyperparameters(self):
        self.assertEqual(NATIVE_VOCAB, 49152)
        self.assertEqual(EOS_NATIVE, 2)
        self.assertEqual(SECOND_NATIVE_INDEX, 1)
        self.assertEqual(COMPETITOR_TOPK, 8)
        self.assertEqual(MARGIN_LOSS_WEIGHT, 0.20)

    def test_trainonly_second_non_eos_has_real_loss_and_gradient(self):
        with torch.no_grad():
            self.x[0, 2345] = 5.0
        z = self.margin()
        self.assertGreater(float(z.detach()), 0.1)
        z.backward()
        self.assertLess(float(self.x.grad[0, 17]), 0.0)
        self.assertGreater(float(self.x.grad[0, 2345]), 0.0)
        self.assertEqual(int(torch.count_nonzero(self.x.grad).item()), 9)

    def test_context_gold_beats_wrong_reduces_second_margin(self):
        with torch.no_grad():
            self.x[0, 2345] = 5.0
        bad = float(self.margin().detach())
        with torch.no_grad():
            self.x[0, 17] = 10.0
        good = float(self.margin().detach())
        self.assertLess(good, bad)

    def test_distinct_topk_behavior_not_earlier_strongest_single_wrong(self):
        a = torch.zeros((1, NATIVE_VOCAB))
        b = a.clone()
        a[0, 999] = b[0, 999] = 5.0
        b[0, 998] = 4.9
        # Same largest wrong ID/logit, but another plausible wrong ID
        # has real contextual influence in a smooth top-8 objective.
        self.assertGreater(float(self.margin(b)), float(self.margin(a)))

    def test_wrong_eos_is_included_in_trainonly_second_competitors(self):
        a = torch.zeros((1, NATIVE_VOCAB))
        b = a.clone()
        b[0, 2] = 12.0
        self.assertGreater(float(self.margin(b)), float(self.margin(a)) + 0.5)
        self.assertEqual(b[0, 2].item(), 12.0)

    def test_dev_eval_gold_and_own_previous_are_rejected(self):
        for split in ("DEV", "EVAL", "EVAL128", "J7", "GOLD", "test"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.margin(split=split)
        with self.assertRaises(ValueError):
            self.margin(previous=False)

    def test_first_third_and_terminal_eos_have_zero_additional_loss(self):
        for idx, target in ((0, 17), (2, 17), (3, 2), (39, 2)):
            with self.subTest(idx=idx):
                z = self.margin(idx=idx, target=target)
                self.assertEqual(z.item(), 0.0)
        with self.assertRaises(ValueError):
            self.margin(idx=1, target=2)

    def test_full_ce_unchanged_other_positions(self):
        for idx, target, weight in ((0, 17, 1.0), (2, 17, 3.0), (3, 2, 1.0)):
            with self.subTest(idx=idx):
                result = training_objective(
                    self.x, target, bpe_index=idx, source_split="TRAIN",
                    human_previous_gold=True, native_ce_position_weight=weight)
                baseline = weight * F.cross_entropy(self.x, torch.tensor([target]))
                self.assertTrue(torch.allclose(result, baseline, atol=1e-6))
        result2 = training_objective(
            self.x, 17, bpe_index=1, source_split="TRAIN",
            human_previous_gold=True, native_ce_position_weight=3.0)
        base2 = 3.0 * F.cross_entropy(self.x, torch.tensor([17]))
        self.assertGreater(float(result2.detach()), float(base2.detach()))

    def test_ce_weights_and_token_contract(self):
        with self.assertRaises(ValueError):
            training_objective(
                self.x, 17, bpe_index=1, source_split="TRAIN",
                human_previous_gold=True, native_ce_position_weight=1.0)
        with self.assertRaises(ValueError):
            self.margin(target=-1)
        with self.assertRaises(ValueError):
            self.margin(target=NATIVE_VOCAB)
        with self.assertRaises(ValueError):
            self.margin(idx=-1)
        with self.assertRaises(ValueError):
            self.margin(logits=torch.zeros((1, 100)))
        with self.assertRaises(ValueError):
            self.margin(logits=torch.full((1, NATIVE_VOCAB), float("nan")))

    def test_input_logits_not_modified_no_inference_logits_logic(self):
        with torch.no_grad():
            self.x[0, EOS_NATIVE] = 11.0
        before = self.x.detach().clone()
        _ = self.margin()
        self.assertTrue(torch.equal(self.x.detach(), before))


if __name__ == "__main__":
    unittest.main()
