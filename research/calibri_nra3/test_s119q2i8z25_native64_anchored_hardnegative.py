"""Z25 prereg 84 legal TRAIN +14 truly new DEV and candidate-specific rank16."""
import unittest
from collections import Counter
import torch
from research.calibri_nra3.s119q2i8z25_84historicaltrain_new14dev_hardnegative_source import (
  rows,verify,digest,KINDS,TOP_K,RANK,RESIDUAL_BOUND,TRAIN_EPOCHS
)
from research.calibri_nra3.s119q2i8z25_native64_anchored_hardnegative_rank16 import (
  VOCAB,HIDDEN,MODES,Native64AnchoredHardNegative,train_matched_control_on_historical_train
)

class TestZ25HardNegativePrereg(unittest.TestCase):
    def test_01_legal_training_only_previous_train_no_spent_dev(self):
        x=verify()
        self.assertEqual((x["historical_previously_consumed_TRAIN_count"],
                          x["brand_new_author_synthetic_DEV_count"]),(84,14))
        self.assertEqual(x["TRAIN_SHA256"],digest(rows("TRAIN")))
        self.assertEqual(x["DEV_SHA256"],digest(rows("DEV")))
        self.assertTrue(x["no_SPENT_DEV_reused_as_training"])
        self.assertTrue(x["new_DEV_disjoint_all_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16_Z18_Z21_Z23"])
        self.assertEqual(Counter(x["kind"] for x in rows("TRAIN")),
                         {k:12 for k in KINDS})
        self.assertEqual(Counter(x["kind"] for x in rows("DEV")),
                         {k:2 for k in KINDS})
        self.assertEqual((TOP_K,RANK,RESIDUAL_BOUND,TRAIN_EPOCHS),(64,16,3.0,30))
        for wrong in ("GOLD","J7","EVAL128","TEST","TRAIN+DEV"):
            with self.assertRaises(ValueError):rows(wrong)

    def test_02_baseline_identity_zero_residual_initial_state(self):
        p=Native64AnchoredHardNegative()
        self.assertEqual(sum(x.numel() for x in p.parameters()),10240)
        logits=torch.full((1,VOCAB),-11.)
        logits[0,33]=5.;logits[0,34]=4.;logits[0,2]=3.
        emb=torch.randn((VOCAB,HIDDEN),generator=torch.Generator().manual_seed(7))
        h=torch.randn((1,HIDDEN),generator=torch.Generator().manual_seed(18))
        ids,scores,residual=p(logits,h,emb)
        self.assertEqual(ids[:3].tolist(),[33,34,2])
        self.assertTrue(torch.equal(scores,logits[0,ids]))
        self.assertTrue(torch.equal(residual,torch.zeros((TOP_K,))))
        self.assertIn(2,ids.tolist())
        self.assertEqual((p.output_gate,p.candidate_accepted),(0,False))

    def test_03_hardnegative_correct_gold_and_original_tensor_frozen(self):
        p=Native64AnchoredHardNegative()
        logits=torch.full((1,VOCAB),-9.,requires_grad=True)
        with torch.no_grad():
            logits[0,67]=5.;logits[0,68]=4.
        h=torch.randn((1,HIDDEN),requires_grad=True)
        emb=torch.randn((VOCAB,HIDDEN),requires_grad=True)
        with self.assertRaises(ValueError):
            p(logits,h,emb)
        emb.requires_grad_(False)
        loss,diag=p.train_only_loss(
            logits,h,emb,68,split="TRAIN",own_first_correct=True,control=False)
        self.assertTrue(diag["eligible"])
        self.assertFalse(diag["original_top1_correct"])
        loss.backward()
        self.assertIsNotNone(p.right.grad)
        self.assertGreater(float(p.right.grad.abs().sum()),0.)
        self.assertIsNone(logits.grad)
        self.assertIsNone(h.grad)
        self.assertIsNone(emb.grad)

    def test_04_target_not_in_top64_never_inserted_no_DEV_train(self):
        p=Native64AnchoredHardNegative()
        logits=torch.full((1,VOCAB),-15.)
        logits[0,:64]=torch.arange(64.)
        h=torch.randn((1,HIDDEN))
        emb=torch.randn((VOCAB,HIDDEN))
        loss,meta=p.train_only_loss(
            logits,h,emb,40000,split="TRAIN",own_first_correct=True,control=False)
        self.assertEqual(float(loss.detach()),0.)
        self.assertFalse(meta["eligible"])
        for split in ("DEV","J7","EVAL128","GOLD"):
            with self.assertRaises(ValueError):
                p.train_only_loss(
                    logits,h,emb,60,split=split,own_first_correct=True)
        loss,meta=p.train_only_loss(
            logits,h,emb,60,split="TRAIN",own_first_correct=False)
        self.assertFalse(meta["eligible"])

    def test_05_wrong_candidate_control_is_never_human_gold(self):
        p=Native64AnchoredHardNegative()
        logits=torch.full((1,VOCAB),-7.)
        logits[0,99]=10;logits[0,100]=9
        h=torch.randn((1,HIDDEN))
        emb=torch.randn((VOCAB,HIDDEN))
        good,mg=p.train_only_loss(
            logits,h,emb,100,split="TRAIN",
            own_first_correct=True,control=False)
        ctrl,mc=p.train_only_loss(
            logits,h,emb,100,split="TRAIN",
            own_first_correct=True,control=True)
        self.assertTrue(mg["eligible"] and mc["eligible"])
        self.assertTrue(mc["train_negative_control_target_shifted"])
        self.assertNotEqual(float(good.detach()),float(ctrl.detach()))
        self.assertEqual(len(MODES),2)

if __name__=="__main__":
    unittest.main()
