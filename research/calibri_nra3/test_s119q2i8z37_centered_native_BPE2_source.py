"""Z37: NEW independent author DEV14, correct candidate-centered RMS against Z35 absolute tanh collapse."""
import unittest
import torch
from collections import Counter
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS
from research.calibri_nra3.s119q2i8z37_new14dev_candidate_centered_rms_source import (
    rows,verify,digest,TOP_K,RANK,TRAIN_EPOCHS,ACTIVATE_SECOND_BPE_ONLY,
)
from research.calibri_nra3.s119q2i8z37_native_BPE2_centered_RMS_reranker import (
    NATIVE_VOCAB,HIDDEN_DIM,SHADOW_PARAMS,
    Position2NativeCandidateComparator,MODES,
)
class TestZ37CenteredRmsNextNativeBPE2Gate(unittest.TestCase):
    def test_01_new_synthetic_dev14_disjoint_all_previous_spent(self):
        p=verify()
        self.assertEqual((p["prior_legal_TRAIN84"],p["brandNEW_author_synthetic_DEV14"]),(84,14))
        self.assertTrue(p["new_DEV_disjoint_all_prior_spent_including_Z34"])
        self.assertTrue(p["no_previous_SPENT_DEV_in_TRAIN"])
        self.assertEqual(p["TRAIN_SHA256"],digest(rows("TRAIN")))
        self.assertEqual(p["NEW_DEV_SHA256"],digest(rows("DEV")))
        self.assertEqual(Counter(r["kind"] for r in rows("TRAIN")),{k:12 for k in KINDS})
        self.assertEqual(Counter(r["kind"] for r in rows("DEV")),{k:2 for k in KINDS})
        self.assertEqual((TOP_K,RANK,TRAIN_EPOCHS,ACTIVATE_SECOND_BPE_ONLY),(64,32,25,True))
        for wrong in ("GOLD","J7","EVAL128","TEST"):
            with self.assertRaises(ValueError):rows(wrong)
    def test_02_second_only_position_and_identity_at_init(self):
        model=Position2NativeCandidateComparator()
        self.assertEqual(sum(p.numel() for p in model.parameters()),SHADOW_PARAMS)
        logit=torch.full((1,NATIVE_VOCAB),-12.)
        logit[0,45]=5.;logit[0,46]=4.;logit[0,2]=3.
        h=torch.randn(1,HIDDEN_DIM)
        own=torch.randn(1,HIDDEN_DIM)
        emb=torch.randn(NATIVE_VOCAB,HIDDEN_DIM)
        ids,score,residual=model(logit,h,own,emb,native_bpe_position=2)
        self.assertTrue(torch.equal(score,logit[0,ids]))
        self.assertTrue(torch.equal(residual,torch.zeros_like(residual)))
        self.assertEqual(ids[:3].tolist(),[45,46,2])
        for other in (1,3,4,6):
            with self.assertRaises(ValueError):
                model(logit,h,own,emb,native_bpe_position=other)
        self.assertEqual(model.output_gate,0)
        self.assertFalse(model.candidate_accepted)
    def test_03_true_TRAIN_hardnegative_gradient_and_control(self):
        model=Position2NativeCandidateComparator()
        logit=torch.full((1,NATIVE_VOCAB),-9.)
        logit[0,37]=5;logit[0,38]=4
        h=torch.randn(1,HIDDEN_DIM)
        own=torch.randn(1,HIDDEN_DIM)
        emb=torch.randn(NATIVE_VOCAB,HIDDEN_DIM)
        good,meta=model.train_only_loss(logit,h,own,emb,
            gold_second=38,own_first_correct=True,
            split="TRAIN",wrong_control=False)
        bad,other=model.train_only_loss(logit,h,own,emb,
            gold_second=38,own_first_correct=True,
            split="TRAIN",wrong_control=True)
        self.assertTrue(meta["eligible"] and other["eligible"])
        self.assertNotEqual(float(good.detach()),float(bad.detach()))
        good.backward()
        self.assertIsNotNone(model.output_project.grad)
        self.assertGreater(float(model.output_project.grad.abs().sum()),0.)
        self.assertIsNone(logit.grad)
        self.assertIsNone(own.grad)
        self.assertIsNone(emb.grad)
        self.assertEqual(len(MODES),2)
    def test_04_candidate_centered_normalized_readout_prevents_absolute_tanh_saturation(self):
        torch.manual_seed(75)
        model=Position2NativeCandidateComparator()
        with torch.no_grad():
            # Extreme simulated pre-tanh common-mode formerly yielded
            # identical absolute ±1.5 output across all candidate IDs.
            model.output_project.fill_(35000.)
        logit=torch.full((1,NATIVE_VOCAB),-9.)
        logit[0,20]=5.
        h=torch.randn(1,HIDDEN_DIM)
        own=torch.randn(1,HIDDEN_DIM)
        emb=torch.randn(NATIVE_VOCAB,HIDDEN_DIM)
        ids,score,residual=model(logit,h,own,emb,native_bpe_position=2)
        self.assertTrue(torch.isfinite(residual).all())
        self.assertLessEqual(float(residual.abs().max()),1.500001)
        self.assertAlmostEqual(float(residual.mean()),0.,places=5)
        self.assertGreater(float(residual.max()-residual.min()),.20)
        self.assertGreater(float(residual.std()),.03)
        self.assertTrue((score-logit[0,ids]).abs().max()<=1.500001)
    def test_05_never_insert_missing_gold_or_train_DEV(self):
        model=Position2NativeCandidateComparator()
        logit=torch.full((1,NATIVE_VOCAB),-9.)
        logit[0,:64]=torch.arange(64.).float()
        h=torch.randn(1,HIDDEN_DIM)
        own=torch.randn(1,HIDDEN_DIM)
        emb=torch.randn(NATIVE_VOCAB,HIDDEN_DIM)
        loss,p=model.train_only_loss(logit,h,own,emb,
            gold_second=44000,own_first_correct=True,
            split="TRAIN",wrong_control=False)
        self.assertFalse(p["eligible"])
        self.assertEqual(float(loss.detach()),0.)
        loss,p=model.train_only_loss(logit,h,own,emb,
            gold_second=1,own_first_correct=False,
            split="TRAIN",wrong_control=False)
        self.assertFalse(p["eligible"])
        for split in ("DEV","J7","GOLD","EVAL128"):
            with self.assertRaises(ValueError):
                model.train_only_loss(logit,h,own,emb,
                    gold_second=1,own_first_correct=True,
                    split=split,wrong_control=False)
if __name__=="__main__":unittest.main()
