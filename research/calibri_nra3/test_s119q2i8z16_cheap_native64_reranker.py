"""Z16: SHA-locked 28TRAIN14DEV and lightweight original43M readout rank2 tests."""
import unittest
from collections import Counter
import torch
from torch import nn

from research.calibri_nra3.s119q2i8z16_fresh28train14dev_rerank_source import (
    verify,rows,digest,KINDS,CANDIDATE_K,
)
from research.calibri_nra3.s119q2i8z16_trainonly_lowrank_native64_reranker import (
    VOCAB,HIDDEN,RANK,K,RESIDUAL_GAIN,
    NativeBPE2CandidateReranker,capture_existing_full43m_hidden,
)

class FakeOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head=nn.Linear(HIDDEN,12,bias=False)

class TestZ16PrereggedReranker(unittest.TestCase):
    def test_01_new_human_ru_source_SHA_and_no_spent(self):
        m=verify()
        self.assertEqual((m["TRAIN_n"],m["DEV_n"]),(28,14))
        self.assertEqual(Counter(x["kind"] for x in rows("TRAIN")),
                         {k:4 for k in KINDS})
        self.assertEqual(Counter(x["kind"] for x in rows("DEV")),
                         {k:2 for k in KINDS})
        self.assertEqual(m["TRAIN_sha256"],digest(rows("TRAIN")))
        self.assertEqual(m["DEV_sha256"],digest(rows("DEV")))
        self.assertTrue(m["disjoint_all_prior_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14"])
        self.assertEqual(CANDIDATE_K,64)
        for bad in ("GOLD","EVAL128","J7","TEST","TRAIN+DEV"):
            with self.assertRaises(ValueError):rows(bad)
    def test_02_exact_1280_params_and_baseline_initial_native_top64(self):
        r=NativeBPE2CandidateReranker()
        self.assertEqual(sum(x.numel() for x in r.parameters()),1280)
        self.assertEqual((VOCAB,HIDDEN,RANK,K,RESIDUAL_GAIN),(49152,320,2,64,.25))
        self.assertEqual((r.output_gate,r.candidate_accepted),(0,False))
        h=torch.arange(HIDDEN,dtype=torch.float32).view(1,HIDDEN)
        logits=torch.zeros((1,VOCAB))
        logits[0,20]=12;logits[0,21]=10;logits[0,2]=9
        emb=torch.zeros((VOCAB,HIDDEN))
        ids,scores=r(logits,h,emb)
        self.assertEqual(ids[0].item(),20)
        self.assertEqual(ids[1].item(),21)
        self.assertTrue((ids==2).any())
        self.assertTrue(torch.equal(scores,logits[0,ids]))
    def test_03_only_TRAIN_gold_in_existing_top64_is_differentiable(self):
        g=torch.Generator().manual_seed(18)
        r=NativeBPE2CandidateReranker()
        logits=torch.zeros((1,VOCAB),requires_grad=True)
        with torch.no_grad():
            logits[0,100]=5;logits[0,101]=4
        h=torch.randn((1,HIDDEN),generator=g,requires_grad=True)
        emb=torch.zeros((VOCAB,HIDDEN),requires_grad=True)
        with torch.no_grad():
            emb[100]=torch.randn((HIDDEN,),generator=g)
            emb[101]=torch.randn((HIDDEN,),generator=g)
        loss,diagnostic=r.trainonly_loss(
            logits,h,emb,101,split="TRAIN",own_first_correct=True)
        self.assertTrue(diagnostic["eligible_for_gradient"])
        self.assertGreater(float(loss.detach()),0.)
        loss.backward()
        self.assertIsNone(logits.grad)
        self.assertIsNone(h.grad)
        self.assertIsNone(emb.grad)
        self.assertIsNotNone(r.right.grad)
        self.assertGreater(float(r.right.grad.abs().sum()),0.)
        self.assertEqual(len(r.state_dict()),2)
    def test_04_top64_human_gold_never_inserted_and_not_dev(self):
        r=NativeBPE2CandidateReranker()
        logits=torch.full((1,VOCAB),-999.)
        logits[0,:64]=torch.arange(64,dtype=torch.float32)
        emb=torch.zeros((VOCAB,HIDDEN))
        h=torch.randn((1,HIDDEN))
        loss,meta=r.trainonly_loss(
            logits,h,emb,35000,split="TRAIN",own_first_correct=True)
        self.assertEqual(float(loss.detach()),0.)
        self.assertFalse(meta["eligible_for_gradient"])
        loss,meta=r.trainonly_loss(
            logits,h,emb,60,split="TRAIN",own_first_correct=False)
        self.assertFalse(meta["eligible_for_gradient"])
        for split in ("DEV","GOLD","EVAL128","J7"):
            with self.assertRaises(ValueError):
                r.trainonly_loss(logits,h,emb,60,split=split,
                                 own_first_correct=True)
    def test_05_original43M_hook_reads_existing_h320_once_and_rolls_back(self):
        model=FakeOriginal()
        old=set(model.lm_head._forward_pre_hooks)
        with capture_existing_full43m_hidden(model) as readings:
            y=model.lm_head(torch.randn((1,7,HIDDEN)))
            self.assertEqual(y.shape,(1,7,12))
        self.assertEqual(len(readings),1)
        self.assertEqual(readings[0].shape,(1,HIDDEN))
        self.assertFalse(readings[0].requires_grad)
        self.assertEqual(set(model.lm_head._forward_pre_hooks),old)
        with self.assertRaises(ValueError):
            with capture_existing_full43m_hidden(model):
                pass
        self.assertEqual(set(model.lm_head._forward_pre_hooks),old)
    def test_06_strict_hidden_vocab_and_logits_dims(self):
        r=NativeBPE2CandidateReranker()
        good=torch.zeros((1,VOCAB))
        emb=torch.zeros((VOCAB,HIDDEN))
        for logits,h,e in (
            (torch.zeros((1,24)),torch.zeros((1,HIDDEN)),emb),
            (good,torch.zeros((1,HIDDEN-1)),emb),
            (good,torch.zeros((1,HIDDEN)),torch.zeros((100,HIDDEN))),
        ):
            with self.assertRaises(ValueError):r(logits,h,e)

if __name__=="__main__":
    unittest.main()
