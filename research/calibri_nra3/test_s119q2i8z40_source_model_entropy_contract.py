"""Source-only contract tests; no original S142 loading or physical scoring."""
import math
import unittest
import torch
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as s
from research.calibri_nra3.s119q2i8z40_causal_BPE234_entropy_guard import (
    CausalBPE234CandidateComparator, MODES, LOGIT_KL_CEILING
)


class TestZ40PrereRegistration(unittest.TestCase):
    def test_protected_splits_shas_and_no_previous_prompts(self):
        out=s.verify()
        self.assertTrue(out["zero_past_spent_or_train_prompt_overlap"])
        self.assertEqual((len(s.rows("TRAIN")),len(s.rows("DEV"))),(84,14))
        self.assertEqual((out["previous_legal_TRAIN84"],out["NEW_original_author_internal_DEV14"]),(84,14))
        self.assertEqual(s.digest(s.rows("DEV")),out["DEV_SHA256"])
        self.assertTrue(out["not_external_blind_heldout"])
        self.assertFalse(out["candidate_accepted"])
        self.assertEqual(out["output_gate"],0)
        self.assertEqual((s.SHADOW_PARAMS,s.POSITIONS,s.EVAL_OWN_NATIVE_TOKENS),(31680,(2,3,4),6))

    def test_output_rank_uses_frozen_49k_without_gold_candidates(self):
        net=CausalBPE234CandidateComparator()
        self.assertEqual(sum(p.numel() for p in net.parameters()),31680)
        logits=torch.zeros((1,49152))
        logits[0,400]=3.
        h=torch.randn(1,320)
        prev=torch.randn(1,320)
        emb=torch.randn(49152,320)
        for pos in s.POSITIONS:
            ids,scores,residual=net(logits,h,prev,emb,position=pos)
            self.assertEqual((len(ids),len(ids.unique()),len(scores)),(64,64,64))
            self.assertTrue((ids==400).any())
            self.assertLessEqual(float(residual.abs().max()),1.50001)
            self.assertEqual(int(scores.argmax()),0)
            with self.assertRaises(ValueError):
                net(logits,h,prev,emb,position=5)
        self.assertFalse(logits.requires_grad)
        self.assertFalse(emb.requires_grad)
        self.assertEqual(net.output_gate,0)

    def test_training_uses_only_tagged_teacher_TRAIN_not_dev(self):
        net=CausalBPE234CandidateComparator()
        logits=torch.zeros((1,49152))
        logits[0,22]=4.
        emb=torch.randn(49152,320)
        row={"source_id":"S119Q2I8Z40_LEGAL_TEST","split":"TRAIN",
             "position":3,"gold_prefix_only":True,"gold_token":22,
             "original_logits":logits,"h320":torch.randn(1,320),
             "prior_emb":torch.randn(1,320)}
        loss=net.train_loss(row,emb,wrong_label=False)
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertTrue(all(v.grad is not None for v in net.parameters()))
        self.assertIsNone(logits.grad)
        self.assertIsNone(emb.grad)
        bad=dict(row,split="DEV")
        with self.assertRaises(ValueError):
            net.train_loss(bad,emb,wrong_label=False)
        bad=dict(row,gold_prefix_only=False)
        with self.assertRaises(ValueError):
            net.train_loss(bad,emb,wrong_label=False)

    def test_inference_entropy_guard_uses_no_gold_and_limits_collapse(self):
        baseline=torch.linspace(3.,-2.,64)
        modified=baseline+torch.linspace(2.,-2.,64)
        out,alpha,delta=CausalBPE234CandidateComparator.guarded_scores(baseline,modified)
        self.assertIn(alpha,(1.,.75,.5,.375,.25,.125,.0625,.03125,0.))
        self.assertGreaterEqual(delta,-s.ENTROPY_FLOOR_TOLERANCE_BITS-1e-6)
        kl=torch.nn.functional.kl_div(torch.nn.functional.log_softmax(out,dim=0),
            torch.nn.functional.softmax(baseline,dim=0),reduction="sum")
        self.assertLessEqual(float(kl),LOGIT_KL_CEILING+1e-6)
        self.assertEqual(len(MODES),2)


if __name__=="__main__":
    unittest.main()
