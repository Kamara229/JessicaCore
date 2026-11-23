"""Z28 precommit: full native49152 TRAIN BPE1-4 and EOS head-only objective."""
from __future__ import annotations
from collections import Counter
import unittest
import torch
from research.calibri_nra3.s119q2i8z28_precommit_84train14fresh_native_head_multistep import (
    rows,verify,digest,KINDS,TRAIN_EPOCHS,HEAD_GAIN,HEAD_LR,
    TRAIN_SUPERVISED_POSITIONS,TRAIN_WEIGHTS,EVAL_NATIVE_FIRST_K,
)
from research.calibri_nra3.s119q2i8z28_fullnative_trainonly_multitoken_head_loss import (
    select_train_positions,full_native_trainonly_pos_loss,VOCAB,EOS,HEAD_TOTAL_PARAMS,
    verify_shadow_head_params,
)

class TestZ28FutureNativeTrainOnly(unittest.TestCase):
    def test_01_legitimate_prior_train_unseen_new_dev(self):
        s=verify()
        self.assertEqual((s["prior_legal_TRAIN_count"],s["NEW_author_synthetic_DEV_count"]),(84,14))
        self.assertEqual(s["TRAIN_SHA256"],digest(rows("TRAIN")))
        self.assertEqual(s["DEV_SHA256"],digest(rows("DEV")))
        self.assertEqual(Counter(r["kind"] for r in rows("TRAIN")),{k:12 for k in KINDS})
        self.assertEqual(Counter(r["kind"] for r in rows("DEV")),{k:2 for k in KINDS})
        self.assertTrue(s["strict_no_previous_SPENT_DEV_in_training"])
        self.assertTrue(s["NEW_DEV_disjoint_all_old_including_Z25"])
        self.assertEqual((TRAIN_EPOCHS,HEAD_GAIN,HEAD_LR,EVAL_NATIVE_FIRST_K),
                         (2,.25,.006,6))
        self.assertEqual((TRAIN_SUPERVISED_POSITIONS,TRAIN_WEIGHTS),
                         ((0,1,2,3,-1),(2.,1.,1.,1.,2.)))
        self.assertEqual(HEAD_TOTAL_PARAMS,247040)
        for v in ("J7","GOLD","EVAL128","TRAIN+DEV","unknown"):
            with self.assertRaises(ValueError):rows(v)

    def test_02_first4_and_last_EOS_five_distinct_pinned(self):
        example=[128,129,130,131,132,134,EOS]
        self.assertEqual(select_train_positions(example),
                         [(0,2.),(1,1.),(2,1.),(3,1.),(6,2.)])
        with self.assertRaises(ValueError):
            select_train_positions([1,2,3,EOS])
        with self.assertRaises(ValueError):
            select_train_positions([1,2,3,4,5])

    def test_03_true49152_gradient_early_eos_and_selfrepeat(self):
        gold=[17,18,19,20,21,EOS]
        logits=torch.zeros((1,VOCAB),requires_grad=True)
        with torch.no_grad():
            logits[0,17]=1
            logits[0,EOS]=3
        loss,detail=full_native_trainonly_pos_loss(
            logits,gold,0,2.,split="TRAIN")
        self.assertGreater(float(loss),0)
        self.assertFalse(detail["is_REAL_native_EOS_target"])
        self.assertGreater(detail["premature_EOS_margin"],0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()),0)
        logits.grad=None
        eos_loss,detail=full_native_trainonly_pos_loss(
            logits,gold,5,2.,split="TRAIN")
        self.assertTrue(detail["is_REAL_native_EOS_target"])
        self.assertEqual(detail["premature_EOS_margin"],0.)
        eos_loss.backward()
        self.assertLess(float(logits.grad[0,EOS]),0.)

    def test_04_native_repeat_penalty_and_dev_never_fit(self):
        gold=[13,14,15,16,17,EOS]
        logits=torch.zeros((1,VOCAB),requires_grad=True)
        with torch.no_grad():
            logits[0,13]=4
            logits[0,14]=-1
        loss,d=full_native_trainonly_pos_loss(
            logits,gold,1,1.,split="TRAIN")
        self.assertGreater(d["echo_margin"],1.0)
        self.assertGreater(float(loss.detach()),0)
        for split in ("DEV","J7","EVAL128","GOLD"):
            with self.assertRaises(ValueError):
                full_native_trainonly_pos_loss(logits,gold,1,1.,split=split)
        with self.assertRaises(ValueError):
            full_native_trainonly_pos_loss(logits,gold,2,2.5,split="TRAIN")

    def test_05_head_params_real_orig_model_shape(self):
        from torch import nn
        class NativeHeadAlignmentShape(nn.Module):
            def __init__(self):
                super().__init__()
                self.down=nn.Linear(320,4,bias=False)
                self.up=nn.Linear(4,VOCAB,bias=False)
                self.output_bias=nn.Parameter(torch.zeros(VOCAB))
                self.output_gate=0
                self.candidate_accepted=False
                self.activation_allowed=False
        head=NativeHeadAlignmentShape()
        self.assertEqual(sum(p.numel() for p in head.parameters()),HEAD_TOTAL_PARAMS)
        self.assertTrue(verify_shadow_head_params(head))
        with self.assertRaises(ValueError):
            verify_shadow_head_params(head,original_old_head=head)

if __name__=="__main__":
    unittest.main()
