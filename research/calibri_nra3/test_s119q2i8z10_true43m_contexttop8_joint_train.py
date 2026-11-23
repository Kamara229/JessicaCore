"""S119q2i8z10 actual-43M training loss wiring guard; no real model in unit tests."""
import unittest
from types import SimpleNamespace
import torch
from research.calibri_nra3.s119q2i8z10_true43m_joint_train_contexttop8 import (
    STAGE,EARLY23_MULTIPLIER,CONTEXT_TOP8_MARGIN_WEIGHT,train_ce_weights,
    trainonly_context_margin,train_joint_true43m_context_top8_ab,
)
from research.calibri_nra3.s119q2i8z8_trainonly_second_bpe_context_margin import (
    NATIVE_VOCAB, COMPETITOR_TOPK,
)

class TestZ10ContextMarginTrain(unittest.TestCase):
    def test_locked_baseline_3x_and_top8_weight(self):
        self.assertIn("CONTEXTUAL_TOP8",STAGE)
        self.assertEqual(EARLY23_MULTIPLIER,3.0)
        self.assertEqual(CONTEXT_TOP8_MARGIN_WEIGHT,0.20)
        self.assertEqual(COMPETITOR_TOPK,8)
        self.assertEqual(NATIVE_VOCAB,49152)
        tok=SimpleNamespace(eos_token_id=2)
        for obj in ("ALL_NATIVE_CE_EARLY23_X3","ALL_NATIVE_CE_EARLY23_X3_CONTEXT_TOP8"):
            self.assertEqual(train_ce_weights([17,36,71,94,2],tok,objective=obj),
                             [1.,3.,3.,1.,1.])
    def test_same_finite_native_logits_control_zero_vs_B_gradient(self):
        x=torch.zeros((1,NATIVE_VOCAB),requires_grad=True)
        a=trainonly_context_margin(x,17,objective="ALL_NATIVE_CE_EARLY23_X3",
                                   source_split="TRAIN",index=1)
        b=trainonly_context_margin(x,17,objective="ALL_NATIVE_CE_EARLY23_X3_CONTEXT_TOP8",
                                   source_split="TRAIN",index=1)
        self.assertEqual(float(a.detach()),0.)
        self.assertGreater(float(b.detach()),0.)
        b.backward()
        self.assertLess(float(x.grad[0,17]),0.)
    def test_fail_closed_no_dev_own_or_other_position(self):
        x=torch.zeros((1,NATIVE_VOCAB))
        for split in ("DEV","EVAL128","GOLD","J7"):
            with self.assertRaises(Exception):
                trainonly_context_margin(
                    x,17,objective="ALL_NATIVE_CE_EARLY23_X3_CONTEXT_TOP8",
                    source_split=split,index=1)
        with self.assertRaises(Exception):
            trainonly_context_margin(x,17,objective="ALL_NATIVE_CE_EARLY23_X3_CONTEXT_TOP8",
                                     source_split="TRAIN",index=2)
        with self.assertRaises(Exception):
            train_joint_true43m_context_top8_ab(
                None,None,None,None,None,None,[],None,
                condition="GT_NOCARRY_LOCAL",expected_total=27,expected_new=21,
                rounds=1,new_id_prefix="S119Q2I8Z9_",
                objective="ALL_NATIVE_CE_EARLY23_X3_CONTEXT_TOP8")
if __name__=="__main__":unittest.main()
