"""Z28 prereg FIRST4 + real EOS fullnative49152 training, no past DEV leak."""
import unittest
from collections import Counter
import torch
from research.calibri_nra3.s119q2i8z28_precommit_84train14fresh_native_head_multistep import (
    rows,verify,digest,KINDS,TRAIN_EPOCHS,HEAD_LR,HEAD_GAIN,
    TRAIN_SUPERVISED_POSITIONS,TRAIN_WEIGHTS,
)
from research.calibri_nra3.s119q2i8z28_fullnative_trainonly_multitoken_head_loss import (
    VOCAB,EOS,HEAD_TOTAL_PARAMS,
    select_train_positions,full_native_trainonly_pos_loss,
)

class TestZ28MultiBPERepetitionFixPrereg(unittest.TestCase):
    def test_01_only_prior_legitimate_84_train_vs_completely_new_14_dev(self):
        s=verify()
        self.assertEqual(s["prior_legal_TRAIN_count"],84)
        self.assertEqual(s["NEW_author_synthetic_DEV_count"],14)
        self.assertEqual(s["TRAIN_SHA256"],digest(rows("TRAIN")))
        self.assertEqual(s["DEV_SHA256"],digest(rows("DEV")))
        self.assertEqual(Counter(r["kind"] for r in rows("TRAIN")),{k:12 for k in KINDS})
        self.assertEqual(Counter(r["kind"] for r in rows("DEV")),{k:2 for k in KINDS})
        self.assertTrue(s["strict_no_previous_SPENT_DEV_in_training"])
        self.assertTrue(s["NEW_DEV_disjoint_all_old_including_Z25"])
        self.assertEqual((TRAIN_EPOCHS,HEAD_LR,HEAD_GAIN),(2,.006,.25))
        self.assertEqual(HEAD_TOTAL_PARAMS,247040)
        self.assertEqual((TRAIN_SUPERVISED_POSITIONS,TRAIN_WEIGHTS),
                         ((0,1,2,3,-1),(2.,1.,1.,1.,2.)))
        for bad in ("J7","EVAL128","GOLD","TEST","TRAIN+DEV"):
            with self.assertRaises(ValueError):rows(bad)
    def test_02_teacher_forced_human_BPE1_4_and_real_final_EOS(self):
        gold=[155,111,344,788,901,2]
        actual=select_train_positions(gold)
        self.assertEqual([i for i,w in actual],[0,1,2,3,5])
        self.assertEqual([w for i,w in actual],[2.,1.,1.,1.,2.])
        with self.assertRaises(ValueError):
            select_train_positions([155,111,2])
        with self.assertRaises(ValueError):
            select_train_positions([155,111,344,788,901,7])
    def test_03_full_native49152_true_EOS_and_repeat_margin_gradient(self):
        native=torch.zeros((1,VOCAB),requires_grad=True)
        gold=[155,111,344,788,901,2]
        m=[]
        for idx,w in select_train_positions(gold):
            loss,meta=full_native_trainonly_pos_loss(
                native,gold,idx,w,split="TRAIN")
            self.assertTrue(meta["TRAIN_correct_next_BPE_target_only"])
            m.append(meta)
            self.assertTrue(loss.requires_grad)
            loss.backward(retain_graph=True)
        self.assertEqual(sum(x["is_REAL_native_EOS_target"] for x in m),1)
        self.assertGreater(float(native.grad.abs().sum()),0.)
        with self.assertRaises(ValueError):
            full_native_trainonly_pos_loss(native,gold,1,1.,split="DEV")
        with self.assertRaises(ValueError):
            full_native_trainonly_pos_loss(native,gold,4,1.,split="TRAIN")
    def test_04_native_repeat_margin_applies_only_when_gold_differs(self):
        native=torch.zeros((1,VOCAB),requires_grad=True)
        a=[155,111,333,222,900,2]
        loss,meta=full_native_trainonly_pos_loss(
            native,a,2,1.,split="TRAIN")
        self.assertGreater(meta["echo_margin"],0.)
        self.assertGreater(meta["premature_EOS_margin"],0.)
        b=[155,111,111,222,900,2]
        loss2,meta2=full_native_trainonly_pos_loss(
            native,b,2,1.,split="TRAIN")
        self.assertEqual(meta2["echo_margin"],0.)
        self.assertTrue(torch.isfinite(loss2))
if __name__=="__main__":
    unittest.main()
