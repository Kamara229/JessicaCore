"""S119q2i8r supervised full 49152 native learned EOS margin target only."""
import unittest
import torch
from research.calibri_nra3.s119q2i8r_headonly_fullnative_eos_margin_train import (
    STAGE,eos_margin_loss,VOCAB,GOLD_EOS,TRAINABLE_HEAD,EXPECTED_TRAIN,
)
class TestEOSMargin(unittest.TestCase):
    def test_01_predeclared_current_arch_and_train_budget(self):
        self.assertEqual(VOCAB,49152)
        self.assertEqual(GOLD_EOS,2)
        self.assertEqual(TRAINABLE_HEAD,247040)
        self.assertEqual(EXPECTED_TRAIN,34)
    def test_02_train_gold_prefers_correct_not_EOS(self):
        logits=torch.zeros((1,VOCAB),requires_grad=True)
        e=eos_margin_loss(logits,125,2,is_last_gold_BPE=False)
        e.backward()
        self.assertGreater(float(logits.grad[0,2]),0.)
        self.assertLess(float(logits.grad[0,125]),0.)
    def test_03_EOS_only_when_terminal_BPE(self):
        logits=torch.zeros((1,VOCAB),requires_grad=True)
        x=eos_margin_loss(logits,2,2,is_last_gold_BPE=True)
        x.backward()
        self.assertLess(float(logits.grad[0,2]),0.)
    def test_04_no_wrong_or_nonterminal_EOS_labels(self):
        logits=torch.zeros((1,VOCAB),requires_grad=True)
        for target,last in ((2,False),(44,True)):
            with self.assertRaises(Exception):
                eos_margin_loss(logits,target,2,is_last_gold_BPE=last)
    def test_05_full_real49k_only(self):
        with self.assertRaises(Exception):
            eos_margin_loss(torch.zeros((1,500)),24,2,is_last_gold_BPE=False)
if __name__=="__main__":
    unittest.main()
