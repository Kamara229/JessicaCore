"""S119q2i8t joint original43M EOS-vs-gold TRAIN safety contracts."""
import unittest
import torch
from research.calibri_nra3.s119q2i8t_true43m_joint_native_eos_margin_train import (
    STAGE,MARGIN_WEIGHT,train_joint_true43m_ce_vs_eosmargin,HEAD_GAIN,
)
from research.calibri_nra3.s119q2i8r_headonly_fullnative_eos_margin_train import eos_margin_loss
class TestJointNativeEOS(unittest.TestCase):
    def test_01_exact_existing_arch_and_preregistered_lambda(self):
        self.assertEqual(HEAD_GAIN,.25)
        self.assertEqual(MARGIN_WEIGHT,1.)
        self.assertIn("REAL43M_FULL_JOINT",STAGE)
    def test_02_margin_has_human_not_manual_fixed_logits(self):
        log=torch.zeros((1,49152),requires_grad=True)
        a=eos_margin_loss(log,103,2,is_last_gold_BPE=False)
        a.backward()
        self.assertGreater(float(log.grad[0,2]),0)
        self.assertLess(float(log.grad[0,103]),0)
    def test_03_last_EOS_human_gradient_negative(self):
        log=torch.zeros((1,49152),requires_grad=True)
        a=eos_margin_loss(log,2,2,is_last_gold_BPE=True)
        a.backward()
        self.assertLess(float(log.grad[0,2]),0)
    def test_04_training_function_without_source_or_valid_condition_fails_closed(self):
        with self.assertRaises(Exception):
            train_joint_true43m_ce_vs_eosmargin(
                None,None,None,None,None,None,[],None,
                condition="LOCAL",expected_new=21,
                expected_total=27,new_id_prefix="S119Q2I8T_",
                rounds=1,continuation_weight=1.,objective="ALL_NATIVE_CE")
if __name__=="__main__":
    unittest.main()
