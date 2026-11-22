"""Z2 existing onpath core-gate train-only regularization true physical 43M."""
import unittest
from types import SimpleNamespace
import torch
from research.calibri_nra3.s119q2i8z2_true43m_joint_train_CoreL2 import (
    STAGE,EARLY23_MULTIPLIER,CORE_GATE_SHRINK_LAMBDA,
    train_ce_weights,actual_learned_Core_gate_tanh,trainonly_Core_regularizer,
    train_joint_true43m_shrink_Core_ab,
)
class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.midlayer_gates=torch.nn.Parameter(torch.tensor([.4,.3,-.2]))
class Child:
    def __init__(self):
        self.child=SimpleNamespace(child=Dummy())
class TestZ2CoreGateTrain(unittest.TestCase):
    def test_predeclared_train_only_gain(self):
        self.assertIn("CORE_GATE_SHRINK_TRAINONLY",STAGE)
        self.assertEqual(CORE_GATE_SHRINK_LAMBDA,2.)
        self.assertEqual(EARLY23_MULTIPLIER,3.)
    def test_control_same_gold_second_third_EOS(self):
        tok=SimpleNamespace(eos_token_id=2)
        for obj in ("ALL_NATIVE_CE_EARLY23_X3","ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE"):
            self.assertEqual(train_ce_weights([17,36,71,94,2],tok,objective=obj),
                             [1.,3.,3.,1.,1.])
    def test_penalty_grad_core_only_not_NRA3_or_NPA1(self):
        m=Child()
        base=trainonly_Core_regularizer(m,"ALL_NATIVE_CE_EARLY23_X3")
        self.assertEqual(float(base),0.)
        pen=trainonly_Core_regularizer(m,"ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE")
        self.assertGreater(float(pen),0.)
        pen.backward()
        g=m.child.child.midlayer_gates.grad
        self.assertGreater(float(g[0]),0.)
        self.assertEqual(float(g[1]),0.)
        self.assertEqual(float(g[2]),0.)
    def test_reject_invalid_control(self):
        with self.assertRaises(Exception):
            trainonly_Core_regularizer(Child(),"ALL_NATIVE_CE_PLUS_EOS_MARGIN")
        with self.assertRaises(Exception):
            train_joint_true43m_shrink_Core_ab(
                None,None,None,None,None,None,[],None,
                condition="GT_NOCARRY_LOCAL",expected_total=27,
                expected_new=21,rounds=1,new_id_prefix="S119Q2I8Z2_",
                objective="ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE")
if __name__=="__main__":unittest.main()
