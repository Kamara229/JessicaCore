"""S119q2i8x original49152 strongest wrong nonEOS ranking loss safety."""
import unittest
import torch
from types import SimpleNamespace
from research.calibri_nra3.s119q2i8x_true43m_joint_native_hardneg_train import (
    STAGE,EARLY23_MULTIPLIER,HARDNEG_LAMBDA,train_ce_weights,
    hardneg_native49k_nonEOS,train_joint_true43m_early23_hardneg_ab,
)
class TestXHardNegative(unittest.TestCase):
    def test_01_exact_precommitted_gain(self):
        self.assertEqual(EARLY23_MULTIPLIER,3.)
        self.assertEqual(HARDNEG_LAMBDA,.5)
        self.assertIn("FULL_GT12_CORE_NRA3_NPA1",STAGE)
    def test_02_same_native_BPE2_3_weight_for_both(self):
        tok=SimpleNamespace(eos_token_id=2)
        for objective in ("ALL_NATIVE_CE_EARLY23_X3","ALL_NATIVE_CE_EARLY23_X3_HARDNEG"):
            self.assertEqual(train_ce_weights([33,41,78,28,2],tok,
                             objective=objective),[1.,3.,3.,1.,1.])
    def test_03_gold_gradient_good_wrong_gradient_bad_EOS_untouched(self):
        z=torch.full((1,49152),-8.,requires_grad=True)
        with torch.no_grad():
            z[0,2]=12.;z[0,13]=5.;z[0,14]=4.
        v,bad=hardneg_native49k_nonEOS(z,14)
        self.assertEqual(bad,13)
        v.backward()
        self.assertLess(float(z.grad[0,14]),0.)
        self.assertGreater(float(z.grad[0,13]),0.)
        self.assertEqual(float(z.grad[0,2]),0.)
        self.assertEqual(float(z[0,2]),12.)
    def test_04_fail_closed_vocab_EOS_and_nan(self):
        for z,gold in ((torch.zeros((1,49152)),2),
                       (torch.zeros((1,5)),4),
                       (torch.full((1,49152),float("nan")),11)):
            with self.assertRaises(Exception):hardneg_native49k_nonEOS(z,gold)
    def test_05_invalid_source_rejected(self):
        with self.assertRaises(Exception):
            train_joint_true43m_early23_hardneg_ab(
                None,None,None,None,None,None,[],None,
                condition="GT_NOCARRY_LOCAL",expected_new=21,
                expected_total=27,rounds=1,new_id_prefix="S119Q2I8X_",
                objective="ALL_NATIVE_CE_EARLY23_X3_HARDNEG")
if __name__=="__main__":unittest.main()
