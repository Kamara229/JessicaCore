"""S119q2i8w strict original43M native49152 supervised SECOND+THIRD training."""
import unittest
from types import SimpleNamespace
from research.calibri_nra3.s119q2i8w_true43m_joint_native_early23_train import (
    STAGE,EARLY23_MULTIPLIER,EARLY23_POSITIONS,train_ce_weights,
    train_joint_true43m_early23_ce_ab,
)
class TestEarly23Weights(unittest.TestCase):
    def setUp(self):
        self.tok=SimpleNamespace(eos_token_id=2)
    def test_01_predeclared_gain_no_new_region(self):
        self.assertEqual(EARLY23_MULTIPLIER,3.0)
        self.assertEqual(EARLY23_POSITIONS,(1,2))
        self.assertIn("FULL_GT12_CORE_NRA3_NPA1",STAGE)
    def test_02_human_native_CE_2nd_3rd_only(self):
        ids=[83,445,276,918,2]
        self.assertEqual(train_ce_weights(ids,self.tok,objective="ALL_NATIVE_CE"),
                         [1.,1.,1.,1.,1.])
        self.assertEqual(train_ce_weights(ids,self.tok,objective="ALL_NATIVE_CE_EARLY23_X3"),
                         [1.,3.,3.,1.,1.])
    def test_03_short_gold_NEVER_weights_native_EOS(self):
        self.assertEqual(train_ce_weights([83,445,2],self.tok,
                          objective="ALL_NATIVE_CE_EARLY23_X3"),[1.,3.,1.])
    def test_04_protect_terminal_and_vocab(self):
        for seq in ([99,2,5],[5,2],[99,49152,2],[99,2,2]):
            with self.assertRaises(Exception):
                train_ce_weights(seq,self.tok,objective="ALL_NATIVE_CE_EARLY23_X3")
    def test_05_invalid_training_inputs_fail_closed(self):
        with self.assertRaises(Exception):
            train_joint_true43m_early23_ce_ab(
                None,None,None,None,None,None,[],None,
                condition="GT_NOCARRY_LOCAL",
                expected_total=27,expected_new=21,rounds=1,
                new_id_prefix="S119Q2I8W_",objective="ALL_NATIVE_CE")
if __name__=="__main__":unittest.main()
