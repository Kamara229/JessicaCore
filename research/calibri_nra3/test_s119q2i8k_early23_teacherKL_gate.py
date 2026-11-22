"""S119q2i8k strictly bounded teacher 135M soft KL on second/third TRAIN BPE."""
import unittest
from research.calibri_nra3.s119q2i8k_real43m_secondthird_teacherKL_train import (
    teacher_kl_only_secondthird_gold_target,
)
class TestTrueReal43mEarlyTeacherKL(unittest.TestCase):
    def test_01_first_BPE_no_teacher(self):
        gold=[155,100,201,202,2]
        self.assertFalse(teacher_kl_only_secondthird_gold_target(0,155,gold,2))
    def test_02_second_third_only(self):
        gold=[155,100,201,202,2]
        self.assertTrue(teacher_kl_only_secondthird_gold_target(1,100,gold,2))
        self.assertTrue(teacher_kl_only_secondthird_gold_target(2,201,gold,2))
        self.assertFalse(teacher_kl_only_secondthird_gold_target(3,202,gold,2))
    def test_03_no_EOS_teacher(self):
        gold=[155,100,201,2]
        self.assertFalse(teacher_kl_only_secondthird_gold_target(3,2,gold,2))
    def test_04_wrong_gold_or_short_gated(self):
        gold=[155,100,201,2]
        with self.assertRaises(Exception):
            teacher_kl_only_secondthird_gold_target(1,999,gold,2)
        with self.assertRaises(Exception):
            teacher_kl_only_secondthird_gold_target(2,2,[155,100,2],2)
    def test_05_source_only_no_accidental_loss_of_native_eos(self):
        gold=[155,100,201,202,2]
        self.assertEqual([i for i,x in enumerate(gold)
                          if teacher_kl_only_secondthird_gold_target(i,x,gold,2)],
                         [1,2])
if __name__=="__main__":
    unittest.main()
