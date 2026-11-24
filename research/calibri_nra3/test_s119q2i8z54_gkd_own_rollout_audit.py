import unittest
from pathlib import Path
import torch
from research.calibri_nra3.s119q2i8z54_gkd_own_rollout_audit import trace,aggregate,MAX_STEPS
class TestZ54GKD(unittest.TestCase):
    def test_same_student_prefix_no_gold_after_divergence(self):
        s_calls=[];t_calls=[]
        def logits(i):
            x=torch.zeros(1,8);x[0,i]=4.;return x
        def student(p):
            s_calls.append(list(p));return logits(4 if len(p)==1 else 2)
        def teacher(p):
            t_calls.append(list(p));return logits(3 if len(p)==1 else 5)
        rows=trace([1],[3,5,2],student,teacher,vocab=8)
        self.assertEqual(s_calls,t_calls)
        self.assertEqual(s_calls,[[1],[1,4]])
        self.assertTrue(rows[0]["teacher_rescues_if_aligned"])
        self.assertFalse(rows[1]["aligned_reference_prefix"])
        self.assertIsNone(rows[1]["teacher_correct_if_aligned"])
        self.assertIsNone(rows[1]["student_premature_eos_if_aligned"])
        a=aggregate([{"steps":rows}])
        self.assertEqual(a["positions_total"],2)
        self.assertEqual(a["aligned"]["teacher_rescues_if_aligned"],1)
        self.assertEqual(a["diverged_no_reference_correctness_claim"]["positions"],1)
    def test_cap_and_eos(self):
        def step(p):
            x=torch.zeros(1,8);x[0,4]=2.;return x
        self.assertEqual(len(trace([1],[4,2],step,step,vocab=8)),MAX_STEPS)
    def test_no_optimizer_or_dev(self):
        p=Path(__file__).with_name("s119q2i8z54_physical43m_gkd_rollout_audit.py").read_text()
        self.assertIn('fresh_new_rows("TRAIN")',p)
        self.assertNotIn('fresh_new_rows("DEV")',p)
        self.assertNotIn(".backward()",p)
        self.assertIn('original_forwards==teacher_forwards',p)
if __name__=="__main__":unittest.main()
