"""S119q2i8o: gain=0/0.25/1 true native-head hook and rank invariants.

Synthetic tensor unit tests only, NEVER opens protected J7 / real DEV labels.
"""
import unittest
import torch
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    NativeHeadAlignment,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    scoped_reduced_native_head_gain,
)
from research.calibri_nra3.s119q2i8o_native_head_gain_physical_ablation import (
    scoped_exact_native_gain,real_native_rank,score_by_position,GAINS,ARMS,LABELS,
)
class TestActualGainAB(unittest.TestCase):
    def test_01_only_three_preregistered_gain_values(self):
        self.assertEqual(GAINS,(0.,.25,1.))
        self.assertEqual(len(ARMS),2)
        self.assertEqual(len(LABELS),6)
    def test_02_strict_hook_scaling_and_cleanup(self):
        torch.manual_seed(155)
        head=NativeHeadAlignment()
        x=torch.randn((1,1,320))
        with torch.no_grad():
            head.up.weight.fill_(.01)
        original=head(x)
        before=set(head._forward_hooks)
        outs={}
        for gain in GAINS:
            with scoped_exact_native_gain(head,gain) as cnt:
                outs[gain]=head(x).detach()
            self.assertEqual(cnt[0],1)
            self.assertEqual(set(head._forward_hooks),before)
        self.assertTrue(torch.allclose(outs[0.],torch.zeros_like(original)))
        self.assertTrue(torch.allclose(outs[1.],original))
        self.assertTrue(torch.allclose(outs[.25],original*.25))
    def test_03_gain_point25_identical_to_existing_original_research_hook(self):
        torch.manual_seed(156)
        head=NativeHeadAlignment()
        with torch.no_grad():
            head.output_bias.fill_(.02)
        x=torch.randn((1,1,320))
        with scoped_reduced_native_head_gain(head) as historical:
            old=head(x)
        with scoped_exact_native_gain(head,.25) as actual:
            new=head(x)
        self.assertEqual(historical[0],actual[0])
        self.assertTrue(torch.equal(new,old))
    def test_04_invalid_gain_must_refuse(self):
        head=NativeHeadAlignment()
        for gain in (.5,2.,-1.,1,False):
            with self.assertRaises(Exception):
                with scoped_exact_native_gain(head,gain):
                    pass
    def test_05_real_rank_49152_distribution(self):
        x=torch.full((1,49152),-7.)
        x[0,4]=8.
        x[0,5]=10.
        self.assertEqual(real_native_rank(x,4)["correct_gold_native_rank_full49152"],2)
    def test_06_exact14_positions_all6_sources_aggregates(self):
        records=[]
        for pos in (1,2):
            for n in range(14):
                p={"correct_gold_native_rank_full49152":2,
                   "gold_native_top1":False,
                   "correct_gold_rank_top5":True,
                   "wrong_top1_minus_gold_logit_gap":.1,
                   "gold_log_probability_native_49152":-3.}
                records.append({
                    "native_gold_answer_index_0based":pos,
                    "genuine_49152_head_gain_results":{k:p for k in LABELS}
                })
        result=score_by_position(records)
        for pos in ("gold_position_2","gold_position_3"):
            self.assertEqual(result[pos][LABELS[0]]["source_count"],14)
            self.assertEqual(result[pos][LABELS[0]]["gold_native_top5"],14)
if __name__=="__main__":
    unittest.main()
