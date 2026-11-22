"""S119q2i8j genuine full 49152 native rank math and no-training contract."""
import unittest
import torch

from research.calibri_nra3.s119q2i8j_original43m_native_goldsecond_rank import (
    NATIVE_VOCAB,POSITIONS,STAGE,LABELS,real_native_rank,score_by_position,
)

class TestRealNativeSecondThirdGoldRank(unittest.TestCase):
    def test_01_true_full_native_gold_rank(self):
        x=torch.full((1,NATIVE_VOCAB),-9.)
        x[0,17]=2.
        x[0,30]=4.
        x[0,49]=3.
        val=real_native_rank(x,17)
        self.assertEqual(val["correct_gold_native_rank_full49152"],3)
        self.assertEqual(val["greedy_original_native_id"],30)
        self.assertEqual(val["wrong_top1_minus_gold_logit_gap"],2.)
        self.assertFalse(val["gold_native_top1"])
        self.assertTrue(val["correct_gold_rank_top5"])
    def test_02_rank1_when_true_top1(self):
        x=torch.zeros((1,NATIVE_VOCAB))
        x[0,123]=7.
        d=real_native_rank(x,123)
        self.assertTrue(d["gold_native_top1"])
        self.assertEqual(d["correct_gold_native_rank_full49152"],1)
        self.assertEqual(d["wrong_top1_minus_gold_logit_gap"],0.)
    def test_03_finiteness_and_vocab_fail_closed(self):
        with self.assertRaises(Exception):
            real_native_rank(torch.zeros((1,7)),0)
        x=torch.zeros((1,NATIVE_VOCAB))
        x[0,100]=torch.nan
        with self.assertRaises(Exception):
            real_native_rank(x,13)
    def test_04_exact_author_dev_second_third_labels_only(self):
        self.assertEqual(POSITIONS,(1,2))
        self.assertEqual(len(LABELS),4)
        self.assertIn("GOLD_SECOND_THIRD",STAGE)
    def test_05_source_only_14x2_aggregate(self):
        records=[]
        for p in POSITIONS:
            for j in range(14):
                records.append({
                    "native_gold_answer_index_0based":p,
                    "four_source_original_native_results":{
                        n:{"correct_gold_native_rank_full49152":3,
                           "wrong_top1_minus_gold_logit_gap":2.0,
                           "gold_native_top1":False,
                           "correct_gold_rank_top5":True,
                           "gold_log_probability_native_49152":-8.0}
                        for n in LABELS
                    }
                })
        score=score_by_position(records)
        self.assertEqual(score["gold_position_2"][LABELS[0]]["source_count"],14)
        self.assertEqual(score["gold_position_2"][LABELS[0]]["gold_native_top1"],0)
        self.assertEqual(score["gold_position_3"][LABELS[0]]["mean_gold_rank"],3.)
        self.assertEqual(score["gold_position_2"]["PAIRED_KL_rank_vs_CE"]["same_gold_rank"],14)
if __name__=="__main__":
    unittest.main()
