"""S119q2i8z4 fail closed native49152 original readout physical gain audit."""
import unittest
from research.calibri_nra3.s119q2i8z4_native_gain_contract import (
    STAGE,STATES,POSITIONS,GAINS,EXPECTED_PHYSICAL_FORWARDS,
    classify_summary,NATIVE_VOCAB
)
class TestZ4NativeGainContract(unittest.TestCase):
    def test_01_exact_predeclared_full43M_physical_budget(self):
        self.assertIn("TRUE43M_12GT",STAGE)
        self.assertEqual(NATIVE_VOCAB,49152)
        self.assertEqual(len(STATES),3)
        self.assertEqual(POSITIONS,(1,2))
        self.assertEqual(GAINS,(0.,.25,1.))
        self.assertEqual(EXPECTED_PHYSICAL_FORWARDS,252)
    def test_02_fail_closed_on_partial_or_altered_original_forward_counts(self):
        for values in ([],[{}]*251,[{}]*253):
            with self.assertRaises(Exception):
                classify_summary(values)
    def test_03_true_gold_next_token_zero_EOS_and_matched_three_gains(self):
        vals=[]
        for state in STATES:
            for pos in POSITIONS:
                for idx in range(14):
                    for g in GAINS:
                        vals.append({"state":state,"position_native_zero_based":pos,
                          "source_id":f"Z2_{idx}","native_head_gain":g,
                          "source_already_spent_author_Z2_DEV":True,
                          "given_GOLD_PREVIOUS_BPE_only_NOT_OWN":True,
                          "correct_gold_native_id":140,
                          "correct_gold_native_rank_full49152":10,
                          "greedy_native_id":444,
                          "correct_gold_top1":False,"correct_gold_rank_top5":False,
                          "correct_gold_native_NLL_nats":7.2,
                          "wrong_top1_minus_gold_native_logit_gap":5.5})
        report=classify_summary(vals)
        self.assertEqual(report[STATES[1]]["teacherforced_previous_gold_position_2"][
                         "native_rank4_gain_0.25"]["n_ALREADY_SPENT_author_DEV"],14)
        vals[0]["source_id"]="OTHER_NOT_SAME_SOURCE"
        with self.assertRaises(Exception):classify_summary(vals)
if __name__=="__main__": unittest.main()
