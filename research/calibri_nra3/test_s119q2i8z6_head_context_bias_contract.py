"""S119q2i8z6 original43M actual native49152 context vs bias score contract."""
import unittest
from research.calibri_nra3.s119q2i8z6_head_context_bias_contract import (
    STAGE, STATES, POSITIONS, VARIANTS, EXPECTED_PHYSICAL,
    EXPECTED_ANALYTIC, summarize,
)
class TestZ6ContextVsBias(unittest.TestCase):
    def test_01_bounded_original43m_native_contract(self):
        self.assertIn("ORIGINAL43M_ALL12GT", STAGE)
        self.assertEqual(len(STATES),3)
        self.assertEqual(POSITIONS,(1,2))
        self.assertEqual(len(VARIANTS),4)
        self.assertEqual(EXPECTED_PHYSICAL,84)
        self.assertEqual(EXPECTED_ANALYTIC,336)
    def test_02_source_fail_closed_for_partial_cases(self):
        for v in ([], [{}]*84, [{}]*252, [{}]*335):
            with self.assertRaises(Exception):summarize(v)
    def test_03_synthetic_only_paired_four_virtual_readouts(self):
        vals=[]
        for model in STATES:
            for pos in POSITIONS:
                for ix in range(14):
                    for variation in VARIANTS:
                        vals.append({
                            "state":model,"position_native_zero_based":pos,
                            "source_id":f"Z2_{ix}","variant":variation,
                            "synthetic_author_Z2_DEV_ALREADY_SPENT":True,
                            "human_GOLD_PREVIOUS_native_context_only_NOT_OWN":True,
                            "correct_native_gold_ID":177,
                            "correct_native_rank_full49152":22,
                            "greedy_native_ID":6200,
                            "correct_native_top1":False,
                            "correct_native_top5":False,
                            "correct_native_NLL_nats":6.0,
                        })
        s=summarize(vals)
        self.assertEqual(s[STATES[0]]["read_only_gold_previous_native_position_2"][
            VARIANTS[0]]["same14_spent_source_count"],14)
        vals[0]["synthetic_author_Z2_DEV_ALREADY_SPENT"]=False
        with self.assertRaises(Exception):summarize(vals)
if __name__=="__main__": unittest.main()
