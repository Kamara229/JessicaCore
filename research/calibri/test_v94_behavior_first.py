"""Small pure-data unit regression checks, offline, no original TEST."""
import unittest
from v94.answers import paired
from v94.policy import evaluate
from v94.repair_plan import plan
from v94.generation import NEW_PROMPTS
from v94.provenance import BOOLQ_RANGE,SCIQ_RANGE

def rec(i,g,p):
    return {"record_index":i,"prompt_SHA":str(i),"gold":g,"pred":p,"correct":int(g==p)}

class TestV94(unittest.TestCase):
    def test_new_holdout_outside_previous_selection(self):
        self.assertTrue(BOOLQ_RANGE[0]>=6032)
        self.assertEqual(BOOLQ_RANGE[1]-BOOLQ_RANGE[0],128)
        self.assertEqual(SCIQ_RANGE[1]-SCIQ_RANGE[0],64)
        self.assertGreaterEqual(len(NEW_PROMPTS),12)

    def test_paired_real_correct_answers_not_logits(self):
        a=[rec(i,i%2,0) for i in range(64)]
        b=[rec(i,i%2,i%2) for i in range(64)]
        s=paired("BoolQ",a,b,2)
        self.assertEqual(s["baseline"]["correct"],32)
        self.assertEqual(s["rewritten"]["correct"],64)
        self.assertEqual(s["correct_answer_gain"],32)
        self.assertGreater(s["paired_95_interval"][0],0)

    def test_sciq_four_way_valid(self):
        a=[rec(i,i%4,0) for i in range(64)]
        b=[rec(i,i%4,i%4) for i in range(64)]
        s=paired("SciQ",a,b,4)
        self.assertEqual(s["rewritten"]["balanced_accuracy"],1)

    def test_quality_not_distances_and_lost_domain_rejected(self):
        baseline={"count":128,"correct":50,"balanced_accuracy":.5}
        qa={"BoolQ":{"baseline":baseline,
                    "rewritten":{**baseline,"correct":52,"balanced_accuracy":.52,"majority_answer_fraction":.5},
                    "correct_answer_gain":2,"accuracy_gain":2/128},
            "SciQ":{"baseline":{"count":64,"correct":20,"balanced_accuracy":.30},
                    "rewritten":{"count":64,"correct":19,"balanced_accuracy":.29,"majority_answer_fraction":.5},
                    "correct_answer_gain":-1,"accuracy_gain":-1/64}}
        generation={"original":{"mean_distinct_ratio":.2,"mean_most_frequent_share":.8},
            "candidate":{"mean_distinct_ratio":.35,"mean_most_frequent_share":.5},
            "paired_full_model_greedy_decode_speed_ratio":.96}
        rate={"source_all_in_bpp":.668}
        result=evaluate(qa,generation,rate)
        self.assertEqual(result["status"],"REJECT_FUNCTIONAL_REGRESSION")
        self.assertNotIn("KL",str(result))
        repair=plan(qa,generation,rate,result)
        self.assertEqual(repair["observed_ranked_incidents"][0]["domain"],"SciQ")

    def test_hard_speed_gate(self):
        qa={}
        for name in ("BoolQ","SciQ"):
            qa[name]={"baseline":{"count":128,"correct":60,"balanced_accuracy":.47},
                      "rewritten":{"count":128,"correct":75,"balanced_accuracy":.54},
                      "correct_answer_gain":15,"accuracy_gain":15/128}
        g={"original":{"mean_distinct_ratio":.2},
           "candidate":{"mean_distinct_ratio":.38,"mean_most_frequent_share":.44},
           "paired_full_model_greedy_decode_speed_ratio":.82}
        decision=evaluate(qa,g,{"source_all_in_bpp":.668})
        self.assertEqual(decision["status"],"REJECT_FUNCTIONAL_REGRESSION")

if __name__=="__main__":unittest.main()
