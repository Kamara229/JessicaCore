"""v96 direct-answer grading and gibberish alarms, standard-library tests only."""
import unittest
from v96.spec import CANARIES,OPEN_ENDED
from v96.grade import grade_answer,string_breakdown,paired_behavior

class CanaryTests(unittest.TestCase):
    def test_explicit_real_ground_truth_and_no_substring_hacks(self):
        self.assertTrue(grade_answer("22.",("22",))["correct"])
        self.assertTrue(grade_answer("Answer: 22, because 13+9=22",("22",))["correct"])
        self.assertFalse(grade_answer("122 22 22",("22",))["correct"])
        self.assertFalse(grade_answer("Unknown; perhaps 22",("22",))["correct"])
        self.assertFalse(grade_answer("mersemersemersemerse",("22",))["correct"])
        self.assertTrue(grade_answer("Pacific Ocean.",("Pacific","Pacific Ocean"))["correct"])

    def test_massive_repetition_not_mistaken_for_quality(self):
        scores=string_breakdown("mersemersemerse", [1234]*32)
        self.assertTrue(scores["repeat_over_75pct"])
        self.assertTrue(scores["only_one_distinct_token"])
        self.assertAlmostEqual(scores["unique_token_fraction"],1/32)
        self.assertTrue(scores["requires_independent_semantic_review"])

    def test_all_prompts_new_and_freeform_unscored(self):
        self.assertGreaterEqual(len(CANARIES),8)
        self.assertEqual(len(OPEN_ENDED),4)
        texts=[p for _,p,_ in CANARIES]
        self.assertEqual(len(texts),len(set(texts)))

    def test_paired_correct_answers_only_not_fp32_output_mimicry(self):
        def item(s,correct):
            return {"prompt_SHA":s,"graded":{"correct":correct}}
        a=[item("a",False),item("b",True)]
        b=[item("a",True),item("b",True)]
        teacher=[item("a",True),item("b",True)]
        paired=paired_behavior(a,b,teacher)
        self.assertEqual(paired["rewrite_correct_answer_gain"],1)
        self.assertTrue(paired["original_FP32_answer_quality_preserved"])
        b[1]["prompt_SHA"]="not matched"
        with self.assertRaises(AssertionError):paired_behavior(a,b,teacher)

if __name__=="__main__":unittest.main()
