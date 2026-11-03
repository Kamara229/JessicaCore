"""v93 safety: tested outcomes not teacher similarities decide candidate quality."""
import unittest
from v93.answers import compare_answers
from v93.policy import assess
from v93.evidence import extract

def item(i,gold,pred):
    return {"record_index":i,"gold":gold,"pred":pred,
        "correct":int(gold==pred),"prompt_SHA":f"hash{i}"}

def sample():
    a=[item(i,0 if i%2 else 1,0) for i in range(12)]
    b=[item(i,0 if i%2 else 1,0 if i<6 else (0 if i%2 else 1)) for i in range(12)]
    return {"original":a,"candidate":b}

def generation(a=.05,b=.15,repeat=.75):
    return {"baseline":{"mean_distinct_ratio":a,"mean_most_frequent_share":.9},
            "candidate":{"mean_distinct_ratio":b,"mean_most_frequent_share":repeat}}

class TestQualityFirst(unittest.TestCase):
    def test_correctness_balanced_by_truth_and_uncertainty(self):
        score=compare_answers(sample())
        self.assertEqual(score["original"]["count"],12)
        self.assertEqual(score["original"]["correct"],6)
        self.assertEqual(score["candidate"]["correct"],9)
        self.assertEqual(score["correct_answer_gain"],3)
        self.assertEqual(score["original"]["by_gold"]["no"]["count"],6)

    def test_quality_gain_is_not_rejected_for_KL_differences(self):
        # Policy has NO KL or NLL argument. Even a hypothetical huge
        # distribution difference cannot enter the decision.
        score=compare_answers(sample())
        result=assess(qa=score,generation=generation(),
            physical_bytes=11_229_303,native_speed_ratio=1.05,
            native_speed_is_full_model=False,generation_prompts=3)
        self.assertEqual(result["status"],"PROMISING_BUT_NOT_YET_PROVEN")
        self.assertNotIn("KL",str(result))
        self.assertNotIn("NLL",str(result))
        self.assertFalse(result["checks"]["enough_independent_VALID_questions"])
        self.assertFalse(result["checks"]["distinct_ratio_at_least_0p25"])

    def test_generating_random_nonsense_without_true_answers_is_not_an_improvement(self):
        score=compare_answers(sample())
        score["correct_answer_gain"]=-1
        result=assess(qa=score,generation=generation(.03,.5,.22),
            physical_bytes=11_229_303,native_speed_ratio=1.1,
            native_speed_is_full_model=False,generation_prompts=80,
            meaningful_quality_checked=False)
        self.assertEqual(result["status"],"REJECT_QUALITY_REGRESSION")
        self.assertFalse(result["checks"]["meaningful_coherent_response_quality_checked"])

    def test_no_promotion_without_full_generation_semantic_review_and_cpu(self):
        score=compare_answers(sample())
        result=assess(qa=score,generation=generation(.05,.55,.1),
            physical_bytes=11_229_303,native_speed_ratio=1.15,
            native_speed_is_full_model=False,generation_prompts=20)
        self.assertNotEqual(result["status"],"STAGE_FOR_HUMAN_QUALITY_REVIEW")
        self.assertFalse(result["automatic_live_changes_authorized"])

if __name__=="__main__":unittest.main()
