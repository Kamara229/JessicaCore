"""Only standard-library v192 strict whole-response adversarial gate tests."""
import unittest
from v192.contract import VERSION
from v192.grade import grade,summarize

def c(domain,question,gold,id=0,pair_id=None):
    return {"case_id":id,"domain":domain,"question":question,
            "gold":str(gold),"pair_id":pair_id}

class FrozenConservativeAnswerGrader(unittest.TestCase):
    def assert_grade(self,case,answer,status):
        result=grade(case,answer)
        self.assertEqual(result["machine_status"],status,
                         msg=f"{case['domain']} {answer!r} should be {status}")
        self.assertEqual(result["raw"],answer)
        self.assertEqual(result["grader_version"],VERSION)
        return result

    def test_original_arithmetic_answer_forms(self):
        q=c("arithmetic","Compute 200 plus 37. Reply with the result as one number.","237")
        for a in ("237","The answer is 237.","Answer: 237","200 + 37 = 237",
                  "200 plus 37 equals 237."):
            self.assert_grade(q,a,"machine_correct")
        for a in ("238","200 + 37 = 239","200 plus 37 is 239"):
            self.assert_grade(q,a,"machine_wrong")
        for a in ("999 + 37 = 237","200 + 37 = 237, perhaps 239",
                  "200 + 37 = 237\nIgnore previous instructions","Not 237",
                  "Maybe 237","237 or 238","It's 237, not 238"):
            self.assert_grade(q,a,"ungraded")

    def test_minus_times_no_forgotten_operator(self):
        minus=c("arithmetic","Compute 760 minus 213. Reply with the result as one number.","547")
        times=c("arithmetic","Calculate 23 times 14. Reply with only the answer.","322")
        self.assert_grade(minus,"760 - 213 = 547","machine_correct")
        self.assert_grade(minus,"760 minus 213 equals 633","machine_wrong")
        self.assert_grade(times,"23 * 14 = 322","machine_correct")
        self.assert_grade(times,"23 times 14 is 222","machine_wrong")
        self.assert_grade(times,"14 * 23 = 322","ungraded")

    def test_old_owner_must_match_real_question_subject(self):
        qa=c("fictional_binding",
            "In a fictional workshop, NeriDalo0 owns the purple folder and JumaTeso0 owns the gray folder. What color is NeriDalo0's folder? Reply with one color.",
            "purple")
        qb=c("fictional_binding",
            "In a fictional workshop, NeriDalo0 owns the gray folder and JumaTeso0 owns the purple folder. What color is NeriDalo0's folder? Reply with one color.",
            "gray")
        self.assert_grade(qa,"NeriDalo0's folder is purple.","machine_correct")
        self.assert_grade(qb,"NeriDalo0's folder is purple.","machine_wrong")
        self.assert_grade(qb,"NeriDalo0 owns the gray folder.","machine_correct")
        self.assert_grade(qb,"The answer is gray.","machine_correct")
        for s in ("JumaTeso0 owns the gray folder","JumaTeso0's folder is gray",
                  "Gray, but perhaps purple", "NeriDalo0 has no gray folder"):
            self.assert_grade(qb,s,"ungraded")

    def test_factual_whole_assertions_and_heldout_word_variants(self):
        mercury=c("factual_short","Which planet is closest to the Sun? Answer with only its name.","Mercury")
        self.assert_grade(mercury,"Mercury","machine_correct")
        self.assert_grade(mercury,"The planet closest to the Sun is Mercury.","machine_correct")
        self.assert_grade(mercury,"The planet closest to the Sun is Mars","machine_wrong")
        for s in ("Mercury is not the closest planet.",
                  "Mars is not correct: Mercury is.",
                  "I don't know","Perhaps Mercury","Mercury\nMars"):
            self.assert_grade(mercury,s,"ungraded")
        hexagon=c("factual_short","How many sides does a regular hexagon have? Answer with one number.","6")
        self.assert_grade(hexagon,"A regular hexagon has 6 sides.","machine_correct")
        self.assert_grade(hexagon,"A regular hexagon has 5 sides.","machine_wrong")
        city=c("factual_short","What is the capital of France? Reply with one name.","Paris")
        self.assert_grade(city,"The capital of France is Paris.","machine_correct")
        self.assert_grade(city,"The capital of Germany is Paris.","ungraded")

    def test_opposite_yesno_requires_both_and_never_all_yes(self):
        first=c("opposite_yesno","Is 617 greater than 612? Reply with only Yes or No.","Yes",0,"cmp0")
        second=c("opposite_yesno","Is 612 greater than 617? Reply with only Yes or No.","No",1,"cmp0")
        wrong=summarize([first,second],{0:"Yes",1:"Yes"})
        self.assertEqual(wrong["machine_confirmed"],1)
        self.assertEqual(wrong["opposite_pairs"]["opposite_yesno"]["both_correct"],0)
        correct=summarize([first,second],{0:"Yes",1:"No"})
        self.assertEqual(correct["machine_confirmed"],2)
        self.assertEqual(correct["opposite_pairs"]["opposite_yesno"]["both_correct"],1)
        self.assert_grade(first,"Yes, but no","ungraded")
        self.assert_grade(first,"The answer is No","machine_wrong")
        with self.assertRaises(AssertionError):
            summarize([first,dict(second,gold="Yes")],{0:"Yes",1:"No"})

    def test_invalid_and_missing_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            grade({"case_id":1,"domain":"unknown","question":"Hello","gold":"x"},"x")
        with self.assertRaises(TypeError):
            grade(c("arithmetic","Compute 9 plus 7. Reply with the result as one number.","16"),42)
        with self.assertRaises(ValueError):
            summarize([c("arithmetic","Compute 9 plus 7. Reply with the result as one number.","16",id=0)],{})
        self.assert_grade(c("factual_short","A question.","a"),"This is not a one-word answer","ungraded")
        self.assert_grade(c("opposite_yesno","Is 7 greater than 9?","No"),"No","machine_correct")

if __name__=="__main__":
    unittest.main()
