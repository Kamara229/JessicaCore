"""v191 only standard-library adversarial tests; never load original models."""
import unittest
from v191.grammar import grade_structured

class PosthocNoGoldLeak(unittest.TestCase):
    def case(self,domain,question,gold,case_id=0):
        return {"case_id":case_id,"domain":domain,"question":question,"gold":gold}

    def check(self,case,answer,expected):
        original={"strict_grade":{"machine_status":"ungraded","recognized_answer":None},
                  "response":{"verbatim_text":answer}}
        return grade_structured(case,original)["machine_status"]==expected

    def test_equation_correct_wrong_and_rejected_gold_trap(self):
        a=self.case("arithmetic","Compute 200 plus 37. Reply with the result as one number.","237")
        b=self.case("arithmetic","Compute 760 minus 213. Reply with the result as one number.","547")
        self.assertTrue(self.check(a,"200 + 37 = 237","machine_correct"))
        self.assertTrue(self.check(a,"200 + 37 = 238","machine_wrong"))
        self.assertTrue(self.check(a,"999 + 37 = 237","ungraded"))
        self.assertTrue(self.check(a,"200 + 37 = 237, but maybe 239","ungraded"))
        self.assertTrue(self.check(b,"760 minus 213 equals 547.","machine_correct"))
        self.assertTrue(self.check(b,"760 - 213 = 633","machine_wrong"))

    def test_fictional_named_owner_changes_only_if_claim_unambiguous(self):
        a=self.case("fictional_binding",
           "In a fictional workshop, NeriDalo0 owns the purple folder and JumaTeso0 owns the gray folder. What color is NeriDalo0's folder? Reply with one color.","purple")
        b=self.case("fictional_binding",
           "In a fictional workshop, NeriDalo0 owns the gray folder and JumaTeso0 owns the purple folder. What color is NeriDalo0's folder? Reply with one color.","gray")
        self.assertTrue(self.check(a,"NeriDalo0's folder is purple.","machine_correct"))
        self.assertTrue(self.check(b,"NeriDalo0's folder is purple.","machine_wrong"))
        self.assertTrue(self.check(b,"NeriDalo0 owns the gray folder.","machine_correct"))
        self.assertTrue(self.check(b,"JumaTeso0 owns the purple folder.","ungraded"))
        self.assertTrue(self.check(b,"NeriDalo0's folder is gray, maybe purple.","ungraded"))

    def test_fact_anchored_and_absent_statement_is_ungraded(self):
        planet=self.case("factual_short",
          "Which planet is closest to the Sun? Answer with only its name.","Mercury")
        sides=self.case("factual_short",
          "How many sides does a regular hexagon have? Answer with one number.","6")
        self.assertTrue(self.check(planet,"The planet closest to the Sun is called Mercury.","machine_correct"))
        self.assertTrue(self.check(planet,"The planet closest to the Sun is called Mars.","machine_wrong"))
        self.assertTrue(self.check(planet,"Mercury is not the correct planet.","ungraded"))
        self.assertTrue(self.check(sides,"A regular hexagon has 6 sides.","machine_correct"))
        self.assertTrue(self.check(sides,"A regular hexagon has 5 sides.","machine_wrong"))

    def test_do_not_overwrite_precommitted_grade(self):
        case=self.case("arithmetic",
          "Compute 200 plus 37. Reply with the result as one number.","237")
        original={"strict_grade":{"machine_status":"machine_wrong","recognized_answer":"1"},
                  "response":{"verbatim_text":"200 + 37 = 237"}}
        grade=grade_structured(case,original)
        self.assertEqual(grade["machine_status"],"machine_wrong")
        self.assertFalse(grade["changed_from_original_v190"])

if __name__=="__main__":
    unittest.main()
