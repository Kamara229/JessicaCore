"""Audit exact full expressions without promoting arbitrary gold mentions."""
import unittest
from v190.cases import cases
from v191.grammar import grade_structured

class PosthocNoGoldLeak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=cases()

    def check(self,i,answer,expected):
        row=self.rows[i]
        old={"strict_grade":{"machine_status":"ungraded","recognized_answer":None},
             "response":{"verbatim_text":answer}}
        return grade_structured(row,old)["machine_status"]==expected

    def test_equation_correct_wrong_and_rejected_gold_trap(self):
        self.assertTrue(self.check(0,"200 + 37 = 237","machine_correct"))
        self.assertTrue(self.check(0,"200 + 37 = 238","machine_wrong"))
        self.assertTrue(self.check(0,"999 + 37 = 237","ungraded"))
        self.assertTrue(self.check(0,"200 + 37 = 237, but maybe 239","ungraded"))
        self.assertTrue(self.check(8,"760 minus 213 equals 547.","machine_correct"))
        self.assertTrue(self.check(8,"760 - 213 = 633","machine_wrong"))

    def test_fictional_named_owner_changes_only_if_claim_unambiguous(self):
        self.assertTrue(self.check(32,"NeriDalo0's folder is purple.","machine_correct"))
        self.assertTrue(self.check(33,"NeriDalo0's folder is purple.","machine_wrong"))
        self.assertTrue(self.check(33,"NeriDalo0 owns the gray folder.","machine_correct"))
        self.assertTrue(self.check(33,"JumaTeso0 owns the purple folder.","ungraded"))
        self.assertTrue(self.check(33,"NeriDalo0's folder is gray, maybe purple.","ungraded"))

    def test_fact_anchored_and_absent_statement_is_ungraded(self):
        self.assertTrue(self.check(48,"The planet closest to the Sun is called Mercury.","machine_correct"))
        self.assertTrue(self.check(48,"The planet closest to the Sun is called Mars.","machine_wrong"))
        self.assertTrue(self.check(48,"Mercury is not the correct planet.","ungraded"))
        self.assertTrue(self.check(51,"A regular hexagon has 6 sides.","machine_correct"))
        self.assertTrue(self.check(51,"A regular hexagon has 5 sides.","machine_wrong"))

    def test_do_not_overwrite_precommitted_grade(self):
        case=self.rows[0]
        original={"strict_grade":{"machine_status":"machine_wrong","recognized_answer":"1"},
                  "response":{"verbatim_text":"200 + 37 = 237"}}
        grade=grade_structured(case,original)
        self.assertEqual(grade["machine_status"],"machine_wrong")
        self.assertFalse(grade["changed_from_original_v190"])

if __name__=="__main__":
    unittest.main()
