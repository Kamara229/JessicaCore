"""Pure-standard-library S27 audit invariants. No models, weights, or QA tuning."""
import unittest
from calibri2.s27.posthoc_frozen_answer_audit import (
    NOVEL,DOMAINS,VALID_TWO,exact_paired_p_value,binomial_wilson,
    audit_rows
)

class AuditTests(unittest.TestCase):
    def record(self,id_,domain,correct=False,color="red",noun="fork"):
        return {
            "id":id_,"domain":domain,"question":"FAKE_TEST_ONLY_"+id_,
            "gold":color+" "+noun if domain==NOVEL else "Yes",
            "expected_token_ids_full_EOS":[123,456,2],
            "actual_token_ids":[123,456,2] if correct else [444,555,2],
            "actual_text":color+" "+noun if correct else "red bag",
            "strict_full_EOS":correct,"correct_first_token":correct,
            "EOS_emitted":True
        }
    def three(self):
        before=[];after=[];public=[]
        for dom,count in DOMAINS.items():
            for i in range(count):
                id_=dom+"_"+str(i)
                before.append(self.record(id_,dom,i==0))
                after.append(self.record(id_,dom,i in (0,1)))
                public.append(self.record(id_,dom,i in (0,1,2)))
        openq=[{"question":"q"+str(i),"actual_answer":"answer"} for i in range(16)]
        return ({"every_144_actual_answer_and_gold":before,
                 "every_16_open_actual_answer":openq},
                {"every_144_actual_answer_and_gold":after,
                 "every_16_open_actual_answer":openq},
                {"every_144_actual_answer_and_gold":public,
                 "every_16_open_actual_answer":openq})
    def test_valid_two_word_requires_lexical_token(self):
        self.assertIsNotNone(VALID_TWO.fullmatch("red fork"))
        self.assertIsNotNone(VALID_TWO.fullmatch(" blue glove."))
        self.assertIsNone(VALID_TWO.fullmatch("red nonsense"))
        self.assertIsNone(VALID_TWO.fullmatch("red fork green"))
        self.assertIsNone(VALID_TWO.fullmatch("yes no"))
    def test_exact_paired_significance(self):
        self.assertEqual(exact_paired_p_value([True,False],[True,True]),
                         {"gained":1,"lost":0,"p_two_sided":1.0})
        self.assertEqual(exact_paired_p_value([True,False],[True,False]),
                         {"gained":0,"lost":0,"p_two_sided":1.0})
        self.assertEqual(exact_paired_p_value([False]*6,[True]*6)["p_two_sided"],.03125)
    def test_audit_preserves_strict_full_and_controls(self):
        r=audit_rows(*self.three())
        self.assertEqual(r[NOVEL]["S24_complete_correct"],1)
        self.assertEqual(r[NOVEL]["S26_complete_correct"],2)
        self.assertEqual(r[NOVEL]["PUBLIC_UNMATCHED_135M_FP32_complete_correct"],3)
        self.assertEqual(len(r["open_response_pairs"]),16)
        self.assertEqual(r["color_one_word_control"]["S26_complete_correct"],2)
        self.assertEqual(r["number_one_word_control"]["S26_complete_correct"],2)
        self.assertTrue(0 <= binomial_wilson(0,96)[1] < .05)
    def test_modified_frozen_gold_rejected(self):
        a,b,c=self.three()
        b["every_144_actual_answer_and_gold"][0]["gold"]="blue fork"
        with self.assertRaisesRegex(AssertionError,"QA content"):
            audit_rows(a,b,c)
    def test_missing_case_rejected(self):
        a,b,c=self.three()
        b["every_144_actual_answer_and_gold"].pop()
        with self.assertRaisesRegex(AssertionError,"144 archived rows"):
            audit_rows(a,b,c)

if __name__=="__main__":
    unittest.main()
