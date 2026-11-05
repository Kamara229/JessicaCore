"""Strict source audit: known 96 questions × 2 formats, no QA leakage."""
import unittest
from v169.sources import all_cases
from v188.run import precommit,paired_rows

class FrozenSourceCapability(unittest.TestCase):
    def test_frozen_genuine_historical_benchmark(self):
        cases=all_cases()
        frozen=precommit(cases)
        self.assertEqual(len(cases),192)
        self.assertEqual(frozen["total_actual_original_answers"],192)
        self.assertEqual(set(frozen["domain_counts"]),
                         {"arithmetic","binding","yesno","factual","logic"})
        self.assertEqual(sum(v["direct"] for v in frozen["domain_counts"].values()),96)
        self.assertEqual(sum(v["brief"] for v in frozen["domain_counts"].values()),96)
        self.assertEqual(len(set(row["sha256"] for row in
                        frozen["exact_prompt_hashes"])),192)
        self.assertTrue(frozen["private_v89_TEST_unread"])
        self.assertFalse(frozen["new_teacher_fit_or_quantizer_tuning"])

    def test_paired_matched_answer_scoring_distinctness_and_nonparse(self):
        records=[{
            "question":"x","style":"direct","domain":"yesno",
            "pair_id":"yn_0","machine_correct":True,
            "token_ids":[0],"verbatim_text":"Yes",
        },{
            "question":"x contradiction","style":"direct","domain":"yesno",
            "pair_id":"yn_0","machine_correct":True,
            "token_ids":[1],"verbatim_text":"No",
        }]
        actual=paired_rows(records,"direct","yesno")
        self.assertEqual(len(actual),1)
        self.assertTrue(actual[0]["correct_both"])
        self.assertTrue(actual[0]["distinct_first_tokens"])
        records[1]["machine_correct"]=None
        self.assertFalse(paired_rows(records,"direct","yesno")[0]["correct_both"])
        self.assertFalse(paired_rows(records,"direct","yesno")[0]["both_are_parseable"])

    def test_incomplete_source_prompts_rejected(self):
        with self.assertRaises(AssertionError):
            precommit(all_cases()[:96])

if __name__=="__main__":
    unittest.main()
