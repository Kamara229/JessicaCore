"""Check original source audit joins are exact, not fabricated or shuffled."""
import json,os,copy,unittest
from pathlib import Path
from v169.sources import all_cases
from v189.run import source_integrity,score_both
from v189 import protocol as p

class GenuineOriginalSourceComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(os.environ.get("V188_SOURCE",
            "inputs/v188/v188_actual_original_FP32_192QA_complete.json"))
        cls.data=json.loads(path.read_text())
        cls.cases=all_cases()

    def test_all_192_original_prompt_hashes_and_source_revision(self):
        result=source_integrity(self.data,self.cases)
        self.assertTrue(result["no_historical_question_or_style_drift"])
        self.assertEqual(result["verified_old_original_135m_questions"],192)
        self.assertEqual(result["old_fp32_unique_parameters"],134515008)
        self.assertEqual(p.EXPECTED_GENERATIONS,192)

    def test_tampered_or_reordered_previous_raw_answer_rejected(self):
        corruption=copy.deepcopy(self.data)
        corruption["actual_original_FP32_192_answers"][0]["question_id"]=12
        with self.assertRaises(AssertionError):
            source_integrity(corruption,self.cases)
        corruption=copy.deepcopy(self.data)
        corruption["actual_original_FP32_192_answers"].reverse()
        with self.assertRaises(AssertionError):
            source_integrity(corruption,self.cases)

    def test_no_ungraded_equals_false_and_exact_original_roundtrip(self):
        raw=self.data["actual_original_FP32_192_answers"]
        evaluation=score_both(raw,raw)
        self.assertEqual(evaluation["old"]["all"]["machine_correct"],66)
        self.assertEqual(evaluation["new"]["all"]["machine_correct"],66)
        self.assertEqual(evaluation["old"]["all"]["ungraded"],57)
        self.assertEqual(evaluation["new"]["all"]["ungraded"],57)
        for style in p.STYLE_CHOICES:
            self.assertEqual(evaluation["paired_summary"][style]["new_minus_old_confirmed"],0)
            self.assertEqual(evaluation["paired_summary"][style]["new_correct_old_not"],0)
            self.assertEqual(evaluation["paired_summary"][style]["old_correct_new_not"],0)

if __name__=="__main__":
    unittest.main()
