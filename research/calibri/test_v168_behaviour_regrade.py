"""CALIBRI v168 audit: no inaccurate 'FP32 0/48' claim ever again."""
import unittest
from v168.review import (validate_annotations,KNOWN_CORRECT_FP32,
                        KNOWN_INCORRECT_FP32,FREE_FACTUAL_RANGE)

class ReviewControl(unittest.TestCase):
    def test_human_graded_source_counts(self):
        self.assertTrue(validate_annotations())
        self.assertEqual(len(KNOWN_CORRECT_FP32),30)
        self.assertEqual(len(KNOWN_INCORRECT_FP32),10)
        self.assertEqual(len(FREE_FACTUAL_RANGE),40)

    def test_false_exact_string_criterion_is_fixed(self):
        self.assertIn(0,KNOWN_CORRECT_FP32) # '7 + 5 = 12.'
        self.assertIn(8,KNOWN_CORRECT_FP32) # 'The capital of France is Paris.'
        self.assertIn(13,KNOWN_CORRECT_FP32) # 'three sides', not strict '3'
        self.assertIn(21,KNOWN_INCORRECT_FP32) # inconsistent badge color
        self.assertIn(6,KNOWN_INCORRECT_FP32) # '42+7=59' wrong

    def test_do_not_grade_free_creative_and_instruction_as_wrong(self):
        self.assertTrue(all(i not in KNOWN_CORRECT_FP32 and
                            i not in KNOWN_INCORRECT_FP32 for i in range(40,48)))

if __name__=="__main__":
    unittest.main()
