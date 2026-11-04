"""No change to v106 morphology rules or old v96 original output."""
from __future__ import annotations
import unittest
from v108.spec import GOLD_PROMPTS,OPEN_PROMPTS,ALL_PROMPTS,GENERATED_TOKENS
from v108.generation import build_chat
from v106.signals import generation_signals,DUPLICATE_COVERAGE_THRESHOLD,COMPRESSED_SIZE_RATIO_THRESHOLD

class MockTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        if tokenize or not add_generation_prompt:
            raise AssertionError("test requires original generation chat frame")
        return "<s><user>"+messages[0]["content"]+"</user><assistant>"

class Tests(unittest.TestCase):
    def test_new_original_prompt_set_is_frozen_disjoint_and_objectively_gold(self):
        self.assertEqual((len(GOLD_PROMPTS),len(OPEN_PROMPTS),len(ALL_PROMPTS)),(6,3,9))
        self.assertEqual(len({t[1] for t in ALL_PROMPTS}),9)
        self.assertEqual(GENERATED_TOKENS,32)
        self.assertEqual([x[2][0] for x in GOLD_PROMPTS],
                         ["31","23","36","Madrid","Athens","Mercury"])
    def test_exact_real_user_question_retained_with_assistant_open_frame(self):
        s=build_chat(MockTokenizer(),"What is 9 times 4?")
        self.assertIn("What is 9 times 4?",s)
        self.assertTrue(s.endswith("<assistant>"))
    def test_frozen_historical_detector_unchanged(self):
        self.assertEqual(DUPLICATE_COVERAGE_THRESHOLD,.50)
        self.assertEqual(COMPRESSED_SIZE_RATIO_THRESHOLD,.60)
        repeated=generation_signals("tempfile "*22,list(range(32)))
        self.assertTrue(repeated["exploratory_fivegram_AND_zlib_anomaly_alarm"])
        fluent=generation_signals(
            "The mountain train reached the valley station at dawn.",
            list(range(32)))
        self.assertFalse(fluent["exploratory_fivegram_AND_zlib_anomaly_alarm"])
    def test_known_truths_not_automatically_teacher_labels(self):
        self.assertEqual(GOLD_PROMPTS[0][2],("31",))
        self.assertNotIn("42",GOLD_PROMPTS[0][2])
if __name__=="__main__":unittest.main()
