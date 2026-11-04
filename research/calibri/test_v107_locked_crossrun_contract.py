"""Read-only validation source guards; no dynamic classifier thresholds."""
from __future__ import annotations
import unittest
import json
from v106.signals import generation_signals
from v107.contract import SOURCE_FILES
from v107.crossrun import verify

class Tests(unittest.TestCase):
    def test_exact_two_previous_runs_are_pinned_and_disjoint(self):
        self.assertEqual(set(SOURCE_FILES),{"v103","v104"})
        self.assertNotEqual(SOURCE_FILES["v103"]["sha256"],SOURCE_FILES["v104"]["sha256"])
        self.assertEqual({v["expected_prompts"] for v in SOURCE_FILES.values()},{2,4})
    def test_genuine_word_repetition_detected_without_old_token_repeat_threshold(self):
        x=generation_signals(" constell"*30,list(range(30)))
        self.assertTrue(x["exploratory_fivegram_AND_zlib_anomaly_alarm"])
        self.assertGreater(x["unique_token_fraction_observed"],.90)
    def test_fluent_wrong_response_not_confused_with_char_level_gibberish(self):
        r=generation_signals("The capital of Switzerland is Geneva.\nsystem\nuser",list(range(18)))
        self.assertFalse(r["exploratory_fivegram_AND_zlib_anomaly_alarm"])
        self.assertTrue(r["simulated_role_line_present"])
    def test_wrong_source_sha_fails_before_answer_scoring(self):
        with self.assertRaises(AssertionError):
            verify(json.dumps({"experiment":"fake"}).encode("utf-8"),"v103")
    def test_source_unknown_key_cannot_bypass_sha(self):
        with self.assertRaises(KeyError):
            verify(b"{}", "NOT_A_SOURCE")
if __name__=="__main__":
    unittest.main()
