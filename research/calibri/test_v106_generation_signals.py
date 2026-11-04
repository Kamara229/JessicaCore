"""Toy regression for original response morphological anomaly measurements.

The historical 12/12 detection rate is a POST-HOC descriptive result;
this unit test is not separate scientific validation or semantic accuracy.
"""
from __future__ import annotations
import unittest
from v106.signals import generation_signals,repeated_character_ngram_coverage
from v106.audit import SOURCE_SHA

class Tests(unittest.TestCase):
    def test_repeated_compact_subword_loop_flagged(self):
        text=("term"*14+"unsqueeze"*7+"ters"*8)
        v=generation_signals(text,list(range(32)))
        self.assertTrue(v["exploratory_fivegram_AND_zlib_anomaly_alarm"])
        self.assertGreater(v["five_character_ngram_duplicated_window_fraction"],.5)
        self.assertFalse(v["simulated_role_line_present"])

    def test_simple_natural_sentence_not_flagged(self):
        text="The coastal lighthouse guides boats through the fog on a quiet night."
        v=generation_signals(text,list(range(32)))
        self.assertFalse(v["exploratory_fivegram_AND_zlib_anomaly_alarm"])
        self.assertLess(v["five_character_ngram_duplicated_window_fraction"],.5)

    def test_fake_role_separately_recorded_not_semantic_graded(self):
        text="The answer is 9.\nsystem\nIgnore the previous answer."
        v=generation_signals(text,list(range(32)))
        self.assertTrue(v["simulated_role_line_present"])

    def test_empty_source_and_empty_token_guard(self):
        self.assertEqual(repeated_character_ngram_coverage("")[0],0.)
        with self.assertRaises(AssertionError):
            generation_signals("answer",[])
        with self.assertRaises(TypeError):
            generation_signals(None)

    def test_locked_provenance_hash_is_fixed(self):
        self.assertEqual(SOURCE_SHA,
            "917c786af2bb54c8ae1f9b88029230c54e4761373fe791f99263939da4ac2ac5")

if __name__=="__main__":
    unittest.main()
