"""S9 new real-tokenizer interface contracts; mock offsets ONLY, no 43M."""
from __future__ import annotations
import dataclasses
import unittest

from research.nra3_s9.new_ru_train72 import build_s9_train72
from research.nra3_s9.locked_real_token_alignment import align_new_train72
from research.nra3_s4.test_russian_ud_train import (
    CharFastTokenizer,DuplicatingTemplateTokenizer,MergedWordsTokenizer,
)


class Train72StrictAlignment(unittest.TestCase):
    def test_all_72_new_training_sentences_can_map_exact_codepoint_tokens(self):
        tokenizer=CharFastTokenizer()
        for row in build_s9_train72():
            aligned=align_new_train72(tokenizer,row)
            self.assertEqual(aligned.original_prompt_sha256,row.sha256())
            self.assertEqual(len(aligned.word_alignment),len(row.words))
            self.assertEqual(aligned.complete_prefix_position,
                             aligned.tokenizer_token_count-1)
            self.assertGreater(aligned.tokenizer_token_count,len(row.words))
            self.assertTrue(all(len(w.token_positions)>=1
                                for w in aligned.word_alignment))
            self.assertFalse(aligned.independent_eval_labels_loaded)

    def test_reject_corrupted_gold_even_if_same_prompt_and_valid_token_ids(self):
        row=build_s9_train72()[0]
        tok=CharFastTokenizer()
        aligned=align_new_train72(tok,row)
        with self.assertRaises(ValueError):
            align_new_train72(tok,row,expected_input_ids=[1]*9)
        amended=dataclasses.replace(row,roles=("NONE",)+row.roles[1:])
        self.assertEqual(amended.sha256(),row.sha256())
        with self.assertRaises(ValueError):
            align_new_train72(tok,amended)
        with self.assertRaises(ValueError):
            align_new_train72(DuplicatingTemplateTokenizer(),row)
        with self.assertRaises(ValueError):
            align_new_train72(MergedWordsTokenizer(),row)
        with self.assertRaises(ValueError):
            align_new_train72(tok,dataclasses.replace(row,split="EVAL"))
        self.assertTrue(aligned.tokenizer_token_count>0)


if __name__=="__main__":
    unittest.main()
