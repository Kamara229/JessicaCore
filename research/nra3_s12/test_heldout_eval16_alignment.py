"""S12b actual fast-tokenizer EVAL-only sealed source contract, no Calibri call."""
from __future__ import annotations

from dataclasses import replace
import unittest

from research.nra3_s12.heldout_ru_eval16 import (
    build_eval16,SPLIT,EVAL16_FULL_CANONICAL_SHA,
)
from research.nra3_s12.locked_real_eval16_token_alignment import align_heldout16
from research.nra3_s4.test_russian_ud_train import (
    CharFastTokenizer,DuplicatingTemplateTokenizer,MergedWordsTokenizer,
)


class LockedRealEval16Alignment(unittest.TestCase):
    def test_all_16_true_eval_source_words_have_exact_chat_offsets(self):
        tok=CharFastTokenizer()
        rows=build_eval16()
        self.assertEqual(len(rows),16)
        self.assertEqual(len(EVAL16_FULL_CANONICAL_SHA),64)
        for row in rows:
            a=align_heldout16(tok,row)
            self.assertTrue(a.independent_eval_labels_loaded)
            self.assertEqual(a.original_prompt_sha256,row.sha256())
            self.assertEqual(len(a.word_alignment),len(row.words))
            self.assertEqual(a.complete_prefix_position,a.tokenizer_token_count-1)
            self.assertTrue(all(x.last_token>=0 for x in a.word_alignment))

    def test_refuses_sealed_eval_gold_edits_and_mixed_BPE_tokens(self):
        row=build_eval16()[0]
        tok=CharFastTokenizer()
        good=align_heldout16(tok,row)
        self.assertTrue(good.independent_eval_labels_loaded)
        with self.assertRaises(ValueError):
            align_heldout16(tok,row,expected_input_ids=[1,2])
        with self.assertRaises(ValueError):
            align_heldout16(DuplicatingTemplateTokenizer(),row)
        with self.assertRaises(ValueError):
            align_heldout16(MergedWordsTokenizer(),row)
        with self.assertRaises(ValueError):
            align_heldout16(tok,replace(row,split="NRA3_S9_TRAIN"))
        edited=replace(row,roles=("NONE",)+row.roles[1:])
        self.assertEqual(edited.sha256(),row.sha256())
        with self.assertRaises(ValueError):
            align_heldout16(tok,edited)


if __name__=="__main__":
    unittest.main()
