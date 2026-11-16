"""NRA3 S17b original frozen43M fast-tokenizer source-only offset tests."""
from __future__ import annotations

from dataclasses import replace
import unittest

from research.nra3_s17.new_ru24_directed_eval_prereg import (
    build_new_eval24,SPLIT,S17_EVAL24_COMPLETE_SHA,
)
from research.nra3_s17.locked_original43m_new24_token_alignment import (
    align_directed_new24,
)
from research.nra3_s4.test_russian_ud_train import (
    CharFastTokenizer,DuplicatingTemplateTokenizer,MergedWordsTokenizer
)


class Original43MNewDirectedWordpieceSource(unittest.TestCase):
    def test_all_24_true_new_russian_eval_wordpiece_offsets(self):
        tok=CharFastTokenizer()
        rows=build_new_eval24()
        self.assertEqual(len(rows),24)
        self.assertEqual(S17_EVAL24_COMPLETE_SHA,
          "fca54736aa555d9c5b3665d1bb9a1fc1979ad7abdeb083af015d90dc05ed1a32")
        self.assertTrue(any("," in x.text for x in rows))
        self.assertTrue(any(any(w.upos=="PRON" for w in x.words) for x in rows))
        for row in rows:
            aligned=align_directed_new24(tok,row)
            self.assertTrue(aligned.independent_eval_labels_loaded)
            self.assertEqual(aligned.original_prompt_sha256,row.text_sha256())
            self.assertEqual(len(aligned.word_alignment),len(row.words))
            self.assertEqual(aligned.complete_prefix_position,aligned.tokenizer_token_count-1)
            self.assertTrue(all(w.last_token>=0 for w in aligned.word_alignment))
            positions=[w.last_token for w in aligned.word_alignment]
            self.assertEqual(positions,sorted(positions))

    def test_refuses_changed_gold_text_id_or_tokenizer_merging_or_duplicate(self):
        row=build_new_eval24()[0]
        tok=CharFastTokenizer()
        good=align_directed_new24(tok,row)
        self.assertTrue(good.independent_eval_labels_loaded)
        with self.assertRaises(ValueError):
            align_directed_new24(tok,row,expected_input_ids=[1,2,3])
        with self.assertRaises(ValueError):
            align_directed_new24(DuplicatingTemplateTokenizer(),row)
        with self.assertRaises(ValueError):
            align_directed_new24(MergedWordsTokenizer(),row)
        with self.assertRaises(ValueError):
            align_directed_new24(tok,replace(row,split="TRAIN"))
        changed=replace(row.words[0],role="RECIPIENT")
        mutated=replace(row,words=(changed,)+row.words[1:])
        self.assertEqual(row.text_sha256(),mutated.text_sha256())
        with self.assertRaises(ValueError):
            align_directed_new24(tok,mutated)
        with self.assertRaises(ValueError):
            align_directed_new24(tok,replace(row,event_modality="FUTURE_UNREALIZED"))


if __name__=="__main__":
    unittest.main()
