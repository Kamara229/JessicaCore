"""S15b original fast Calibri tokenizer exact SHA pinned 24 EVAL-only words."""
from __future__ import annotations

from dataclasses import replace
import unittest

from research.nra3_s15.new_ru_eval24_prereg import (
    new_russian_eval24,SPLIT,EVAL24_CANONICAL_SHA,
)
from research.nra3_s15.locked_real_eval24_token_alignment import align_heldout24
from research.nra3_s4.test_russian_ud_train import (
    CharFastTokenizer,DuplicatingTemplateTokenizer,MergedWordsTokenizer,
)


class FrozenOriginalCalibriRealEval24Alignment(unittest.TestCase):
    def test_all_24_true_new_russian_eval_wordpiece_offsets(self):
        tok=CharFastTokenizer()
        rows=new_russian_eval24()
        self.assertEqual(len(rows),24)
        self.assertEqual(EVAL24_CANONICAL_SHA,
          "640bdfb0d817d1467eaf308fc1ba5dd9043361721af293b9666d89f4a2c7276d")
        self.assertGreaterEqual(sum(bool(x.antecedents) for x in rows),4)
        self.assertTrue(any("," in x.text for x in rows))
        for row in rows:
            aligned=align_heldout24(tok,row)
            self.assertTrue(aligned.independent_eval_labels_loaded)
            self.assertEqual(aligned.original_prompt_sha256,row.sha256())
            self.assertEqual(len(aligned.word_alignment),len(row.words))
            self.assertEqual(aligned.complete_prefix_position,aligned.tokenizer_token_count-1)
            self.assertTrue(all(x.last_token>=0 for x in aligned.word_alignment))
            positions=[w.last_token for w in aligned.word_alignment]
            self.assertEqual(positions,sorted(positions))

    def test_rejects_unregistered_eval24_gold_changes_future_and_bad_tokenizer(self):
        row=new_russian_eval24()[0]
        tok=CharFastTokenizer()
        good=align_heldout24(tok,row)
        self.assertTrue(good.independent_eval_labels_loaded)
        with self.assertRaises(ValueError):
            align_heldout24(tok,row,expected_input_ids=[1,2,3])
        with self.assertRaises(ValueError):
            align_heldout24(DuplicatingTemplateTokenizer(),row)
        with self.assertRaises(ValueError):
            align_heldout24(MergedWordsTokenizer(),row)
        with self.assertRaises(ValueError):
            align_heldout24(tok,replace(row,split="TRAIN"))
        changed=replace(row,roles=("NONE",)+row.roles[1:])
        self.assertEqual(row.sha256(),changed.sha256())
        with self.assertRaises(ValueError):
            align_heldout24(tok,changed)
        with self.assertRaises(ValueError):
            align_heldout24(tok,replace(row,event_truth="NEGATED"))
        with self.assertRaises(ValueError):
            align_heldout24(tok,replace(row,construction="from_eVAL16"))


if __name__=="__main__":
    unittest.main()
