"""S18b0 strict 96 TRAIN-only original-tokenizer offset preflight tests.

Mock tokenizer here validates contract only, not physical original43M CPU.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

from research.nra3_s4.test_russian_ud_train import (
    CharFastTokenizer, DuplicatingTemplateTokenizer, MergedWordsTokenizer,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    S18_SPLIT, S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
    build_s18_new_train96, seal_s18_new96_train,
)
from research.nra3_s18.locked_original43m_new96_token_alignment import (
    align_s18_new96_train,
)


class LockedNew96OriginalTokenAlignment(unittest.TestCase):
    def test_all_new96_source_only_full_sha_and_offsets(self):
        rows = build_s18_new_train96()
        manifest = seal_s18_new96_train()
        self.assertEqual(len(rows), 96)
        self.assertEqual(
            manifest["S18_complete_original_96_new_TRAIN_full_annotation_SHA256"],
            S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
        )
        self.assertEqual(manifest["S18_new_directed_verb_to_argument_links"], 300)
        tok = CharFastTokenizer()
        for row in rows:
            self.assertEqual(row.split, S18_SPLIT)
            aligned = align_s18_new96_train(tok, row)
            self.assertEqual(aligned.original_prompt_sha256, row.text_sha256())
            self.assertFalse(aligned.independent_eval_labels_loaded)
            self.assertEqual(len(aligned.word_alignment), len(row.words))
            self.assertTrue(all(
                z.last_token >= 0 and z.last_token < aligned.tokenizer_token_count
                for z in aligned.word_alignment
            ))
            self.assertEqual(
                aligned.complete_prefix_position, aligned.tokenizer_token_count - 1
            )
            self.assertEqual(
                [z.last_token for z in aligned.word_alignment],
                sorted(z.last_token for z in aligned.word_alignment),
            )

    def test_reject_wrong_token_ids_merged_words_and_duplicate_prompt(self):
        row = build_s18_new_train96()[0]
        tok = CharFastTokenizer()
        aligned = align_s18_new96_train(tok, row)
        self.assertFalse(aligned.independent_eval_labels_loaded)
        self.assertEqual(
            align_s18_new96_train(
                tok, row, expected_input_ids=aligned.prefill_token_ids
            ), aligned
        )
        with self.assertRaises(ValueError):
            align_s18_new96_train(tok, row, expected_input_ids=[1, 2, 3])
        with self.assertRaises(ValueError):
            align_s18_new96_train(DuplicatingTemplateTokenizer(), row)
        with self.assertRaises(ValueError):
            align_s18_new96_train(MergedWordsTokenizer(), row)
        with self.assertRaises(ValueError):
            align_s18_new96_train(None, row)

    def test_reject_gold_and_lemma_tampering_without_text_sha_change(self):
        row = build_s18_new_train96()[0]
        tok = CharFastTokenizer()
        with self.assertRaises(ValueError):
            align_s18_new96_train(tok, replace(row, split="EVAL"))
        with self.assertRaises(ValueError):
            align_s18_new96_train(tok, replace(row, event_modality="NEGATED"))
        changed = replace(row.words[0], semantic_role="SOURCE")
        tampered = replace(row, words=(changed,) + row.words[1:])
        self.assertEqual(tampered.text_sha256(), row.text_sha256())
        with self.assertRaises(ValueError):
            align_s18_new96_train(tok, tampered)
        changed = replace(row.words[0], lemma="fake")
        tampered = replace(row, words=(changed,) + row.words[1:])
        self.assertEqual(tampered.text_sha256(), row.text_sha256())
        with self.assertRaises(ValueError):
            align_s18_new96_train(tok, tampered)

    def test_reject_unregistered_raw_prompt_and_word_offsets(self):
        row = build_s18_new_train96()[3]
        tok = CharFastTokenizer()
        with self.assertRaises(ValueError):
            align_s18_new96_train(tok, replace(row, text=row.text + " extra"))
        altered_word = replace(row.words[0], start=row.words[0].start + 1)
        with self.assertRaises(ValueError):
            align_s18_new96_train(
                tok, replace(row, words=(altered_word,) + row.words[1:])
            )


if __name__ == "__main__":
    unittest.main()
