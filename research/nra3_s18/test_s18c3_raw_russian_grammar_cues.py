"""S18c3 raw-only grammar probes: no labels, no official UPOS or target roles."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    FEATURES, DIM, raw_surface_grammar, feature_at,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    build_s18_new_train96, S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
    seal_s18_new96_train,
)


def flagged(tokens, form, name):
    matching=[w for w in tokens if w.form.casefold()==form.casefold()]
    if len(matching)!=1:
        raise AssertionError("expected only one raw word token: " + form)
    return feature_at(matching[0],name)


class RawOnlyRussianGrammarCueContract(unittest.TestCase):

    def test_no_ud_roles_pos_lemma_or_gold_as_parser_input(self):
        module=Path("research/nra3_s18/s18c3_raw_russian_grammar_cues.py").read_text()
        tree=ast.parse(module)
        banned={"words","upos","deprel","lemma","semantic_role","lexical_family",
                "construction","gold","head"}
        used={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        self.assertTrue(used.isdisjoint(banned))
        self.assertEqual(DIM,18)
        self.assertEqual(len(FEATURES),len(set(FEATURES)))

    def test_all_96_train_raw_unicode_offsets_do_not_require_gold_input(self):
        rows=build_s18_new_train96()
        self.assertEqual(len(rows),96)
        self.assertEqual(seal_s18_new96_train()[
            "S18_complete_original_96_new_TRAIN_full_annotation_SHA256"
        ],S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256)
        for row in rows:
            raw=raw_surface_grammar(row.text)
            self.assertEqual([w.form for w in raw],[w.form for w in row.words])
            self.assertEqual([(w.start,w.end) for w in raw],
                             [(w.start,w.end) for w in row.words])
            self.assertEqual(len(raw),len(row.words))
            self.assertTrue(all(len(w.features)==DIM for w in raw))
            # Source row GOLD used only in this TEST for offset validation.
            # The raw_surface_grammar() INPUT was row.text ONLY.

    def test_short_passive_and_instrumental_detected_without_gold(self):
        samples=("Схема была выслана Антоном Валерии.",
                 "Книга была вручена преподавателем ученице.")
        for text,aux,passive,inst in (
            (samples[0],"была","выслана","Антоном"),
            (samples[1],"была","вручена","преподавателем"),
        ):
            raw=raw_surface_grammar(text)
            self.assertEqual(flagged(raw,aux,"past_auxiliary_byt"),1)
            self.assertEqual(flagged(raw,passive,
                                    "short_passive_participle_shape_possible"),1)
            self.assertEqual(flagged(raw,passive,"previous_word_past_auxiliary"),1)
            self.assertEqual(flagged(raw,inst,"ending_instrumental_possible"),1)
            self.assertEqual(flagged(raw,inst,"previous_word_is_ot"),0)

    def test_recipient_subject_and_source_ot_remain_raw_grammar_indicators(self):
        for text,recipient,verb,prep,source in (
            ("Валерия получила схему от Антона.",
             "Валерия","получила","от","Антона"),
            ("Мария получила письмо от врача.",
             "Мария","получила","от","врача"),
        ):
            raw=raw_surface_grammar(text)
            self.assertEqual(flagged(raw,recipient,
                                    "ending_feminine_nominative_possible"),1)
            self.assertEqual(flagged(raw,verb,"past_verb_shape_possible"),1)
            self.assertEqual(flagged(raw,prep,"preposition_ot"),1)
            self.assertEqual(flagged(raw,source,"previous_word_is_ot"),1)
        # We have not inferred anything about semantic AGENT / RECIPIENT / SOURCE.

    def test_multiple_predicates_punctuation_and_negative_input_only(self):
        text="Пока сосед работал, Антон не выслал книгу Валерии."
        tokens=raw_surface_grammar(text)
        self.assertIn(",",[w.form for w in tokens])
        self.assertEqual(flagged(tokens,"не","negative_particle"),1)
        self.assertEqual(flagged(tokens,"выслал","previous_word_negative_particle"),1)
        self.assertEqual(flagged(tokens,"выслал","past_verb_shape_possible"),1)
        self.assertEqual(flagged(tokens,"книгу",
                                "ending_feminine_accusative_possible"),1)
        self.assertTrue(all(text[w.start:w.end]==w.form for w in tokens))

    def test_nonraw_inputs_and_missing_text_fail_closed(self):
        for value in (None,{},[],123,"", "   "):
            with self.subTest(value=str(value)):
                with self.assertRaises(ValueError):
                    raw_surface_grammar(value)
        with self.assertRaises(ValueError):
            feature_at(raw_surface_grammar("Письмо.")[0],"oracle_agent")
