"""S18c6 ONE-TIME source-only EVAL32 contract. NO models or inference."""
from __future__ import annotations

from dataclasses import replace
import ast
import hashlib
from pathlib import Path
import unittest

from research.nra3_s18.s18c6_new32_one_time_eval_prereg import (
    SPLIT, CATEGORIES, EXPECTED_ROLES, EXPECTED_EDGES,
    PRE_FROZEN_CANDIDATE_2597_FP16_SHA, PRE_FROZEN_CONTROL_2309_FP16_SHA,
    NEW_EVAL_SOURCE_CI,EVAL32_COMPLETE_CANONICAL_SHA256,
    DirectedGold, build_new_eval32,validate_new_eval32,canonical_eval32_bytes,
    seal_new_eval32_source,
)
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s18.ru96_multiverb_train_source import build_s18_new_train96
from research.nra3_s9.new_ru_train72 import build_s9_train72
from research.nra3_s4.russian_ud_train_only import build_train as build_s4_train


class NewOneTimeEval32PreRegistration(unittest.TestCase):

    def test_exact_32_new_source_rows_8_categories_and_104_manually_labeled_links(self):
        rows=build_new_eval32()
        self.assertEqual(len(rows),32)
        self.assertEqual(len(CATEGORIES),8)
        self.assertEqual([r.category for r in rows],
                         [c for c in CATEGORIES for _ in range(4)])
        self.assertTrue(all(r.split==SPLIT for r in rows))
        self.assertEqual(sum(len(r.directed_gold) for r in rows),EXPECTED_EDGES)
        roles={}
        for row in rows:
            for edge in row.directed_gold:
                roles[edge.role]=roles.get(edge.role,0)+1
        self.assertEqual(roles,EXPECTED_ROLES)
        self.assertEqual(roles,{"AGENT":32,"PATIENT":36,"RECIPIENT":32,"SOURCE":4})

    def test_new32_unicode_offset_and_no_duplicates_with_earlier_training(self):
        evals=build_new_eval32()
        old_sha={r.sha256() for r in build_s4_train()}|{
            r.sha256() for r in build_s9_train72()}|{
            r.text_sha256() for r in build_s18_new_train96()}
        self.assertTrue({r.text_sha256() for r in evals}.isdisjoint(old_sha))
        self.assertEqual(len({r.text_sha256() for r in evals}),32)
        for row in evals:
            words=raw_surface_grammar(row.text)
            self.assertTrue(words)
            self.assertEqual(row.text[words[-1].start:words[-1].end],".")
            self.assertEqual(len(words),len(set((w.start,w.end) for w in words)))
            self.assertTrue(all(row.text[t.start:t.end]==t.form for t in words))
            self.assertTrue(all(1<=p<len(words) for p in row.predicate_word_indices))

    def test_gold_modes_never_conflate_recipient_with_agent_or_negation_with_event(self):
        rows=build_new_eval32()
        for row in rows:
            roles=[z.role for z in row.directed_gold]
            if row.category==CATEGORIES[3]:
                self.assertEqual(set(roles),{"PATIENT","SOURCE","RECIPIENT"})
                self.assertNotIn("AGENT",roles)
            if row.category==CATEGORIES[4]:
                self.assertEqual(row.event_modality,"NEGATED")
                self.assertIn("не",row.text.casefold().split())
            if row.category==CATEGORIES[5]:
                self.assertEqual(row.event_modality,"FUTURE_UNREALIZED")
            if row.category==CATEGORIES[7]:
                self.assertEqual(len(row.predicate_word_indices),2)
                self.assertEqual(len(row.directed_gold),5)

    def test_fail_closed_if_new_evaluation_labels_or_order_were_altered(self):
        original=build_new_eval32()
        with self.assertRaises(ValueError):
            validate_new_eval32(original[:-1])
        duplicated=original[:1]+original[:1]+original[2:]
        with self.assertRaises(ValueError):
            validate_new_eval32(duplicated)
        bad=replace(original[0],directed_gold=(
            DirectedGold(2,1,"PATIENT"),)+original[0].directed_gold[1:])
        # Swapping agent to patient also breaks role totals and row role invariant.
        with self.assertRaises(ValueError):
            validate_new_eval32((bad,)+original[1:])
        altered=replace(original[3],predicate_word_indices=(1,))
        with self.assertRaises(ValueError):
            validate_new_eval32(original[:3]+(altered,)+original[4:])

    def test_canonical_sha_is_stable_and_never_training_or_model_scoring(self):
        rows=build_new_eval32()
        canonical=canonical_eval32_bytes(rows)
        sha=hashlib.sha256(canonical).hexdigest()
        seal=seal_new_eval32_source()
        self.assertEqual(seal["new_eval_source_complete_sha256"],sha)
        if EVAL32_COMPLETE_CANONICAL_SHA256 is not None:
            self.assertEqual(sha,EVAL32_COMPLETE_CANONICAL_SHA256)
        self.assertEqual(seal["sha_locked_in_source"],
                         EVAL32_COMPLETE_CANONICAL_SHA256 is not None)
        self.assertEqual(seal["new_eval_gold_predicate_to_argument_links"],104)
        self.assertFalse(seal["candidate_scored_on_eval32"])
        self.assertFalse(seal["original43M_GT_inference_executed_on_eval32"])
        self.assertTrue(seal["new_independent_eval_used_for_training_or_model_selection"] is False)
        self.assertEqual(len(PRE_FROZEN_CANDIDATE_2597_FP16_SHA),64)
        self.assertEqual(len(PRE_FROZEN_CONTROL_2309_FP16_SHA),64)

    def test_ast_only_source_and_raw_tokens_never_candidate_model(self):
        path=Path("research/nra3_s18/s18c6_new32_one_time_eval_prereg.py")
        tree=ast.parse(path.read_text(encoding="utf-8"))
        imports=[(n.module or "") for n in ast.walk(tree)
                 if isinstance(n,ast.ImportFrom)]
        forbidden=("nra3_s12","nra3_s15","nra3_s17",
                   "s18c1_original43m","s18c4_paid","s18c5_original43m",
                   "real43m_new96_readonly_cpu_capture","native_cpu",
                   "transformers","torch")
        self.assertFalse(any(term in name for name in imports for term in forbidden))
        called={node.func.id for node in ast.walk(tree)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        self.assertNotIn("eval",called)
        self.assertNotIn("train",called)
        self.assertNotIn("encode",called)
        self.assertNotIn("predict",called)
        self.assertIsNone(NEW_EVAL_SOURCE_CI if
                          EVAL32_COMPLETE_CANONICAL_SHA256 is None else None)
