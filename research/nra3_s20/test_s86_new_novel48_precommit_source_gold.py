"""S86 one-time new syntax source+constructed gold PRECOMMIT contract tests.

No genuine original43M H8 or trained S85 weights read by this module.
"""
import ast
import unittest
from pathlib import Path

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s82_predicate_centered_situation_fields import AREAS
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,
)
from research.nra3_s20.s79_NEW_NOVEL48_SOURCE_ONLY_locked_before_S78_firstread import (
    new_novel48_source_only,
)
from research.nra3_s20.s86_newSYNTHETIC48_SOURCEONLY_never_train import (
    S86_novel48_source_only,S86_novel48_source_sha256,
)
from research.nra3_s20.s86_NOVEL48_GOLD_SEPARATE_POSTHOC_ONLY import (
    S86_frozen_authored_gold_ONE_SHOT_POSTHOC_only,
    preregistered_S86_firstread_manifest,
)


class TestS86NewSyntaxFirstReadBeforeModel(unittest.TestCase):
    def test_new48_text_distinct_from_old_72_train_and_spent_48(self):
        source=S86_novel48_source_only()
        old={r["text"] for r in source_only_new72()}
        spent={r["text"] for r in new_novel48_source_only()}
        self.assertEqual(len(source),48)
        self.assertEqual(len({r["text"] for r in source}),48)
        self.assertTrue(all(r["text"] not in old and r["text"] not in spent
                            for r in source))
        self.assertEqual(len(S86_novel48_source_sha256()),64)
        self.assertTrue(all(set(r)=={"row_id","text"} for r in source))

    def test_separately_fixed_48_russian_word_anchor_and_eight_labels(self):
        texts=S86_novel48_source_only()
        labels=S86_frozen_authored_gold_ONE_SHOT_POSTHOC_only()
        self.assertEqual(len(labels),48)
        self.assertEqual([r["row_id"] for r in texts],
                         [z["row_id"] for z in labels])
        for text,annotation in zip(texts,labels):
            raw=raw_surface_grammar(text["text"])
            idx=annotation["original_event_word_index_POSTHOC_ONLY"]
            self.assertTrue(0<=idx<len(raw))
            self.assertEqual(raw[idx].form.casefold(),
                             annotation["original_event_word_form_POSTHOC_ONLY"].casefold())
            self.assertEqual(len(annotation["event_situation_factor8_POSTHOC_ONLY"]),8)
            self.assertFalse(annotation["independently_human_adjudicated"])

    def test_all_eight_area_support_and_nonexclusive_factors(self):
        labels=S86_frozen_authored_gold_ONE_SHOT_POSTHOC_only()
        self.assertEqual(len(AREAS),8)
        support=[sum(z["event_situation_factor8_POSTHOC_ONLY"][i] for z in labels)
                 for i in range(8)]
        self.assertTrue(all(v>=5 for v in support))
        self.assertGreaterEqual(sum(sum(z["event_situation_factor8_POSTHOC_ONLY"])>1
                                    for z in labels),15)

    def test_no_gold_or_S85_import_in_frozen_source_module(self):
        p=Path("research/nra3_s20/s86_newSYNTHETIC48_SOURCEONLY_never_train.py")
        tree=ast.parse(p.read_text(encoding="utf8"))
        imports=[z.module or "" for z in tree.body if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("gold" in x.lower() or "s85" in x.lower() or
                             "s79" in x.lower() or "s83" in x.lower() or
                             "eval" in x.lower() or "torch" in x.lower()
                             for x in imports))
        self.assertTrue(all(set(r)=={"row_id","text"}
                            for r in S86_novel48_source_only()))

    def test_firstread_protocol_has_no_score_threshold_optimizing(self):
        manifest=preregistered_S86_firstread_manifest()
        self.assertEqual(manifest["n_new_syntactic_firstread_cases"],48)
        self.assertEqual(manifest["n_author_locked_verb_and_eightfactor_targets"],48)
        self.assertEqual(manifest["new_source_only_SHA256"],S86_novel48_source_sha256())
        self.assertTrue(manifest["S85_source_only_logits_PHYSICALLY_SAVED_BEFORE_gold_open"])
        self.assertTrue(manifest["no_checkpoint_selection_no_threshold_search_no_training_after_firstread"])
        self.assertFalse(manifest["synthetic_author_labels_not_independent_human_truth"] is False)
        self.assertFalse(manifest["candidate_accepted"])
        self.assertFalse(manifest["activation_allowed"])
        self.assertEqual(manifest["output_gate"],0)


if __name__=="__main__":
    unittest.main()
