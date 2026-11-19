"""S83 pre-forward new diverse72 TRAIN-only SOURCE/GOLD separation tests."""
import ast
import unittest
from pathlib import Path
from collections import Counter

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,source_only_new72_sha256,TRAIN_GROUPS,DIAG_GROUPS,
)
from research.nra3_s20.s83_NEW_DIVERSE72_gold_TRAIN_LOSS_ONLY import (
    S83_train_loss_gold_only,precommit_manifest,
)
from research.nra3_s20.s82_predicate_centered_situation_fields import AREAS


class TestS83NewDiverse72TRAIN(unittest.TestCase):
    def test_all_72_new_raw_source_only_unique_and_no_model_labels(self):
        rows=source_only_new72()
        self.assertEqual(len(rows),72)
        self.assertEqual(len({x["text"] for x in rows}),72)
        self.assertEqual(len({x["row_id"] for x in rows}),72)
        self.assertTrue(all(set(x)=={"row_id","group","text"} for x in rows))
        self.assertEqual(set(Counter(x["group"] for x in rows).values()),{8})
        self.assertEqual(len(source_only_new72_sha256()),64)

    def test_new_constructed_train_targets_exact_unicode_event_index(self):
        rows=source_only_new72()
        gold=S83_train_loss_gold_only()
        self.assertEqual(len(gold),72)
        self.assertEqual([x["row_id"] for x in gold],
                         [x["row_id"] for x in rows])
        for s,z in zip(rows,gold):
            tokens=raw_surface_grammar(s["text"])
            index=z["event_raw_word_index_ONLY_loss_or_posthoc"]
            self.assertLess(index,len(tokens))
            self.assertEqual(tokens[index].form.casefold(),
                z["event_word_source_MATCH_ONLY_loss"].casefold())
            self.assertEqual(len(z["eight_area_multi_hot_ONLY_loss"]),8)
            self.assertTrue(z["not_independently_human_adjudicated"])

    def test_grouped_new_train56_and_synthetic_diag16_no_source_overlap(self):
        rows=source_only_new72()
        train=[z for z in rows if z["group"] in TRAIN_GROUPS]
        diag=[z for z in rows if z["group"] in DIAG_GROUPS]
        self.assertEqual(len(train),56)
        self.assertEqual(len(diag),16)
        self.assertTrue(set(z["group"] for z in train).isdisjoint(
                        z["group"] for z in diag))
        self.assertTrue(set(z["text"] for z in train).isdisjoint(
                        z["text"] for z in diag))
        self.assertEqual(len(TRAIN_GROUPS),7)
        self.assertEqual(len(DIAG_GROUPS),2)

    def test_all_eight_areas_with_overlap_in_train_and_grouped_diag(self):
        rows=source_only_new72()
        labels=S83_train_loss_gold_only()
        for selection in (TRAIN_GROUPS,DIAG_GROUPS):
            d=[g for g,r in zip(labels,rows) if r["group"] in selection]
            supports=[sum(z["eight_area_multi_hot_ONLY_loss"][i] for z in d)
                      for i in range(len(AREAS))]
            self.assertTrue(all(x>=2 for x in supports))
            self.assertGreaterEqual(sum(sum(z["eight_area_multi_hot_ONLY_loss"])>=2
                                        for z in d),len(d)//3)

    def test_gold_file_never_imported_by_raw_source_module(self):
        path=Path("research/nra3_s20/s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent.py")
        tree=ast.parse(path.read_text(encoding="utf8"))
        imports=[z.module for z in tree.body if isinstance(z,ast.ImportFrom)]
        self.assertTrue(all("gold" not in (x or "").lower()
                            and "s79" not in (x or "").lower()
                            and "torch" not in (x or "").lower()
                            and "model" not in (x or "").lower()
                            for x in imports))
        self.assertTrue(all(set(z)=={"row_id","group","text"}
                            for z in source_only_new72()))

    def test_manifest_no_training_or_independent_test_claim(self):
        p=precommit_manifest()
        self.assertEqual(p["n_source_rows"],72)
        self.assertEqual(p["n_train_rows"],56)
        self.assertEqual(p["n_synthetic_diag_rows"],16)
        self.assertEqual(p["source_only_sha256"],source_only_new72_sha256())
        self.assertTrue(p["S79_48_one_time_spent_test_NEVER_imported"])
        self.assertTrue(p["new_S82_neural_parameters_not_trained_yet"])
        self.assertTrue(p["independent_prospective_new_first_read_NOT_created"])
        self.assertEqual(p["output_gate"],0)
        self.assertFalse(p["candidate_accepted"])


if __name__=="__main__":
    unittest.main()
