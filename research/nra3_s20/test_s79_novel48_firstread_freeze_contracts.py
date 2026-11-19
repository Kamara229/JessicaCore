"""S79 PRE-inference one-shot novel syntax source/annotation wall, no model."""
from __future__ import annotations
import ast
import hashlib
import unittest
from collections import Counter
from pathlib import Path

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s74_newtrain120_source_ONLY import new_source_only_train120
from research.nra3_s20.s79_NEW_NOVEL48_SOURCE_ONLY_locked_before_S78_firstread import (
    new_novel48_source_only,new_novel48_source_sha256,SOURCE_TEXTS,
)
from research.nra3_s20.s79_NEW_NOVEL48_LABELS_POSTHOC_ONLY_never_in_forward import (
    LABELS,frozen_novel48_labels_for_ONE_FINAL_READ_only,preregistered_manifest,
)


class TestS79PredeclaredFirstReadBoundary(unittest.TestCase):
    def test_48_new_source_cases_different_texts_from_old_S74_training(self):
        source=new_novel48_source_only()
        old=set(z["text"] for z in new_source_only_train120())
        self.assertEqual(len(source),48)
        self.assertEqual(len({x["text"] for x in source}),48)
        self.assertTrue(all(set(r)=={"row_id","text"} for r in source))
        self.assertTrue(all(r["text"] not in old for r in source))
        self.assertEqual(len(new_novel48_source_sha256()),64)
        self.assertEqual(len(SOURCE_TEXTS),48)

    def test_all_48_labels_anchored_to_exact_raw_unicode_verbs(self):
        source=new_novel48_source_only()
        labels=frozen_novel48_labels_for_ONE_FINAL_READ_only()
        self.assertEqual(len(labels),48)
        for row,l in zip(source,labels):
            self.assertEqual(row["row_id"],l["row_id"])
            tokens=raw_surface_grammar(row["text"])
            self.assertGreaterEqual(len(tokens),3)
            self.assertLessEqual(len(tokens),40)
            i=l["constructed_anchor_INDEX_ONLY_FOR_POSTHOC"]
            self.assertEqual(tokens[i].form.casefold(),
                             l["expected_verb_POSTHOC"].casefold())
            self.assertEqual(len(l["eight_area_target_POSTHOC"]),8)
            self.assertTrue(l["not_independently_human_adjudicated"])

    def test_frozen_label_all_eight_domains_represented(self):
        labels=frozen_novel48_labels_for_ONE_FINAL_READ_only()
        supports=Counter(f for x in labels for f in x["factors_posthoc_only"])
        self.assertEqual(len(supports),8)
        self.assertTrue(all(s>=4 for s in supports.values()))
        self.assertGreaterEqual(sum(sum(x["eight_area_target_POSTHOC"])>=2
                                    for x in labels),15)
        p=preregistered_manifest()
        self.assertEqual(p["n_new_syntactic_shift_probes"],48)
        self.assertEqual(p["raw_only_source_sha256"],new_novel48_source_sha256())
        self.assertTrue(p["one_time_score_no_new_S78_fitting_or_hparam_selection"])
        self.assertTrue(p["original_S78_weights_frozen_before_this_source"])
        self.assertTrue(p["manually_constructed_NOT_independently_human_adjudicated"])
        self.assertFalse(p["candidate_accepted"])
        self.assertEqual(p["output_gate"],0)

    def test_source_module_has_no_gold_or_prediction_dependencies(self):
        p=Path("research/nra3_s20/s79_NEW_NOVEL48_SOURCE_ONLY_locked_before_S78_firstread.py")
        tree=ast.parse(p.read_text(encoding="utf8"))
        deps=[]
        for node in tree.body:
            if isinstance(node,ast.ImportFrom):
                deps.append(node.module or "")
            elif isinstance(node,ast.Import):
                deps.extend(z.name for z in node.names)
        self.assertTrue(all("gold" not in s.lower()
                            and "labels" not in s.lower()
                            and "s78" not in s.lower()
                            and "torch" not in s.lower()
                            and "eval" not in s.lower()
                            for s in deps))
        self.assertTrue(all(set(x)=={"row_id","text"}
                            for x in new_novel48_source_only()))

    def test_no_future_retraining_or_test_driven_weight_update(self):
        p=preregistered_manifest()
        self.assertTrue(p["source_gold_forward_wall_required"])
        self.assertTrue(p["fixed_eight_situation_heads_no_head_drop"])
        self.assertTrue(p["fixed_raw_tokenizer_genuine43m_and_first616J7_required"])
        self.assertTrue(p["novel_syntax_stress_NOT_statistically_representative"])
        self.assertTrue(p["no_spent_EVAL128_EVAL32_S66_held_gold"])


if __name__=="__main__":
    unittest.main()
