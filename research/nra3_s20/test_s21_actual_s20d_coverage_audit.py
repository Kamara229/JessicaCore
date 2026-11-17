"""S21 source-only contracts, no fabricated S20d scores and no model forward."""
from __future__ import annotations

from collections import Counter
import inspect
import unittest

from research.nra3_s20.s20_new96_synthetic_graph_train_source import (
    PATTERNS, build_s20_synthetic_train96, train_only_split_contract,
    FROZEN_S20_TRAIN96_SHA256, train_only_manifest,
)
from research.nra3_s20.s21_audit_s20d_fold_coverage import (
    audit, gold_support, micro_f1, PINNED_S20D_FP16_SHA256,
)


class S21ReadOnlyDiagnosticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_s20_synthetic_train96()
        cls.folds = train_only_split_contract()

    def test_original_96_train_source_sealed(self):
        self.assertEqual(train_only_manifest()["source_sha256"],
                         FROZEN_S20_TRAIN96_SHA256)
        self.assertEqual(len(self.rows), 96)
        self.assertEqual(len(self.folds), 20)

    def test_one_structurally_missing_relation_in_construction_holdout(self):
        zero = []
        for axis, group, train, held in self.folds:
            ts, r = gold_support(self.rows, train)
            hs, h = gold_support(self.rows, held)
            for relation, count in hs.items():
                if count and ts[relation] == 0:
                    zero.append((axis,group,relation,count))
            self.assertGreaterEqual(r, 0)
            self.assertGreaterEqual(h, 0)
        self.assertEqual(zero, [(
            "CONSTRUCTION",PATTERNS.index("asserted_coordination"),"COORDINATION",8)])

    def test_separate_role_rejection_not_equal_event_negation(self):
        rejected = Counter()
        negated = Counter()
        for i, row in enumerate(self.rows):
            for a in row.arguments:
                if a.polarity == "REJECTED":
                    rejected[i%12] += 1
            for e in row.events:
                if e.factuality == "NEGATED":
                    negated[i%12] += 1
        self.assertEqual(sum(rejected.values()), 16)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(sum(negated.values()), 24)
        self.assertTrue(set(rejected).isdisjoint(negated))

    def test_condition_relation_has_two_templates_and_both_directions(self):
        groups = []
        for pat in range(12):
            g = self.rows[pat]
            rels = [link for link in g.event_relations
                    if link.relation == "CONDITION_RESULT"]
            if rels:
                groups.append((pat, rels[0].source_event, rels[0].target_event))
        self.assertEqual(len(groups), 2)
        self.assertEqual({(s,t) for _,s,t in groups}, {(1,2),(2,1)})

    def test_fail_closed_rejects_synthetic_fake_artifact(self):
        with self.assertRaisesRegex(ValueError, "not an actual S20d"):
            audit({"stage": "SYNTHETIC_FAKE"}, PINNED_S20D_FP16_SHA256)

    def test_tp_fp_fn_math_and_read_only_source(self):
        self.assertEqual(micro_f1({"TP": 0,"FP": 4,"FN": 2}), 0.)
        self.assertAlmostEqual(micro_f1({"TP": 10,"FP": 2,"FN": 3}),20/25)
        import research.nra3_s20.s21_audit_s20d_fold_coverage as source
        code = inspect.getsource(source)
        self.assertNotIn("optimizer.step(",code)
        self.assertNotIn("model.forward(",code)
        self.assertNotIn("torch.inference_mode(",code)


if __name__ == "__main__":
    unittest.main()
