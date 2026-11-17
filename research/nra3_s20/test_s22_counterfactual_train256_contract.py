"""Predeclared S22 counterfactual synthetic TRAIN graph contracts only."""
from __future__ import annotations

import inspect
import unittest
from collections import Counter

from research.nra3_s19.graph_schema import canonical_train_only_sha, validate_graph
from research.nra3_s19.train_only_event_graph_source import build_s19_train_only
from research.nra3_s20.s20_new96_synthetic_graph_train_source import (
    build_s20_synthetic_train96,
)
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    COUNTERFACTUAL_PAIRS,FAMILIES,PATTERNS,RELATION_OF,
    build_s22_train256, splits, manifest, text_only_inputs,
    verify_coverage, FROZEN_S22_TRAIN256_SHA256,
)


class S22SyntheticGraphTrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_s22_train256()
        cls.splits = splits()

    def test_source_correct_256_distinct_new_human_annotated_graphs(self):
        self.assertEqual(len(FAMILIES),8)
        self.assertEqual(len(PATTERNS),32)
        self.assertEqual(len(self.rows),256)
        self.assertEqual(len({r.text for r in self.rows}),256)
        previous={r.text for r in build_s19_train_only()}
        previous|={r.text for r in build_s20_synthetic_train96()}
        self.assertFalse(previous.intersection(r.text for r in self.rows))
        for row in self.rows:
            validate_graph(row)
            self.assertEqual(row.output_gate,0)
            self.assertFalse(row.candidate_accepted)
            self.assertFalse(any(e.factuality == "UNKNOWN" for e in row.events))

    def test_four_relations_balanced_six_different_syntax_per_relation(self):
        rel, rejected=verify_coverage(self.rows)
        self.assertEqual(rel,{"COORDINATION":48, "CONDITION_RESULT":48,
                              "TEMPORAL_SEQUENCE":48,"CONTRAST":48})
        self.assertEqual(rejected,32)
        self.assertEqual({r:len([p for p in PATTERNS if RELATION_OF.get(p)==r])
                          for r in rel},{r:6 for r in rel})
        self.assertEqual(len(self.splits),40)

    def test_each_holdout_has_all_relation_status_and_rejection_classes_in_training(self):
        for axis,group,tr,held in self.splits:
            self.assertTrue(set(tr).isdisjoint(held))
            self.assertEqual(set(tr)|set(held),set(range(256)))
            self.assertEqual(len(held),(32 if axis=="LEXICAL" else 8))
            relation=Counter(l.relation for i in tr
                             for l in self.rows[i].event_relations)
            status={e.factuality for i in tr for e in self.rows[i].events}
            self.assertEqual(set(relation),{
                "COORDINATION","CONDITION_RESULT","TEMPORAL_SEQUENCE","CONTRAST"})
            self.assertTrue({"ASSERTED","NEGATED","CONDITIONAL",
                             "FUTURE_UNREALIZED"} <= status)
            self.assertGreater(sum(a.polarity=="REJECTED" for i in tr
                                   for a in self.rows[i].arguments),0)
            if axis=="CONSTRUCTION":
                self.assertTrue(all(i%32==group for i in held))
            else:
                self.assertTrue(all(i//32==group for i in held))

    def test_minimal_if_when_pairs_change_only_connective_and_event_truth(self):
        self.assertEqual(COUNTERFACTUAL_PAIRS,((6,12),(7,13)))
        for family in range(8):
            for conditional, temporal in COUNTERFACTUAL_PAIRS:
                src=self.rows[family*32+conditional]
                tgt=self.rows[family*32+temporal]
                self.assertEqual(
                    src.text.replace("Если","Когда").replace("если","когда"),
                    tgt.text)
                self.assertEqual([e.factuality for e in src.events],
                                 ["CONDITIONAL","CONDITIONAL"])
                self.assertEqual([e.factuality for e in tgt.events],
                                 ["FUTURE_UNREALIZED","FUTURE_UNREALIZED"])
                self.assertEqual(
                    [(a.event_id,a.token,a.role,a.polarity) for a in src.arguments],
                    [(a.event_id,a.token,a.role,a.polarity) for a in tgt.arguments])
                self.assertEqual(len(src.event_relations),1)
                self.assertEqual(len(tgt.event_relations),1)
                self.assertEqual(src.event_relations[0].relation,"CONDITION_RESULT")
                self.assertEqual(tgt.event_relations[0].relation,"TEMPORAL_SEQUENCE")
                direction=(2,1) if conditional==7 else (1,2)
                self.assertEqual(
                    (src.event_relations[0].source_event,src.event_relations[0].target_event),
                    direction)
                self.assertEqual(
                    (tgt.event_relations[0].source_event,tgt.event_relations[0].target_event),
                    direction)

    def test_rejected_is_not_negated_and_four_separate_surface_realizations(self):
        for family in range(8):
            for pat in range(24,28):
                row=self.rows[family*32+pat]
                self.assertEqual(len(row.events),1)
                self.assertEqual(row.events[0].factuality,"ASSERTED")
                self.assertEqual(len([a for a in row.arguments
                    if a.role=="RECIPIENT" and a.polarity=="REJECTED"]),1)
                self.assertEqual(len([a for a in row.arguments
                    if a.role=="RECIPIENT" and a.polarity=="POSITIVE"]),1)
            for pat in (28,29):
                row=self.rows[family*32+pat]
                self.assertEqual(row.events[0].factuality,"NEGATED")
                self.assertEqual(sum(a.polarity=="REJECTED" for a in row.arguments),0)

    def test_frozen_source_sha_and_explicit_never_eval_never_live_contract(self):
        m=manifest()
        self.assertEqual(m["source_sha256"],canonical_train_only_sha(self.rows))
        if FROZEN_S22_TRAIN256_SHA256:
            self.assertEqual(m["source_sha256"],FROZEN_S22_TRAIN256_SHA256)
        self.assertEqual(m["rows_synthetic_TRAIN_ONLY"],256)
        self.assertEqual(m["grouped_TRAIN_only_folds"],40)
        self.assertEqual(m["minimal_if_vs_when_pairs"],16)
        self.assertEqual(m["role_rejection_gold_links"],32)
        self.assertTrue(m["new_true43m_GT_CPU_not_yet_captured"])
        self.assertTrue(m["no_model_forward_no_training_no_independent_EVAL"])
        self.assertFalse(m["candidate_accepted"])
        self.assertFalse(m["activation_allowed"])
        self.assertEqual(m["output_gate"],0)
        self.assertEqual(text_only_inputs(),tuple(row.text for row in self.rows))
        self.assertTrue(all(isinstance(t,str) for t in text_only_inputs()))
        import research.nra3_s20.s22_new256_counterfactual_train_graph_source as src
        code=inspect.getsource(src)
        for forbidden in ("optimizer.step(", "load_file(", "torch.nn", ".backward(",
                          "model.forward(", "torch.inference_mode("):
            self.assertNotIn(forbidden,code)


if __name__ == "__main__":
    unittest.main()
