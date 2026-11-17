"""S20 immutable TRAIN source contracts; never use a model or EVAL corpus."""
from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

from research.nra3_s19.graph_schema import (
    raw_model_input, canonical_train_only_sha, validate_graph,
)
from research.nra3_s19.train_only_event_graph_source import (
    build_s19_train_only, TRAIN_ONLY_FROZEN_SHA256,
)
from research.nra3_s20.s20_new96_synthetic_graph_train_source import (
    FAMILY_COUNT, PATTERN_COUNT, EXPECTED_ROWS, FAMILIES, PATTERNS,
    build_s20_synthetic_train96, group_rows, train_only_split_contract,
    raw_text_only_train_input, train_only_manifest,
)


def forms(graph):
    return tuple(t.form for t in graph.tokens)


def event_at(g, word):
    matches = [e for e in g.events if g.tokens[e.predicate - 1].form == word]
    if len(matches) != 1:
        raise AssertionError("S20 test requires exactly one event word")
    return matches[0]


class S20SyntheticTrainOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_s20_synthetic_train96()
        cls.matrix = group_rows()

    def test_all_ninety_six_are_sealed_source_train_with_unique_raw_texts(self):
        self.assertEqual((FAMILY_COUNT,PATTERN_COUNT,EXPECTED_ROWS),(8,12,96))
        self.assertEqual(len(self.rows),96)
        self.assertEqual(len(self.matrix),96)
        self.assertEqual(len({g.text for g in self.rows}),96)
        self.assertEqual(
            {g.text for g in self.rows}.intersection(
                g.text for g in build_s19_train_only()),set())
        self.assertEqual(raw_text_only_train_input(),
                         tuple(g.text for g in self.rows))
        for g in self.rows:
            validate_graph(g)
            self.assertFalse(g.candidate_accepted)
            self.assertEqual(g.output_gate,0)
            self.assertEqual(raw_model_input(g.text).tokens,g.tokens)
            self.assertTrue(all(g.text[t.start:t.end]==t.form
                                for t in g.tokens))

    def test_counts_include_multiple_instances_of_each_scoped_class(self):
        m=train_only_manifest()
        self.assertEqual(m["train_only_sentences"],96)
        self.assertEqual(m["gold_events"],160)
        self.assertEqual(m["gold_directed_role_edges"],432)
        self.assertEqual(m["gold_rejected_recipients"],16)
        self.assertEqual(m["gold_coreference_edges"],64)
        self.assertEqual(m["gold_event_relation_edges"],64)
        self.assertEqual(m["factuality_class_counts"],{
            "ASSERTED":80,"NEGATED":24,
            "CONDITIONAL":32,"FUTURE_UNREALIZED":24})
        self.assertEqual(m["event_relation_class_counts"],{
            "COORDINATION":8,"CONTRAST":16,
            "CONDITION_RESULT":16,"TEMPORAL_SEQUENCE":24})
        self.assertEqual(m["source_sha256"],
                         canonical_train_only_sha(self.rows))
        self.assertEqual(m["original_s19_source_SHA_preserved"],
                         TRAIN_ONLY_FROZEN_SHA256)
        self.assertTrue(m["all_human_graph_annotations_target_only"])
        self.assertTrue(m["original43M_GT_hidden_for_new96_NOT_YET_CAPTURED"])
        self.assertFalse(m["independent_EVAL_accessed"])
        self.assertFalse(m["model_inference_or_train_performed"])

    def test_every_lexical_family_has_each_event_and_relation_type(self):
        for fam in range(FAMILY_COUNT):
            graphs=[self.matrix[(fam, p)] for p in range(PATTERN_COUNT)]
            event_types={e.factuality for g in graphs for e in g.events}
            rel_types={l.relation for g in graphs for l in g.event_relations}
            self.assertEqual(event_types,{
                "ASSERTED","NEGATED","CONDITIONAL","FUTURE_UNREALIZED"})
            self.assertEqual(rel_types,{
                "COORDINATION","CONTRAST","CONDITION_RESULT","TEMPORAL_SEQUENCE"})
            self.assertEqual(sum(a.polarity=="REJECTED"
                                 for g in graphs for a in g.arguments),2)

    def test_train_only_fold_schedule_leaves_no_family_or_template_leak(self):
        folds=train_only_split_contract()
        self.assertEqual(len(folds),20)
        self.assertEqual(len([f for f in folds if f[0]=="LEXICAL"]),8)
        self.assertEqual(len([f for f in folds if f[0]=="CONSTRUCTION"]),12)
        for kind, target, tr, held in folds:
            self.assertFalse(set(tr)&set(held))
            self.assertEqual(set(tr)|set(held),set(range(96)))
            if kind=="LEXICAL":
                self.assertEqual((len(tr),len(held)),(84,12))
                self.assertTrue(all(i//12==target for i in held))
            else:
                self.assertEqual((len(tr),len(held)),(88,8))
                self.assertTrue(all(i%12==target for i in held))
            statuses={
                e.factuality for i in tr for e in self.rows[i].events
            }
            self.assertEqual(
                statuses,{"ASSERTED","NEGATED","CONDITIONAL","FUTURE_UNREALIZED"},
                "All TRAIN folds must actually contain the rare truth labels")
            self.assertGreater(
                sum(a.polarity=="REJECTED"
                    for i in tr for g in (self.rows[i],)
                    for a in g.arguments),0,
                "All TRAIN folds must contain a rejected role")
        self.assertEqual(len({
            i for kind,_,_,held in folds if kind=="LEXICAL" for i in held}),96)
        self.assertEqual(len({
            i for kind,_,_,held in folds if kind=="CONSTRUCTION" for i in held}),96)

    def test_negated_event_and_rejected_participant_are_distinct(self):
        for fam in range(8):
            f=FAMILIES[fam]
            for pat in ("rejected_recipient_preposed_time",
                        "rejected_recipient_reversed_order"):
                g=self.matrix[(fam,PATTERNS.index(pat))]
                self.assertEqual(tuple(e.factuality for e in g.events),
                                 ("ASSERTED",))
                rejected=[a for a in g.arguments if a.polarity=="REJECTED"]
                affirmed=[a for a in g.arguments
                          if a.role=="RECIPIENT" and a.polarity=="POSITIVE"]
                self.assertEqual(len(rejected),len(affirmed))
                self.assertEqual(len(rejected),1)
                self.assertEqual(g.tokens[rejected[0].token-1].form,f.recipient)
                self.assertEqual(
                    g.tokens[affirmed[0].token-1].form,
                    f.alternative_recipient)
            neg=self.matrix[(fam,PATTERNS.index("negated_inverted_word_order"))]
            self.assertEqual([e.factuality for e in neg.events],["NEGATED"])
            self.assertTrue(all(a.polarity=="POSITIVE" for a in neg.arguments))

    def test_reversed_condition_and_reversed_temporal_keep_true_direction(self):
        for fam in range(8):
            f=FAMILIES[fam]
            for pat,label in (
                ("conditional_if_subordinate_after_result","CONDITION_RESULT"),
                ("past_temporal_subordinate_after_result","TEMPORAL_SEQUENCE"),
            ):
                g=self.matrix[(fam,PATTERNS.index(pat))]
                self.assertEqual(len(g.events),2)
                self.assertEqual(
                    [(l.source_event,l.target_event,l.relation)
                     for l in g.event_relations],
                    [(2,1,label)])
                self.assertEqual(len(g.coreferences),1)
                c=g.coreferences[0]
                self.assertEqual(g.tokens[c.mention-1].form,"его")
                self.assertEqual(g.tokens[c.antecedent-1].form,f.object)
                self.assertLess(c.antecedent,c.mention)
                if label=="CONDITION_RESULT":
                    self.assertTrue(
                        all(e.factuality=="CONDITIONAL" for e in g.events))

    def test_future_analytic_uses_infinitive_while_conditional_marks_both(self):
        for fam in range(8):
            f=FAMILIES[fam]
            analytic=self.matrix[(fam,PATTERNS.index("analytic_future"))]
            self.assertEqual(len(analytic.events),1)
            e=analytic.events[0]
            self.assertEqual(analytic.tokens[e.predicate-1].form,
                             f.second_infinitive)
            self.assertEqual(e.factuality,"FUTURE_UNREALIZED")
            self.assertIn("будет",forms(analytic))
            conditional=self.matrix[(fam,PATTERNS.index("conditional_if_future"))]
            self.assertEqual(len(conditional.events),2)
            self.assertEqual(len(conditional.event_relations),1)
            self.assertEqual(conditional.event_relations[0].relation,
                             "CONDITION_RESULT")

    def test_source_rejects_mutation_and_never_exposes_gold_as_raw_input(self):
        m=train_only_manifest()
        self.assertEqual(len(m["source_sha256"]),64)
        changed=replace(self.rows[0],text=self.rows[0].text+" х")
        with self.assertRaises(ValueError):
            validate_graph(changed)
        self.assertEqual(
            tuple(type(x).__name__ for x in raw_text_only_train_input()),
            ("str",)*96)
        import research.nra3_s20.s20_new96_synthetic_graph_train_source as src
        code=inspect.getsource(src)
        self.assertNotIn("optimizer.step(",code)
        self.assertNotIn("EVAL32", " ".join(src.__dict__.keys()))


if __name__=="__main__":
    unittest.main()
