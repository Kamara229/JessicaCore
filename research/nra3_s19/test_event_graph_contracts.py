"""S19 source-only unittest: never load a model, weights or spent EVAL."""
from __future__ import annotations

from dataclasses import replace
import ast
import inspect
import unittest

from research.nra3_s19.graph_schema import (
    Argument, Coreference, Event, EventRelation,
    canonical_train_only_sha, make_train_graph, raw_model_input,
    validate_graph, SPLIT,
)
from research.nra3_s19.train_only_event_graph_source import (
    build_s19_train_only, train_only_manifest,
)


class TrainOnlyEventGraphContracts(unittest.TestCase):
    def test_new_graphs_have_complete_raw_offsets_and_are_sealed_train_only(self):
        rows = build_s19_train_only()
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(set(x.text for x in rows)), 9)
        for row in rows:
            self.assertEqual(row.split, SPLIT)
            self.assertEqual(row.tokens, raw_model_input(row.text).tokens)
            self.assertEqual(row.output_gate, 0)
            self.assertFalse(row.candidate_accepted)
            validate_graph(row)
            self.assertTrue(all(row.text[t.start:t.end] == t.form for t in row.tokens))
        manifest = train_only_manifest()
        self.assertEqual(manifest["new_train_examples"], 9)
        self.assertEqual(manifest["new_train_annotations_sha256"],
                         canonical_train_only_sha(rows))
        self.assertFalse(manifest["evaluation_reused"])
        self.assertFalse(manifest["trained_or_inferred"])
        self.assertFalse(manifest["live_changed"])

    def test_conditional_and_temporal_have_same_roles_but_different_event_graph(self):
        conditional, temporal = build_s19_train_only()[4:6]
        self.assertEqual(
            [(a.event_id, a.token, a.role, a.polarity)
             for a in conditional.arguments],
            [(a.event_id, a.token, a.role, a.polarity)
             for a in temporal.arguments])
        self.assertEqual(
            [(c.mention, c.antecedent) for c in conditional.coreferences],
            [(c.mention, c.antecedent) for c in temporal.coreferences])
        self.assertEqual([e.factuality for e in conditional.events],
                         ["CONDITIONAL", "CONDITIONAL"])
        self.assertEqual([e.factuality for e in temporal.events],
                         ["FUTURE_UNREALIZED", "FUTURE_UNREALIZED"])
        self.assertEqual(conditional.event_relations[0].relation,
                         "CONDITION_RESULT")
        self.assertEqual(temporal.event_relations[0].relation,
                         "TEMPORAL_SEQUENCE")

    def test_negation_scope_and_rejected_recipient_are_different(self):
        negated = build_s19_train_only()[6]
        rejected = build_s19_train_only()[7]
        self.assertEqual([e.factuality for e in negated.events],
                         ["ASSERTED", "NEGATED"])
        self.assertEqual([a.polarity for a in negated.arguments],
                         ["POSITIVE"] * 5)
        self.assertEqual(rejected.events[0].factuality, "ASSERTED")
        rejected_roles = [(rejected.tokens[a.token-1].form,a.role,a.polarity)
                          for a in rejected.arguments if a.role=="RECIPIENT"]
        self.assertEqual(rejected_roles,
                         [("клиенту","RECIPIENT","REJECTED"),
                          ("юристу","RECIPIENT","POSITIVE")])

    def test_shared_agent_two_predicates_and_coreference_are_distinct_arcs(self):
        row = build_s19_train_only()[3]
        self.assertEqual(len(row.events), 2)
        agents = [a for a in row.arguments if a.role == "AGENT"]
        self.assertEqual(len(agents), 2)
        self.assertEqual(agents[0].token, agents[1].token)
        self.assertNotEqual(agents[0].event_id, agents[1].event_id)
        self.assertEqual(len(row.coreferences), 1)
        self.assertEqual(len(row.event_relations), 1)
        self.assertEqual(row.event_relations[0].relation, "COORDINATION")

    def test_corrupted_raw_offsets_or_eval_split_rejected(self):
        row = build_s19_train_only()[0]
        with self.assertRaises(ValueError):
            validate_graph(replace(row, text=row.text+" Другое"))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, split="EVAL32"))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, output_gate=1))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, candidate_accepted=True))

    def test_reject_self_coref_duplicate_event_and_invalid_relation(self):
        row = build_s19_train_only()[4]
        with self.assertRaises(ValueError):
            validate_graph(replace(row, coreferences=(Coreference(4, 4),)))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, events=row.events+(row.events[0],)))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, event_relations=(
                EventRelation(1, 1, "CONDITION_RESULT"),)))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, event_relations=(
                EventRelation(1, 2, "UNDECLARED"),)))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, events=(
                replace(row.events[0], factuality="ASSERTED"),row.events[1])))

    def test_rejected_recipient_requires_positive_alternative(self):
        row = build_s19_train_only()[7]
        keep = tuple(a for a in row.arguments
                     if not (a.role == "RECIPIENT" and a.polarity == "POSITIVE"))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, arguments=keep))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, arguments=row.arguments+
                                   (replace(row.arguments[0],role="BAD"),)))
        with self.assertRaises(ValueError):
            validate_graph(replace(row, arguments=row.arguments+
                                   (row.arguments[0],)))

    def test_no_spent_eval_model_or_learned_gold_inputs_imported(self):
        import research.nra3_s19.graph_schema as schema
        import research.nra3_s19.train_only_event_graph_source as source
        for module in (schema, source):
            tree = ast.parse(inspect.getsource(module))
            names = [node.module for node in ast.walk(tree)
                     if isinstance(node,ast.ImportFrom)]
            names += [alias.name for node in ast.walk(tree)
                      if isinstance(node,ast.Import)
                      for alias in node.names]
            self.assertFalse(any(
                name and ("s18c6" in name or "s18c7" in name or
                          "nra3_s17" in name or "nra3_s15" in name or
                          "nra3_s12" in name or "torch" in name)
                for name in names))
        raw = raw_model_input("Мария отправила договор.")
        self.assertFalse(hasattr(raw, "events"))
        self.assertFalse(hasattr(raw, "arguments"))
        self.assertFalse(hasattr(raw, "coreferences"))


if __name__ == "__main__":
    unittest.main()
