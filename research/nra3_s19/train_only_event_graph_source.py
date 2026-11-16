"""S19 fresh human-authored TRAIN-only semantic graph examples.

No EVAL32, EVAL128, S17c/S15c/S12c or J7 evaluation is loaded.
These annotations are supervision targets, never neural inputs or proof
of generalization. All events remain independent of word-level role labels.
"""
from __future__ import annotations

from research.nra3_s19.graph_schema import (
    Argument, Coreference, Event, EventRelation,
    canonical_train_only_sha, make_train_graph, raw_model_input,
)


def _row(text, heads, roles, refs=(), relations=()):
    tokens = raw_model_input(text).tokens
    def idx(form):
        found = [x.index for x in tokens if x.form == form]
        if len(found) != 1:
            raise ValueError("S19 author must disambiguate repeated raw word: " + form)
        return found[0]

    events = tuple(Event(i + 1, idx(head), factuality)
                   for i, (head, factuality) in enumerate(heads))
    args = tuple(Argument(event_id, idx(word), role, polarity)
                 for event_id, word, role, polarity in roles)
    coref = tuple(Coreference(idx(mention), idx(antecedent))
                  for mention, antecedent in refs)
    links = tuple(EventRelation(source, target, label)
                  for source, target, label in relations)
    return make_train_graph(text, events, args, coref, links)


def _r(event_id, word, role, polarity="POSITIVE"):
    return event_id, word, role, polarity


def build_s19_train_only():
    """Nine new TRAIN patterns; nothing is trained, predicted or evaluated."""
    return (
        # Analytic future: "будет" AUX plus infinitive predicate.
        _row("Ирина будет передавать Олегу документы.",
             (("передавать", "FUTURE_UNREALIZED"),),
             (_r(1, "Ирина", "AGENT"), _r(1, "Олегу", "RECIPIENT"),
              _r(1, "документы", "PATIENT"))),
        # Synthetic finite future with inverse order and intervening adjunct.
        _row("Документы заказчику на следующей неделе передаст Виктор.",
             (("передаст", "FUTURE_UNREALIZED"),),
             (_r(1, "Виктор", "AGENT"), _r(1, "Документы", "PATIENT"),
              _r(1, "заказчику", "RECIPIENT"))),
        # Coordinated independent finite verbs with distinct agents.
        _row("Павел проверил счёт, а Анна отправила письмо Олегу.",
             (("проверил", "ASSERTED"), ("отправила", "ASSERTED")),
             (_r(1, "Павел", "AGENT"), _r(1, "счёт", "PATIENT"),
              _r(2, "Анна", "AGENT"), _r(2, "письмо", "PATIENT"),
              _r(2, "Олегу", "RECIPIENT")),
             relations=((1, 2, "COORDINATION"),)),
        # Two future verbs share AGENT, two objects are coreferent.
        _row("Ольга подготовит договор и отправит его клиенту.",
             (("подготовит", "FUTURE_UNREALIZED"),
              ("отправит", "FUTURE_UNREALIZED")),
             (_r(1, "Ольга", "AGENT"), _r(1, "договор", "PATIENT"),
              _r(2, "Ольга", "AGENT"), _r(2, "его", "PATIENT"),
              _r(2, "клиенту", "RECIPIENT")),
             refs=(("его", "договор"),),
             relations=((1, 2, "COORDINATION"),)),
        # Condition and consequence are not asserted to have occurred.
        _row("Если Игорь подпишет договор, Мария отправит его клиенту.",
             (("подпишет", "CONDITIONAL"), ("отправит", "CONDITIONAL")),
             (_r(1, "Игорь", "AGENT"), _r(1, "договор", "PATIENT"),
              _r(2, "Мария", "AGENT"), _r(2, "его", "PATIENT"),
              _r(2, "клиенту", "RECIPIENT")),
             refs=(("его", "договор"),),
             relations=((1, 2, "CONDITION_RESULT"),)),
        # Same predicates but temporal clause is not a condition operator.
        _row("Когда Игорь подпишет договор, Мария отправит его клиенту.",
             (("подпишет", "FUTURE_UNREALIZED"),
              ("отправит", "FUTURE_UNREALIZED")),
             (_r(1, "Игорь", "AGENT"), _r(1, "договор", "PATIENT"),
              _r(2, "Мария", "AGENT"), _r(2, "его", "PATIENT"),
              _r(2, "клиенту", "RECIPIENT")),
             refs=(("его", "договор"),),
             relations=((1, 2, "TEMPORAL_SEQUENCE"),)),
        # Whole-event negation affects the SECOND event only.
        _row("Игорь подписал договор, но Мария не отправила его клиенту.",
             (("подписал", "ASSERTED"), ("отправила", "NEGATED")),
             (_r(1, "Игорь", "AGENT"), _r(1, "договор", "PATIENT"),
              _r(2, "Мария", "AGENT"), _r(2, "его", "PATIENT"),
              _r(2, "клиенту", "RECIPIENT")),
             refs=(("его", "договор"),),
             relations=((1, 2, "CONTRAST"),)),
        # Rejected role scope, not negated event.
        _row("Мария отправила договор не клиенту, а юристу.",
             (("отправила", "ASSERTED"),),
             (_r(1, "Мария", "AGENT"), _r(1, "договор", "PATIENT"),
              _r(1, "клиенту", "RECIPIENT", "REJECTED"),
              _r(1, "юристу", "RECIPIENT"))),
        # Past subordinate temporal relation and cross-clause pronoun.
        _row("Когда Мария закончила отчёт, Сергей передал его директору.",
             (("закончила", "ASSERTED"), ("передал", "ASSERTED")),
             (_r(1, "Мария", "AGENT"), _r(1, "отчёт", "PATIENT"),
              _r(2, "Сергей", "AGENT"), _r(2, "его", "PATIENT"),
              _r(2, "директору", "RECIPIENT")),
             refs=(("его", "отчёт"),),
             relations=((1, 2, "TEMPORAL_SEQUENCE"),)),
    )


def train_only_manifest():
    rows = build_s19_train_only()
    return {
        "stage": "NRA3_S19_NEW_SOURCE_ONLY_GRAMMAR_EVENT_GRAPH",
        "split": "TRAIN_ONLY",
        "new_train_examples": len(rows),
        "new_train_annotations_sha256": canonical_train_only_sha(rows),
        "gold_role_links": sum(len(x.arguments) for x in rows),
        "events": sum(len(x.events) for x in rows),
        "coreference_links": sum(len(x.coreferences) for x in rows),
        "event_relations": sum(len(x.event_relations) for x in rows),
        "evaluation_reused": False,
        "trained_or_inferred": False,
        "model_accepted": False,
        "output_gate": 0,
        "live_changed": False,
        "cpu_bpp_j7_certified": False,
    }
