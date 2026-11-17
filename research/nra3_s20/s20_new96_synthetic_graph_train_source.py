"""S20: 96 NEW, deliberately varied synthetic Russian TRAIN-only event graphs.

Twelve authored grammar patterns x eight curated noun/verb/recipient families.
All word/event/role labels are training targets, NEVER neural inputs.
This is still a SMALL SYNTHETIC source and NOT independent EVAL or proof of
natural-Russian generalization. No held-out independent test used or loaded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from research.nra3_s19.graph_schema import (
    Argument, Coreference, Event, EventRelation,
    canonical_train_only_sha, make_train_graph, raw_model_input,
)
from research.nra3_s19.train_only_event_graph_source import (
    build_s19_train_only, TRAIN_ONLY_FROZEN_SHA256,
)

S20_SOURCE_SPLIT = "NRA3_S20_96_NEW_SYNTHETIC_TRAIN_ONLY"
FAMILY_COUNT = 8
PATTERN_COUNT = 12
EXPECTED_ROWS = FAMILY_COUNT * PATTERN_COUNT


@dataclass(frozen=True)
class Family:
    male: str
    female: str
    object: str
    first_past: str
    first_future: str
    second_past: str
    second_future: str
    second_infinitive: str
    recipient: str
    alternative_recipient: str


FAMILIES = (
    Family("Игорь", "Мария", "договор", "подписал", "подпишет",
           "отправила", "отправит", "отправлять", "клиенту", "юристу"),
    Family("Павел", "Анна", "отчёт", "проверил", "проверит",
           "передала", "передаст", "передавать", "директору", "бухгалтеру"),
    Family("Сергей", "Ольга", "акт", "подготовил", "подготовит",
           "показала", "покажет", "показывать", "заказчику", "инженеру"),
    Family("Виктор", "Елена", "счёт", "согласовал", "согласует",
           "направила", "направит", "направлять", "менеджеру", "аудитору"),
    Family("Максим", "Ирина", "протокол", "утвердил", "утвердит",
           "передала", "передаст", "передавать", "секретарю", "коллеге"),
    Family("Андрей", "Наталья", "проект", "изучил", "изучит",
           "переслала", "перешлёт", "пересылать", "архитектору", "заказчику"),
    Family("Олег", "Татьяна", "план", "одобрил", "одобрит",
           "отправила", "отправит", "отправлять", "инженеру", "руководителю"),
    Family("Роман", "Светлана", "чертёж", "подписал", "подпишет",
           "передала", "передаст", "передавать", "мастеру", "начальнику"),
)

PATTERNS = (
    "asserted_coordination",
    "rejected_recipient_preposed_time",
    "negated_then_asserted_concession",
    "conditional_if_future",
    "future_temporal_when",
    "past_temporal_subordinate_first",
    "analytic_future",
    "negated_then_asserted_contrast",
    "conditional_if_subordinate_after_result",
    "past_temporal_subordinate_after_result",
    "rejected_recipient_reversed_order",
    "negated_inverted_word_order",
)


def _arg(event_id, form, role, polarity="POSITIVE"):
    return (event_id, form, role, polarity)


def _row(text, heads, arguments, refs=(), links=()):
    """Resolve exact RAW word positions for TRAIN gold, never forward inputs."""
    tokens = raw_model_input(text).tokens

    def word(form):
        positions = tuple(t.index for t in tokens if t.form == form)
        if len(positions) != 1:
            raise ValueError("S20 target must refer to exactly one raw token: " + form)
        return positions[0]

    graph = make_train_graph(
        text,
        tuple(Event(k+1, word(form), status)
              for k, (form, status) in enumerate(heads)),
        tuple(Argument(e, word(form), role, polarity)
              for e, form, role, polarity in arguments),
        tuple(Coreference(word(mention), word(antecedent))
              for mention, antecedent in refs),
        tuple(EventRelation(i, j, relation)
              for i, j, relation in links),
    )
    return graph


def build_family_pattern(f: Family, pattern: str):
    """Semantic role arcs explicitly fixed per surface template."""
    a, b, obj = f.male, f.female, f.object
    fp, ff = f.first_past, f.first_future
    sp, sf, si = f.second_past, f.second_future, f.second_infinitive
    r, alt = f.recipient, f.alternative_recipient

    # Reusable semantic arcs depend on which event comes first in text.
    first = (_arg(1, a, "AGENT"), _arg(1, obj, "PATIENT"))
    second_pronoun = (
        _arg(2, b, "AGENT"), _arg(2, "его", "PATIENT"),
        _arg(2, r, "RECIPIENT"),
    )
    send_first = (
        _arg(1, b, "AGENT"), _arg(1, obj, "PATIENT"),
        _arg(1, r, "RECIPIENT"),
    )
    sign_second = (
        _arg(2, a, "AGENT"), _arg(2, "его", "PATIENT"),
    )

    if pattern == "asserted_coordination":
        return _row(
            f"{a} {fp} {obj}, а {b} {sp} его {r}.",
            ((fp, "ASSERTED"), (sp, "ASSERTED")),
            first + second_pronoun,
            refs=(("его", obj),), links=((1, 2, "COORDINATION"),))

    if pattern == "rejected_recipient_preposed_time":
        return _row(
            f"Вчера {b} {sp} {obj} не {r}, а {alt}.",
            ((sp, "ASSERTED"),),
            (_arg(1, b, "AGENT"), _arg(1, obj, "PATIENT"),
             _arg(1, r, "RECIPIENT", "REJECTED"),
             _arg(1, alt, "RECIPIENT")))

    if pattern == "negated_then_asserted_concession":
        return _row(
            f"{b} не {sp} {obj} {r}, хотя {a} {fp} его.",
            ((sp, "NEGATED"), (fp, "ASSERTED")),
            send_first + sign_second,
            refs=(("его", obj),), links=((2, 1, "CONTRAST"),))

    if pattern == "conditional_if_future":
        return _row(
            f"Если завтра {a} {ff} {obj}, {b} {sf} его {r}.",
            ((ff, "CONDITIONAL"), (sf, "CONDITIONAL")),
            first + second_pronoun,
            refs=(("его", obj),), links=((1, 2, "CONDITION_RESULT"),))

    if pattern == "future_temporal_when":
        return _row(
            f"Когда {a} наконец {ff} {obj}, {b} {sf} его {r}.",
            ((ff, "FUTURE_UNREALIZED"), (sf, "FUTURE_UNREALIZED")),
            first + second_pronoun,
            refs=(("его", obj),), links=((1, 2, "TEMPORAL_SEQUENCE"),))

    if pattern == "past_temporal_subordinate_first":
        return _row(
            f"После того как {a} {fp} {obj}, {b} {sp} его {r}.",
            ((fp, "ASSERTED"), (sp, "ASSERTED")),
            first + second_pronoun,
            refs=(("его", obj),), links=((1, 2, "TEMPORAL_SEQUENCE"),))

    if pattern == "analytic_future":
        return _row(
            f"Завтра {b} будет {si} {obj} {r}.",
            ((si, "FUTURE_UNREALIZED"),),
            (_arg(1, b, "AGENT"), _arg(1, obj, "PATIENT"),
             _arg(1, r, "RECIPIENT")))

    if pattern == "negated_then_asserted_contrast":
        return _row(
            f"{a} не {fp} {obj}, однако {b} {sp} его {r}.",
            ((fp, "NEGATED"), (sp, "ASSERTED")),
            first + second_pronoun,
            refs=(("его", obj),), links=((1, 2, "CONTRAST"),))

    if pattern == "conditional_if_subordinate_after_result":
        return _row(
            f"{b} {sf} {obj} {r}, если {a} {ff} его.",
            ((sf, "CONDITIONAL"), (ff, "CONDITIONAL")),
            send_first + sign_second,
            refs=(("его", obj),), links=((2, 1, "CONDITION_RESULT"),))

    if pattern == "past_temporal_subordinate_after_result":
        return _row(
            f"{b} {sp} {obj} {r} после того, как {a} {fp} его.",
            ((sp, "ASSERTED"), (fp, "ASSERTED")),
            send_first + sign_second,
            refs=(("его", obj),), links=((2, 1, "TEMPORAL_SEQUENCE"),))

    if pattern == "rejected_recipient_reversed_order":
        return _row(
            f"{b} {sp} {obj} {alt}, а не {r}.",
            ((sp, "ASSERTED"),),
            (_arg(1, b, "AGENT"), _arg(1, obj, "PATIENT"),
             _arg(1, alt, "RECIPIENT"),
             _arg(1, r, "RECIPIENT", "REJECTED")))

    if pattern == "negated_inverted_word_order":
        return _row(
            f"{obj.capitalize()} {r} {b} не {sp}.",
            ((sp, "NEGATED"),),
            (_arg(1, b, "AGENT"),
             _arg(1, obj.capitalize(), "PATIENT"),
             _arg(1, r, "RECIPIENT")))

    raise ValueError("unknown S20 precommitted source-only pattern")


def build_s20_synthetic_train96():
    """All rows are TRAIN; no data source for EVAL or real-model inference."""
    rows = tuple(build_family_pattern(f, pat)
                 for f in FAMILIES for pat in PATTERNS)
    if len(rows) != EXPECTED_ROWS:
        raise AssertionError("S20 exact 8x12 synthetic TRAIN source changed")
    if len({g.text for g in rows}) != len(rows):
        raise AssertionError("S20 duplicate TRAIN surface text")
    previous = {g.text for g in build_s19_train_only()}
    if {g.text for g in rows}.intersection(previous):
        raise AssertionError("S20 synthetic TRAIN must not repeat old S19 nine text")
    return rows


def group_rows():
    """TRAIN-internal grouping; ensure syntax and lexical family coverage."""
    out = {}
    for i, g in enumerate(build_s20_synthetic_train96()):
        fam, pat = divmod(i, PATTERN_COUNT)
        out[(fam, pat)] = g
    if len(out) != EXPECTED_ROWS:
        raise AssertionError("S20 full cross product missing")
    return out


def train_only_split_contract():
    """Predeclared no-leakage folds; NOT a pristine independent EVAL."""
    n = EXPECTED_ROWS
    folds = []
    for fam in range(FAMILY_COUNT):
        held = tuple(i for i in range(n) if i // PATTERN_COUNT == fam)
        train = tuple(i for i in range(n) if i // PATTERN_COUNT != fam)
        folds.append(("LEXICAL", fam, train, held))
    for pat in range(PATTERN_COUNT):
        held = tuple(i for i in range(n) if i % PATTERN_COUNT == pat)
        train = tuple(i for i in range(n) if i % PATTERN_COUNT != pat)
        folds.append(("CONSTRUCTION", pat, train, held))
    if len(folds) != FAMILY_COUNT + PATTERN_COUNT:
        raise AssertionError("S20 strict cross-validation grouping changed")
    for _, _, tr, held in folds:
        if set(tr) & set(held) or set(tr) | set(held) != set(range(n)):
            raise AssertionError("S20 TRAIN-internal fold leakage")
    return tuple(folds)


def raw_text_only_train_input():
    """Physical feature capture MUST call ONLY this plain-text interface."""
    return tuple(g.text for g in build_s20_synthetic_train96())


def train_only_manifest():
    rows = build_s20_synthetic_train96()
    categories = {}
    rel = {}
    for row in rows:
        for e in row.events:
            categories[e.factuality] = categories.get(e.factuality, 0) + 1
        for link in row.event_relations:
            rel[link.relation] = rel.get(link.relation, 0) + 1
    sha = canonical_train_only_sha(rows)
    return {
        "stage": S20_SOURCE_SPLIT,
        "source_sha256": sha,
        "train_only_sentences": len(rows),
        "curated_lexical_families": FAMILY_COUNT,
        "authored_syntax_patterns": PATTERN_COUNT,
        "synthetic_templates_not_independent_natural_Russian": True,
        "gold_events": sum(len(g.events) for g in rows),
        "gold_directed_role_edges": sum(len(g.arguments) for g in rows),
        "gold_rejected_recipients": sum(
            a.polarity == "REJECTED" for g in rows for a in g.arguments),
        "gold_coreference_edges": sum(len(g.coreferences) for g in rows),
        "gold_event_relation_edges": sum(len(g.event_relations) for g in rows),
        "factuality_class_counts": categories,
        "event_relation_class_counts": rel,
        "strict_grouped_TRAIN_internal_folds": len(train_only_split_contract()),
        "original_s19_source_SHA_preserved": TRAIN_ONLY_FROZEN_SHA256,
        "all_human_graph_annotations_target_only": True,
        "original43M_GT_hidden_for_new96_NOT_YET_CAPTURED": True,
        "model_inference_or_train_performed": False,
        "independent_EVAL_accessed": False,
        "model_accepted": False,
        "output_gate": 0,
        "LIVE_changed": False,
    }
