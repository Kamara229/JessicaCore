"""S19 research-only typed event graph; strict separation of raw inputs from TRAIN gold.

NO model inference, training, learned POS/UD, official evaluation, LIVE, or
deployment. Graphs are human-annotated TRAIN targets, NEVER model features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar

SPLIT = "NRA3_S19_NEW_HUMAN_AUTHORED_TRAIN_ONLY_NOT_EVAL"
ROLES = frozenset(("AGENT", "PATIENT", "RECIPIENT", "SOURCE",
                   "DESTINATION", "EXPERIENCER", "INSTRUMENT", "LOCATION"))
FACTUALITY = frozenset(("ASSERTED", "NEGATED", "FUTURE_UNREALIZED",
                        "CONDITIONAL", "UNKNOWN"))
POLARITY = frozenset(("POSITIVE", "REJECTED"))
RELATIONS = frozenset(("COORDINATION", "CONDITION_RESULT",
                       "TEMPORAL_SEQUENCE", "CONTRAST"))


@dataclass(frozen=True)
class RawToken:
    index: int
    form: str
    start: int
    end: int


@dataclass(frozen=True)
class RawModelInput:
    """Only this text-and-offset type is permitted as input to future models."""
    text: str
    tokens: tuple[RawToken, ...]


@dataclass(frozen=True)
class Event:
    event_id: int
    predicate: int  # one-based raw token position
    factuality: str  # event status, not a participant role


@dataclass(frozen=True)
class Argument:
    event_id: int
    token: int
    role: str
    polarity: str = "POSITIVE"  # REJECTED applies to this role, not the event


@dataclass(frozen=True)
class Coreference:
    mention: int
    antecedent: int


@dataclass(frozen=True)
class EventRelation:
    source_event: int
    target_event: int
    relation: str


@dataclass(frozen=True)
class TrainOnlyGoldGraph:
    """Human targets only: NEVER pass this object to a neural feature function."""
    text: str
    tokens: tuple[RawToken, ...]
    events: tuple[Event, ...]
    arguments: tuple[Argument, ...]
    coreferences: tuple[Coreference, ...] = ()
    event_relations: tuple[EventRelation, ...] = ()
    split: str = SPLIT
    output_gate: int = 0
    candidate_accepted: bool = False


def raw_model_input(text: str) -> RawModelInput:
    """Gold-blind, exact UTF-8/Unicode word offsets; does not receive annotations."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("S19 expects nonempty plain text")
    items = raw_surface_grammar(text)
    return RawModelInput(text, tuple(
        RawToken(i, item.form, item.start, item.end)
        for i, item in enumerate(items, start=1)))


def validate_graph(g: TrainOnlyGoldGraph) -> None:
    """Reject ambiguous/inconsistent human TRAIN graphs; never score a model."""
    if not isinstance(g, TrainOnlyGoldGraph):
        raise TypeError("S19 accepts only a typed TRAIN-only gold graph")
    if g.split != SPLIT or g.output_gate != 0 or g.candidate_accepted:
        raise ValueError("S19 cannot be EVAL or an activated candidate")
    if raw_model_input(g.text).tokens != g.tokens:
        raise ValueError("S19 token positions must come solely from raw text")
    n = len(g.tokens)
    if not g.events:
        raise ValueError("S19 requires at least one event")
    ids = set()
    heads = set()
    for e in g.events:
        if (not isinstance(e.event_id, int) or e.event_id < 1 or
                e.event_id in ids or not 1 <= e.predicate <= n or
                e.predicate in heads or e.factuality not in FACTUALITY):
            raise ValueError("S19 invalid event anchor, id or modality")
        ids.add(e.event_id)
        heads.add(e.predicate)
    if not g.arguments:
        raise ValueError("S19 requires supervised predicate-argument links")
    seen_args = set()
    for a in g.arguments:
        key = (a.event_id, a.token, a.role)
        if (a.event_id not in ids or not 1 <= a.token <= n or
                a.token in heads and a.token == next(
                    e.predicate for e in g.events if e.event_id == a.event_id) or
                a.role not in ROLES or a.polarity not in POLARITY or
                key in seen_args):
            raise ValueError("S19 invalid/contradictory role or scope")
        seen_args.add(key)
    seen_refs = set()
    for c in g.coreferences:
        if (not 1 <= c.antecedent < c.mention <= n or
                c.mention in seen_refs):
            raise ValueError("S19 non-anaphoric or duplicate coreference")
        seen_refs.add(c.mention)
    seen_rel = set()
    for rel in g.event_relations:
        key = (rel.source_event, rel.target_event, rel.relation)
        if (rel.source_event not in ids or rel.target_event not in ids or
                rel.source_event == rel.target_event or
                rel.relation not in RELATIONS or key in seen_rel):
            raise ValueError("S19 invalid/doubled event relation")
        seen_rel.add(key)
        if rel.relation == "CONDITION_RESULT":
            src = next(e for e in g.events if e.event_id == rel.source_event)
            dst = next(e for e in g.events if e.event_id == rel.target_event)
            if src.factuality != "CONDITIONAL" or dst.factuality != "CONDITIONAL":
                raise ValueError("S19 conditional dependency requires both unasserted")
    if any(a.polarity == "REJECTED" and not any(
            b.event_id == a.event_id and b.role == a.role and
            b.polarity == "POSITIVE" for b in g.arguments)
           for a in g.arguments):
        raise ValueError("S19 rejected role needs an affirmed alternative")


def make_train_graph(
    text: str,
    events: tuple[Event, ...],
    arguments: tuple[Argument, ...],
    coreferences: tuple[Coreference, ...] = (),
    event_relations: tuple[EventRelation, ...] = (),
) -> TrainOnlyGoldGraph:
    tokens = raw_model_input(text).tokens
    g = TrainOnlyGoldGraph(text, tokens, events, arguments,
                           coreferences, event_relations)
    validate_graph(g)
    return g


def canonical_train_only_sha(graphs: tuple[TrainOnlyGoldGraph, ...]) -> str:
    """SHA of all TRAIN-only gold/offsets; no heldout, predictions or weights."""
    for g in graphs:
        validate_graph(g)
    if len({g.text for g in graphs}) != len(graphs):
        raise ValueError("S19 duplicate TRAIN text")
    payload = json.dumps([asdict(g) for g in graphs], sort_keys=True,
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
