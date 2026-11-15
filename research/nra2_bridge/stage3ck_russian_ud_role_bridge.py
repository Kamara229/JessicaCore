"""NRA2 S3CK: Russian UD grammar bridge, OFFLINE CONTRACT ONLY.

This component accepts external, gold-free Universal Dependencies (UD)
annotations. It does NOT perform Russian POS tagging, dependency parsing,
coreference resolution, or train weights by itself. Its job is to convert
validated morphological/dependency annotations into stable, explicit
grammatical feature links for a future neural adapter.

Semantic source/target and graph reachability are deliberately NOT returned:
grammatical subject is not automatically graph source (e.g. "R1 получил от R0").
At inference the annotations must be predicted from user text, never supplied
from graph truth/labels/held-out EVAL metadata.

This file never imports torch, touches frozen Calibri/GT/NRA/J7 checkpoints,
loads TRAIN/DEV/EVAL, changes output gates, or contacts LIVE/Jessica/Render.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

UD_POS = frozenset({
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
    "VERB", "X",
})
NOMINAL_POS = frozenset({"NOUN", "PROPN", "PRON"})
ENTITY = re.compile(r"R[0-5]")
FEATURE = re.compile(r"[A-Z][A-Za-z0-9]*=[A-Za-z0-9]+")


class GrammarContractError(ValueError):
    """Fail closed on malformed, ambiguous or untrusted annotation."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise GrammarContractError("S3CK " + reason)


@dataclass(frozen=True)
class UDWord:
    word_id: int
    text: str
    lemma: str
    upos: str
    head: int
    deprel: str
    start: int
    end: int
    feats: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityMention:
    """Entity identity from TEXT-ONLY mention alignment; never src/dst gold."""
    entity: str
    word_id: int


@dataclass(frozen=True)
class GrammarArc:
    head_id: int
    child_id: int
    relation: str


@dataclass(frozen=True)
class GrammarRole:
    """Syntactic role, not directional or semantic source/target."""
    role: str
    predicate_id: int
    word_id: int
    entity: str | None


@dataclass(frozen=True)
class GrammarFrame:
    predicate_id: int
    predicate_lemma: str
    words: tuple[UDWord, ...]
    arcs: tuple[GrammarArc, ...]
    roles: tuple[GrammarRole, ...]
    unresolved_pronouns: tuple[int, ...]
    semantic_direction: None = None
    graph_reachability: None = None

    def role_entities(self, role: str) -> tuple[str, ...]:
        return tuple(x.entity for x in self.roles
                     if x.role == role and x.entity is not None)


def _validate_sentence(text: str, words: tuple[UDWord, ...]) -> int:
    require(isinstance(text, str) and bool(text.strip())
            and unicodedata.normalize("NFC", text) == text,
            "nonempty NFC text required")
    require(isinstance(words, tuple) and len(words) >= 2,
            "one annotated sentence and >=2 words required")
    require(all(isinstance(w, UDWord) for w in words),
            "UDWord objects required")
    ids = [w.word_id for w in words]
    require(ids == list(range(1, len(words) + 1)),
            "UD word IDs must be unique, contiguous and ordered")
    roots = [w.word_id for w in words if w.head == 0]
    require(len(roots) == 1, "exactly one UD root required")
    previous_end = 0
    for w in words:
        require(w.upos in UD_POS and isinstance(w.lemma, str)
                and bool(w.lemma.strip()) and isinstance(w.deprel, str)
                and bool(w.deprel) and isinstance(w.text, str)
                and bool(w.text), "invalid UPOS/lemma/deprel/word")
        require(isinstance(w.start, int) and isinstance(w.end, int)
                and previous_end <= w.start < w.end <= len(text)
                and text[w.start:w.end] == w.text,
                "invalid original-text character offsets")
        previous_end = w.end
        require(isinstance(w.head, int)
                and 0 <= w.head <= len(words)
                and w.head != w.word_id,
                "invalid dependency head")
        require((w.head == 0) == (w.deprel == "root"),
                "root relation must match head zero")
        require(isinstance(w.feats, tuple)
                and len(set(w.feats)) == len(w.feats)
                and all(isinstance(f, str) and FEATURE.fullmatch(f)
                        for f in w.feats), "invalid UD morphological feats")
    # All nodes must reach the one root, never a cycle.
    for word_id in ids:
        seen = set()
        cursor = word_id
        while cursor:
            require(cursor not in seen, "cycle in UD tree")
            seen.add(cursor)
            cursor = words[cursor - 1].head
    root = roots[0]
    require(words[root - 1].upos == "VERB",
            "first prototype requires a verbal root; otherwise abstain")
    require(sum(w.upos == "VERB" for w in words) == 1,
            "multiple verbal predicates require a later clause-aware model")
    return root


def extract_grammar(
    text: str,
    words: tuple[UDWord, ...],
    mentions: tuple[EntityMention, ...] = (),
    *,
    language: str = "ru",
) -> GrammarFrame:
    """Build grammatical roles/links from externally predicted UD.

    Fail closed instead of guessing a missing parser output or anaphora.
    This method cannot infer a graph edge/reachability or true semantic
    direction. Offsets must refer to original user text, never to a gold row.
    """
    require(language == "ru", "Russian-only first prototype")
    root = _validate_sentence(text, words)
    require(isinstance(mentions, tuple), "entity mention tuple required")
    by_word: dict[int, str] = {}
    for m in mentions:
        require(isinstance(m, EntityMention)
                and isinstance(m.entity, str)
                and ENTITY.fullmatch(m.entity) is not None
                and isinstance(m.word_id, int)
                and 1 <= m.word_id <= len(words),
                "invalid text-only entity mention")
        w = words[m.word_id - 1]
        require(w.upos in NOMINAL_POS and w.upos != "PRON",
                "pronoun coreference must remain explicitly unresolved")
        require(m.word_id not in by_word,
                "one surface mention cannot represent two entities")
        by_word[m.word_id] = m.entity
    arcs = tuple(GrammarArc(w.head, w.word_id, w.deprel)
                 for w in words if w.head != 0)

    # UD grammatical labels are not semantic source/target. For a passive,
    # the subject is the patient. For an active, it is a subject (not agent).
    passive = any(w.deprel == "nsubj:pass" and w.head == root
                  for w in words) or "Voice=Pass" in words[root - 1].feats
    role_map = {
        "nsubj": "passive_subject" if passive else "subject",
        "nsubj:pass": "passive_subject",
        "obj": "direct_object",
        "iobj": "indirect_object",
        "obl:agent": "oblique_agent",
    }
    roles = []
    for w in words:
        if w.head == root and w.deprel in role_map:
            require(w.upos in NOMINAL_POS,
                    "argument role must attach to a noun/proper noun/pronoun")
            roles.append(GrammarRole(
                role_map[w.deprel], root, w.word_id, by_word.get(w.word_id)))
    # A single predicate should not have two distinct grammatical subjects
    # or direct objects; such parses are rejected rather than silently used.
    for kind in ("subject", "passive_subject", "direct_object",
                 "indirect_object", "oblique_agent"):
        require(sum(r.role == kind for r in roles) <= 1,
                "ambiguous repeated core grammatical role")
    unresolved = tuple(w.word_id for w in words
                       if w.upos == "PRON" and w.head == root
                       and w.deprel in role_map)
    return GrammarFrame(
        root, words[root - 1].lemma, words, arcs, tuple(roles), unresolved,
    )
