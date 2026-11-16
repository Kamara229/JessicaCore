"""S18c3 raw-Russian grammar cues, input-only (not gold or POS oracle).

A deliberately small, inexpensive starting point to distinguish common
case/voice cues in Russian: short-passive suffixes, past copula, likely
instrumental and dative endings, overt 'от' source preposition.
Used ONLY in research / no model inference activation, no weights, no
trained encoder, no official POS/UD predictions and NO accuracy claim.

Input must be exactly RAW text; annotations, source TRAIN row, UPOS,
dependency heads, semantic roles and lemma are forbidden as inputs.
Text source indexing and punctuation are derived by plain Unicode
regex, not from manually annotated TRAIN token spans.
S18 source below is inspected ONLY in independent *test* code.
Do not claim general Russian morphological tagging: cues are fallible.
Strict online role inference is not established by previous S95 full-
sentence original Calibri feature captures.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

# Each bit is an observed surface indicator, NOT inferred gold UPOS/case/role.
FEATURES = (
    "initial_uppercase",
    "all_lowercase",
    "is_punctuation",
    "preposition_other",
    "preposition_ot",
    "previous_word_is_ot",
    "ending_instrumental_possible",
    "ending_dative_possible",
    "ending_feminine_accusative_possible",
    "ending_feminine_nominative_possible",
    "past_auxiliary_byt",
    "previous_word_past_auxiliary",
    "past_verb_shape_possible",
    "short_passive_participle_shape_possible",
    "finite_future_verb_shape_possible",
    "negative_particle",
    "previous_word_negative_particle",
    "female_adjective_accusative_shape_possible",
)
DIM = len(FEATURES)
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:-[^\W_]+)*|[^\w\s]", flags=re.UNICODE)
INSTR_SUFFIX = ("ом", "ем", "ой", "ей", "ою", "ею", "ами", "ями")
DATIVE_SUFFIX = ("ам", "ям", "е", "и", "у", "ю")
PASSIVE_SUFFIX = ("ена", "ана", "ена", "ено", "аны", "ены", "ана", "яна",
                  "н", "на", "но", "ны", "т", "та", "то", "ты")
FUTURE_SUFFIX = ("ет", "ит", "ут", "ют", "ят", "ат", "ем", "им", "ете", "ите")
PAST_AUX = frozenset(("был", "была", "были", "было"))
NEGATION = frozenset(("не", "ни"))
PREPOSITION = frozenset(("от", "до", "из", "к", "ко", "у", "в", "во",
                         "на", "по", "при", "для", "без", "под", "над",
                         "с", "со", "за", "о", "об"))


@dataclass(frozen=True)
class SurfaceToken:
    form: str
    start: int
    end: int
    features: tuple[int, ...]


def raw_surface_grammar(text: str) -> tuple[SurfaceToken, ...]:
    """Compute fixed feature ABI from text only, independent of gold.

    'previous' is strictly preceding word, not the next word.
    No annotation fields can be accepted here by construction.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("S18c3 requires a nonempty plain Russian text string")
    matches = tuple(TOKEN_PATTERN.finditer(text))
    if not matches:
        raise ValueError("S18c3 raw source text has no tokens")
    result = []
    previous_word = None
    for m in matches:
        word = m.group()
        lower = word.casefold()
        has_letters = any(c.isalpha() for c in word)
        is_punct = not has_letters and not any(c.isdigit() for c in word)
        features = (
            int(word[0].isupper()),
            int(word.islower()),
            int(is_punct),
            int(lower in PREPOSITION and lower != "от"),
            int(lower == "от"),
            int(previous_word == "от"),
            int(has_letters and lower.endswith(INSTR_SUFFIX)),
            int(has_letters and lower.endswith(DATIVE_SUFFIX)),
            int(has_letters and lower.endswith(("у", "ю"))),
            int(has_letters and lower.endswith(("а", "я"))),
            int(lower in PAST_AUX),
            int(previous_word in PAST_AUX),
            int(has_letters and lower.endswith(("л", "ла", "ли", "ло"))),
            int(has_letters and lower.endswith(PASSIVE_SUFFIX)),
            int(has_letters and lower.endswith(FUTURE_SUFFIX)),
            int(lower in NEGATION),
            int(previous_word in NEGATION),
            int(has_letters and lower.endswith(("ую", "юю"))),
        )
        if len(features) != DIM or any(v not in (0, 1) for v in features):
            raise AssertionError("S18c3 raw-only binary surface ABI corruption")
        result.append(SurfaceToken(word, m.start(), m.end(), features))
        if not is_punct:
            previous_word = lower
    if any(text[t.start:t.end] != t.form or t.start >= t.end
           for t in result):
        raise AssertionError("S18c3 raw-only Unicode offsets corrupted")
    if any(result[i].end > result[i + 1].start for i in range(len(result) - 1)):
        raise AssertionError("S18c3 raw-only word boundaries overlap")
    return tuple(result)


def feature_at(token: SurfaceToken, name: str) -> int:
    """Strict immutable feature ordering; no role scores or model data."""
    if name not in FEATURES:
        raise ValueError("S18c3 unknown surface indicator: " + str(name))
    return token.features[FEATURES.index(name)]
