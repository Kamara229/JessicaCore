"""CALIBRI S181: immutable research-only lexical shadow parser for S30/S54.

No model inference, TRAIN answer lookup, token IDs, thresholds, routing,
replacement of frozen decoding, or Jessica/LIVE writes. Parse the literal
question in isolation and return a possible two-word phrase or abstain.
Safety: match full original TRAIN grammar; repeated owner fields must agree,
owners and values must be distinct, both values same-color two-word phrases.
This is a shadow hypothesis, NEVER an approved enhancement.
"""
from __future__ import annotations
import re

# Pinned literal source templates from S30.make_training_pairs and
# S54.NEW_TRAIN_FRAMES. Unit test checks byte-for-byte equality against
# those sources using Python AST without importing neural dependencies.
S30_TEMPLATES = (
    "TRAIN item {i}: {a} holds a {left}; {b} holds a {right}. State {a}'s color and object in two words.",
    "TRAIN sample {i}: In a storeroom {a} receives the {left}, while {b} receives the {right}. Name {a}'s object.",
    "TRAIN card {i}: {b} = {right}; {a} = {left}. What belongs to {a}? Give the two-word item.",
    "TRAIN report {i}: {a} chose {left}. Someone called {b} chose {right}. Reply with {a}'s two-word choice.",
)
S54_TEMPLATES = (
    "TRAIN expansion {i}: Codebook {a} => {left}; codebook {b} => {right}. Emit the two-word code at {a}.",
    "TRAIN exercise {i}: Reorder by labels. {b}={right}, and {a}={left}. Only write the two words paired with {a}.",
    "TRAIN contrast {i}: {a} points to {left} || {b} points to {right}. Transcribe {a}'s color then noun.",
    "TRAIN diagram {i}: {b} marks {right}; by comparison {a} marks {left}. Output the two-word marking for {a}.",
    "TRAIN record {i}: Under header {a}: {left}. Under {b}: {right}. Return the two words for header {a}.",
    "TRAIN routing {i}: A={a}=>{left}; B={b}=>{right}. What two-word code should {a} emit?",
    "TRAIN cross-check {i}: Read {b} ({right}), read {a} ({left}). Which exact two words label {a}?",
    "TRAIN classification {i}: {a} maps onto {left}; {b} maps onto {right}. State {a}'s color-and-noun result.",
)
_FIELDS = re.compile(r"\{(i|a|b|left|right)\}")
_VALUE = {"i": r"\d{1,6}",
          "a": r"[A-Za-z][A-Za-z0-9]*",
          "b": r"[A-Za-z][A-Za-z0-9]*",
          "left": r"[a-z]+ [a-z]+",
          "right": r"[a-z]+ [a-z]+"}


def _compile(template: str) -> re.Pattern:
    segments, seen, start = [], set(), 0
    for match in _FIELDS.finditer(template):
        segments.append(re.escape(template[start:match.start()]))
        name = match.group(1)
        if name in seen:
            segments.append(f"(?P={name})")
        else:
            segments.append(f"(?P<{name}>{_VALUE[name]})")
            seen.add(name)
        start = match.end()
    segments.append(re.escape(template[start:]))
    if set(_VALUE) - seen:
        raise AssertionError("S181 incomplete literal source grammar")
    return re.compile("^" + "".join(segments) + "$")


PATTERNS = tuple((style, n, _compile(t))
                 for style, ts in (("S30", S30_TEMPLATES),
                                   ("S54", S54_TEMPLATES))
                 for n, t in enumerate(ts))


def shadow_candidate(prompt: str) -> str | None:
    """Only literal source evidence, no trained parameters or gold labels."""
    if not isinstance(prompt, str) or len(prompt) > 1024:
        return None
    choices = []
    for _style, _index, pattern in PATTERNS:
        match = pattern.fullmatch(prompt)
        if match is None:
            continue
        g = match.groupdict()
        if g["a"] == g["b"] or g["left"] == g["right"]:
            continue
        lc, ln = g["left"].split()
        rc, rn = g["right"].split()
        if lc != rc or ln == rn:
            continue
        # Every supported original S30/S54 template explicitly requests a.
        choices.append(g["left"])
    return choices[0] if len(choices) == 1 else None


def shadow_audit(questions: list[str]) -> dict:
    """No gold expected outputs are accepted by this function."""
    from collections import Counter
    result = Counter()
    for question in questions:
        candidate = shadow_candidate(question)
        result["abstained" if candidate is None else "candidate_emitted"] += 1
    return {
        "stage": "S181_S30_S54_SHADOW_PROMPT_ONLY",
        "total": len(questions),
        **dict(result),
        "no_gold_at_inference": True,
        "no_model_or_selector_changes": True,
        "not_accepted": True,
    }
