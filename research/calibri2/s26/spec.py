"""Immutable S26 lexical composition specification (no QA templates/answers here).

Four *new* nouns have never appeared in earlier S20 or S24 training.
For each new noun, two color combinations enter TRAIN and two remain
unseen. The eight reserved colors/nouns are the precommitted S26
independent 96-item novel-binding benchmark. Other older words may
be replayed by training, but these eight combinations MUST NOT be
included as either answer or foil.
"""
from __future__ import annotations

COLORS = ("red", "blue", "green", "yellow")
OLD_NOUNS = ("cup", "book", "bag", "lamp", "chair", "hat",
             "plate", "scarf", "clock", "spoon", "pencil", "bottle")
EXTRA_NOUNS = ("fork", "glove", "shirt", "key")
ALL_NOUNS = OLD_NOUNS + EXTRA_NOUNS

# Novel combinations NEVER seen in S20/S24; withheld from S26 training.
RESERVED_BY_NOUN = {
    "fork": ("red", "green"),
    "glove": ("blue", "yellow"),
    "shirt": ("red", "yellow"),
    "key": ("blue", "green"),
}
RESERVED = tuple((color, noun) for noun in EXTRA_NOUNS
                 for color in RESERVED_BY_NOUN[noun])
assert len(RESERVED) == len(set(RESERVED)) == 8
assert set(RESERVED_BY_NOUN) == set(EXTRA_NOUNS)
assert all(len(v) == 2 and len(set(v)) == 2 for v in RESERVED_BY_NOUN.values())


def allowed_combos():
    pairs=[(color, noun) for color in COLORS for noun in ALL_NOUNS
           if (color, noun) not in RESERVED]
    assert len(pairs) == len(COLORS) * len(ALL_NOUNS) - len(RESERVED) == 56
    assert not set(pairs).intersection(RESERVED)
    assert all(sum(x[1] == noun for x in pairs) == 2 for noun in EXTRA_NOUNS)
    return pairs
