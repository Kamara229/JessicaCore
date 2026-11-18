"""S50: native Calibri43M tokenizer for precommitted S48 TRAIN36 Russian quotes.

Input-only plaintext, never S48 gold char spans, S47/S38, independent EVAL.
It fails closed on original model chat template / BPE word boundary changes.
"""
from __future__ import annotations

from functools import lru_cache

import hashlib

from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord, CausalAlignment,
)
from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s20.s48_frozen_manual_train36_source import (
    MANUAL_TRAIN_MARKUP, digest as s48_manual_train_source_digest,
)

S48_TRAIN_COMMIT='95592557e95965b149ee9b7780ceba164f93a7ff'
S48_TRAIN_BLOB='ed6a76001aa1b593ddf2e7877af9ac9cbb59c416'


@lru_cache(maxsize=1)
def s50_quote_train36_plaintexts() -> tuple[str, ...]:
    # Do NOT import S48 HELD26, S47 challenge36 or protected S38.
    # Strip markup WITHOUT parsing/creating the manual quote gold spans.
    rows = tuple(s.replace('⟦','').replace('⟧','')
                 for _,_,s in MANUAL_TRAIN_MARKUP)
    if len(rows)!=36 or len(set(rows))!=36 or any(not s.strip() for s in rows):
        raise ValueError('S50 must use exact previously frozen S48 TRAIN36 plaintext')
    # Digest pins original annotated source provenance, not independent EVAL.
    if len(s48_manual_train_source_digest())!=64:
        raise ValueError('S50 frozen S48 TRAIN36 source digest unavailable')
    return rows


def align_s50_raw_train36(tokenizer, text: str, *, expected_ids=None) -> CausalAlignment:
    if tokenizer is None or text not in set(s50_quote_train36_plaintexts()):
        raise ValueError("S50 accepts only exact sealed S48 TRAIN36 quote plaintext")
    raw = raw_model_input(text)
    try:
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
        start = full.index(text)
        ids = tuple(int(i) for i in tokenizer.encode(
            full, add_special_tokens=False))
        encoded = tokenizer(
            full, add_special_tokens=False, return_offsets_mapping=True)
        output_ids = tuple(int(i) for i in encoded["input_ids"])
        offsets = tuple((int(a), int(b)) for a, b in encoded["offset_mapping"])
    except Exception as ex:
        raise ValueError("S50 original fast tokenizer offsets unavailable") from ex
    if (
        full.count(text) != 1
        or not ids
        or len(offsets) != len(ids)
        or output_ids != ids
        or (expected_ids is not None and ids != tuple(expected_ids))
        or any(a < 0 or b < a or b > len(full) for a, b in offsets)
    ):
        raise ValueError("S50 original Calibri prompt/token IDs drift")
    used = set()
    mapped = []
    for token in raw.tokens:
        lo, hi = start + token.start, start + token.end
        positions = tuple(i for i, (l, r) in enumerate(offsets)
                          if l < hi and r > lo)
        if (
            not positions
            or any(offsets[i][0] == offsets[i][1] or i in used
                   for i in positions)
            or any(not any(offsets[i][0] <= j < offsets[i][1]
                           for i in positions) for j in range(lo, hi))
        ):
            raise ValueError("S50 native BPE overlaps two raw words")
        used.update(positions)
        mapped.append(AlignedWord(token.index, positions, max(positions)))
    if (
        len(mapped) != len(raw.tokens)
        or any(mapped[i].last_token >= mapped[i+1].last_token
               for i in range(len(mapped)-1))
        or mapped[-1].last_token >= len(ids)
    ):
        raise ValueError("S50 original43M native word positions invalid")
    return CausalAlignment(
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        len(ids), ids, tuple(mapped), len(ids)-1, False,
    )
