"""S22: gold-blind native tokenizer alignment for 256 SHA-sealed synthetic TRAIN sentences.

This tokenizer helper receives *plain text*, NEVER a graph, parse or labels.
It fails closed on original model chat template / BPE word boundary changes.
"""
from __future__ import annotations

from functools import lru_cache

import hashlib

from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord, CausalAlignment,
)
from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    EXPECTED, FROZEN_S22_TRAIN256_SHA256, build_s22_train256,
    manifest, text_only_inputs,
)


@lru_cache(maxsize=1)
def s22_train_texts() -> tuple[str, ...]:
    manifest = manifest()
    rows = build_s22_train256()
    if (
        manifest["source_sha256"] != FROZEN_S22_TRAIN256_SHA256
        or manifest["rows_synthetic_TRAIN_ONLY"] != EXPECTED == 256
        or len(rows) != 256
        or len({row.text for row in rows}) != 256
    ):
        raise ValueError("S22 rejects unsealed new256 synthetic TRAIN text")
    return text_only_inputs()


def align_s22_raw_train256(tokenizer, text: str, *, expected_ids=None) -> CausalAlignment:
    if tokenizer is None or text not in set(s22_train_texts()):
        raise ValueError("S22 accepts only exact sealed TRAIN256 plaintext")
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
        raise ValueError("S22 original fast tokenizer offsets unavailable") from ex
    if (
        full.count(text) != 1
        or not ids
        or len(offsets) != len(ids)
        or output_ids != ids
        or (expected_ids is not None and ids != tuple(expected_ids))
        or any(a < 0 or b < a or b > len(full) for a, b in offsets)
    ):
        raise ValueError("S22 original Calibri prompt/token IDs drift")
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
            raise ValueError("S22 native BPE overlaps two raw words")
        used.update(positions)
        mapped.append(AlignedWord(token.index, positions, max(positions)))
    if (
        len(mapped) != len(raw.tokens)
        or any(mapped[i].last_token >= mapped[i+1].last_token
               for i in range(len(mapped)-1))
        or mapped[-1].last_token >= len(ids)
    ):
        raise ValueError("S22 original43M native word positions invalid")
    return CausalAlignment(
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        len(ids), ids, tuple(mapped), len(ids)-1, False,
    )
