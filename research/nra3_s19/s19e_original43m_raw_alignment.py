"""S19e: gold-blind alignment of 9 new SHA-locked human TRAIN sentences.

This tokenizer helper receives *plain text*, NEVER a graph, parse or labels.
It fails closed on original model chat template / BPE word boundary changes.
"""
from __future__ import annotations

import hashlib

from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord, CausalAlignment,
)
from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s19.train_only_event_graph_source import (
    TRAIN_ONLY_FROZEN_SHA256, build_s19_train_only,
    train_only_manifest,
)


def s19_train_texts() -> tuple[str, ...]:
    manifest = train_only_manifest()
    rows = build_s19_train_only()
    if (
        manifest["new_train_annotations_sha256"] != TRAIN_ONLY_FROZEN_SHA256
        or manifest["new_train_examples"] != 9
        or len(rows) != 9
        or len({row.text for row in rows}) != 9
    ):
        raise ValueError("S19e rejects unsealed or duplicate source-only TRAIN")
    return tuple(row.text for row in rows)


def align_s19e_raw_train9(tokenizer, text: str, *, expected_ids=None) -> CausalAlignment:
    if tokenizer is None or text not in set(s19_train_texts()):
        raise ValueError("S19e accepts only exact sealed TRAIN9 plaintext")
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
        raise ValueError("S19e original fast tokenizer offsets unavailable") from ex
    if (
        full.count(text) != 1
        or not ids
        or len(offsets) != len(ids)
        or output_ids != ids
        or (expected_ids is not None and ids != tuple(expected_ids))
        or any(a < 0 or b < a or b > len(full) for a, b in offsets)
    ):
        raise ValueError("S19e original Calibri prompt/token IDs drift")
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
            raise ValueError("S19e native BPE overlaps two raw words")
        used.update(positions)
        mapped.append(AlignedWord(token.index, positions, max(positions)))
    if (
        len(mapped) != len(raw.tokens)
        or any(mapped[i].last_token >= mapped[i+1].last_token
               for i in range(len(mapped)-1))
        or mapped[-1].last_token >= len(ids)
    ):
        raise ValueError("S19e original43M native word positions invalid")
    return CausalAlignment(
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        len(ids), ids, tuple(mapped), len(ids)-1, False,
    )
