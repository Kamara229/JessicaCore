"""S18b0 TRAIN-only original Calibri43M fast-tokenizer/wordpiece alignment.

No original43M, GT, native CB59, S11, S16b, or heldout/EVAL is loaded.
This is a strict preflight contract for future genuine frozen CPU capture.
Only raw row.text (never labels) may enter the original tokenizer/model.
"""
from __future__ import annotations

from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord, CausalAlignment,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    TrainSentence, S18_SPLIT, S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
    build_s18_new_train96, seal_s18_new96_train,
)


def align_s18_new96_train(tokenizer, row: TrainSentence, *,
                          expected_input_ids=None) -> CausalAlignment:
    """Fail closed if annotated TRAIN word surface and real token IDs differ.

    No semantic role, POS, dependency or lemma enters the tokenizer payload.
    Passing a token across an annotated word boundary is forbidden.
    """
    registered = {r.text_sha256(): r for r in build_s18_new_train96()}
    if (
        not isinstance(row, TrainSentence)
        or row.split != S18_SPLIT
        or row != registered.get(row.text_sha256())
        or tokenizer is None
    ):
        raise ValueError("S18b0 rejects unknown/mutated/EVAL source")
    source = seal_s18_new96_train()
    if (
        source["S18_complete_original_96_new_TRAIN_full_annotation_SHA256"]
        != S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256
        or source["S18_original_source_rows"] != 96
        or source["S18_new_directed_verb_to_argument_links"] != 300
    ):
        raise ValueError("S18b0 complete 96 TRAIN source SHA/roles drift")
    # The only user-facing string passed to tokenizer is raw TRAIN text.
    try:
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": row.text}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception as exc:
        raise ValueError("S18b0 original tokenizer chat template unavailable") from exc
    if not isinstance(full, str) or full.count(row.text) != 1:
        raise ValueError("S18b0 requires exactly one raw TRAIN text in template")
    start_text = full.index(row.text)
    try:
        ids = tuple(int(x) for x in tokenizer.encode(full, add_special_tokens=False))
        data = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        returned = tuple(int(x) for x in data["input_ids"])
        offsets = tuple((int(a), int(b)) for a, b in data["offset_mapping"])
    except Exception as exc:
        raise ValueError("S18b0 true fast-tokenizer offsets required") from exc
    if (
        not ids or ids != returned or len(ids) != len(offsets)
        or (expected_input_ids is not None and tuple(expected_input_ids) != ids)
        or any(a < 0 or b < a or b > len(full) for a, b in offsets)
    ):
        raise ValueError("S18b0 true original token IDs and offsets disagree")
    mapped = []
    claimed_pieces = set()
    for word in row.words:
        begin, end = start_text + word.start, start_text + word.end
        pieces = tuple(i for i, (a, b) in enumerate(offsets)
                       if a < end and b > begin)
        if (
            not pieces or any(offsets[i][0] == offsets[i][1] for i in pieces)
            or any(i in claimed_pieces for i in pieces)
            or any(not any(offsets[i][0] <= p < offsets[i][1] for i in pieces)
                   for p in range(begin, end))
        ):
            raise ValueError("S18b0 BPE piece crosses annotated word boundary")
        claimed_pieces.update(pieces)
        mapped.append(AlignedWord(word.index, pieces, max(pieces)))
    if (
        len(mapped) != len(row.words)
        or any(mapped[i].last_token >= mapped[i + 1].last_token
               for i in range(len(mapped) - 1))
        or mapped[-1].last_token >= len(ids)
    ):
        raise ValueError("S18b0 source word ordering/tokens invalid")
    return CausalAlignment(
        row.text_sha256(), len(ids), ids, tuple(mapped), len(ids) - 1, False
    )
