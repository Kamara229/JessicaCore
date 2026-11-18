"""S60 source-restricted original Calibri tokenizer BPE offsets, gold-blind.

Only precommitted S60 32 plaintext rows may be aligned; human-gold source
is NEVER imported here. Native tokenizer, IDs and offsets are unchanged.
All original byte-level duplicate Unicode offsets and shared-BPE raw
word spans are handled by the previously physically verified S51
fractional raw-text overlap logic. Shared BPE positions stay UNSAFE.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

from research.nra3_s20.s51_fractional_native_bpe_overlap import (
    NativeOverlapAlignment,align_offsets_from_raw_text,require,
)
from research.nra3_s20.s60_new32_precommitted_raw_only import (
    raw_only_inputs,
)

@dataclass(frozen=True)
class CausalWord53:
    last_token:int
    token_positions:tuple[int,...]

@dataclass(frozen=True)
class OriginalNativeS60:
    original_prompt_sha256:str
    tokenizer_token_count:int
    complete_prefix_position:int
    independent_eval_labels_loaded:bool
    word_alignment:tuple[CausalWord53,...]
    fractional:NativeOverlapAlignment


def align_s60_real_native(tokenizer,text,*,expected_ids=None):
    require(tokenizer is not None and isinstance(text,str)
            and text in set(raw_only_inputs()),
            "S60 only exact separately precommitted raw32 TRAIN-development text")
    try:
        original_chat=tokenizer.apply_chat_template(
            [{"role":"user","content":text}],tokenize=False,
            add_generation_prompt=True)
        require(isinstance(original_chat,str)
                and original_chat.count(text)==1,
                "original native chat template does not contain exact source text once")
        plain_start=original_chat.index(text)
        ids=tuple(int(z) for z in tokenizer.encode(
            original_chat,add_special_tokens=False))
        o=tokenizer(original_chat,add_special_tokens=False,
                    return_offsets_mapping=True)
        raw_ids=tuple(int(v) for v in o["input_ids"])
        offsets=tuple((int(a),int(b)) for a,b in o["offset_mapping"])
    except Exception as e:
        raise ValueError("S60 original unchanged CPU Calibri fast-tokenizer offsets missing") from e
    require(ids==raw_ids and (expected_ids is None
                             or ids==tuple(int(z) for z in expected_ids)),
            "S60 original genuine CPU Calibri prompt token IDs differ")
    fractions=align_offsets_from_raw_text(
        text,ids,offsets,plain_start,len(original_chat))
    word_mapping=tuple(CausalWord53(max(w.native_token_positions),
                                    w.native_token_positions)
                       for w in fractions.raw_words)
    require(len(word_mapping)==fractions.words_recovered
            and all(row.last_token>=0 for row in word_mapping),
            "S60 raw word/native original wordpiece fractional BPE missing")
    return OriginalNativeS60(
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        len(ids),len(ids)-1,False,word_mapping,fractions)


def align_s60_fractional_native(tokenizer,text,*,expected_ids=None):
    return align_s60_real_native(
        tokenizer,text,expected_ids=expected_ids).fractional
