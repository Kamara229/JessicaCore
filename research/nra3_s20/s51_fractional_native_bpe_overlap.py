"""S51: exact native UTF-8/Unicode BPE character-overlap to raw-word bridge.

Research-only, text/input-offsets only, NO TRAIN annotations, S47/S38
or spent independent EVAL. This preserves genuine original Calibri43M
token IDs and tokenizer offsets, while allowing original BPE spans
to intersect several S18 raw Unicode tokens. Mapping is *fractional*,
not one-to-one. Original hidden320 is read only.

Critical: if a single BPE piece covers >1 raw word, its context vector
is shared. Such pooled representations are explicitly marked
CAUSALLY_UNSAFE_FOR_TOKENWISE_TRAINING; this bridge is diagnostic,
not a safe substitute for online word-at-a-time training/inference.
No assignment of fractional word gold to overlapping BPE tokens.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import torch

from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s20.s50_quote36_original43m_native_alignment import (
    s50_quote_train36_plaintexts,
)

@dataclass(frozen=True)
class NativeOverlapWord:
    index:int
    native_token_positions:tuple[int,...]
    native_token_overlap_weights:tuple[float,...]
    native_token_character_overlap:tuple[int,...]
    shared_native_tokens:tuple[int,...]
    spans_multiple_native_pieces:bool
    causally_safe:bool

@dataclass(frozen=True)
class NativeOverlapAlignment:
    original_plaintext_sha256:str
    original_token_ids:tuple[int,...]
    original_token_count:int
    raw_words:tuple[NativeOverlapWord,...]
    shared_native_token_ids:tuple[int,...]
    ambiguous_raw_word_ids:tuple[int,...]
    words_recovered:int
    invalid_or_uncovered_characters:int
    original_duplicate_native_offset_characters:int
    original_template_crossing_native_token_positions:tuple[int,...]
    source_gold_loaded:bool=False
    original_token_ids_unchanged:bool=True
    causal_token_training_allowed:bool=False


def require(ok,why):
    if not bool(ok):raise ValueError("S51 native BPE char overlap: "+why)


def align_offsets_from_raw_text(
    text:str,token_ids:tuple[int,...],absolute_offsets:tuple[tuple[int,int],...],
    plain_start:int,full_len:int,
) -> NativeOverlapAlignment:
    """Construct exact many-to-many overlap; no model or human gold input.

    Coordinates are from ORIGINAL tokenizer output; overlap token weights
    conserve each original Unicode character, even when several BPE
    byte pieces map to the same character. Returns per-word mass sums 1.
    Disallow uncovered chars. Native pieces crossing text/chat boundary
    are explicitly marked causally unsafe rather than silently dropped.
    """
    raw=raw_model_input(text)
    require(len(raw.tokens)>=3 and len(raw.tokens)<=40,
            "S48 raw quote scanner supports 3..40 words")
    require(plain_start>=0 and full_len>=plain_start+len(text),
            "original template word source is outside offset range")
    require(len(token_ids)>=2 and len(token_ids)==len(absolute_offsets)
            and all(type(q) is int and q>=0 for q in token_ids),
            "exact original BPE token IDs required")
    require(all(type(a) is int and type(b) is int and 0<=a<=b<=full_len
                for a,b in absolute_offsets),
            "native original tokenizer absolute Unicode offsets invalid")
    wordpiece=[]
    uses={}
    template_crossing=set()
    duplicate_offset_chars=0
    for tok in raw.tokens:
        lo=plain_start+tok.start
        hi=plain_start+tok.end
        require(hi>lo and text[tok.start:tok.end]==tok.form,
                "raw word Unicode source token offset changed")
        # Byte-level BPE fast tokenizers legitimately map multiple byte
        # pieces to the SAME original Unicode character (e.g. “ or «).
        # Their offset map is NOT guaranteed to be a disjoint character
        # partition. Give each character total weight ONE, distributed
        # evenly among all native pieces whose offsets cover that char.
        # Never drop, synthesize or change original native token IDs.
        owners={j:[] for j in range(lo,hi)}
        seen_pieces=[]
        overlaps=[]
        for native_j,(a,b) in enumerate(absolute_offsets):
            a1=max(lo,a)
            b1=min(hi,b)
            n=max(0,b1-a1)
            if n<=0:continue
            if a<plain_start or b>plain_start+len(text):
                # A piece crosses into original chat template; retain
                # genuine original offsets but mark causal attribution
                # unsafe, never allow next-token word supervision.
                template_crossing.add(native_j)
            for char in range(a1,b1):
                owners[char].append(native_j)
            seen_pieces.append(native_j)
            overlaps.append(n)
            uses.setdefault(native_j,[]).append(tok.index-1)
        require(bool(seen_pieces) and all(owners.values()),
                "genuine tokenizer left raw word characters uncovered")
        duplicate_offset_chars+=sum(len(v)>1 for v in owners.values())
        masses={j:0. for j in seen_pieces}
        for covers in owners.values():
            unit=1.0/len(covers)
            for j in covers:
                masses[j]+=unit
        # All original Unicode source characters contribute total 1.0
        # irrespective of byte-level duplicate offsets.
        weights=tuple(masses[j]/(hi-lo) for j in seen_pieces)
        require(abs(sum(weights)-1.0)<1e-8
                and all(w>0 for w in weights),
                "native byte-level BPE intersection mass not conserved")
        wordpiece.append((tok.index-1,tuple(seen_pieces),tuple(overlaps),weights))
    shared={j for j,words in uses.items() if len(set(words))>1}
    unsafe_native=shared|template_crossing
    words=tuple(
        NativeOverlapWord(index,positions,weights,overlaps,
                          tuple(j for j in positions if j in shared),
                          len(positions)>1,
                          not any(j in unsafe_native for j in positions))
        for index,positions,overlaps,weights in wordpiece
    )
    ambiguous=tuple(row.index for row in words if not row.causally_safe)
    require(len(words)==len(raw.tokens)
            and len({r.index for r in words})==len(words)
            and all(r.native_token_positions and len(r.native_token_positions)
                    ==len(r.native_token_overlap_weights) for r in words),
            "S51 incomplete text-only BPE to Unicode raw-word reconstruction")
    return NativeOverlapAlignment(
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_ids,len(token_ids),words,
        tuple(sorted(shared)),ambiguous,len(words),0,
        duplicate_offset_chars,tuple(sorted(template_crossing)))


def align_s51_original_tokenizer(tokenizer,text,*,expected_ids=None):
    """Call genuine unchanged Calibri tokenizer; validate exact IDs and offsets."""
    require(tokenizer is not None and text in set(s50_quote_train36_plaintexts()),
            "only frozen S48 TRAIN36 text may enter S51 original CPU mapping")
    try:
        full=tokenizer.apply_chat_template(
            [{"role":"user","content":text}],tokenize=False,
            add_generation_prompt=True)
        require(isinstance(full,str) and full.count(text)==1,
                "tokenizer altered original role/prompt text")
        start=full.index(text)
        ids=tuple(int(v) for v in tokenizer.encode(full,add_special_tokens=False))
        encoded=tokenizer(full,add_special_tokens=False,
                          return_offsets_mapping=True)
        raw_ids=tuple(int(v) for v in encoded["input_ids"])
        offsets=tuple((int(a),int(b)) for a,b in encoded["offset_mapping"])
    except Exception as e:
        raise ValueError("S51 real original tokenizer offsets unavailable") from e
    require(ids==raw_ids and (expected_ids is None
                             or ids==tuple(expected_ids)),
            "S51 original Calibri BPE token IDs changed")
    return align_offsets_from_raw_text(
        text,ids,offsets,start,len(full))


def fractional_native_word_pool(hidden320,alignment:NativeOverlapAlignment):
    """No in-place writes, no gold, output disconnected shadow word320.

    Marks shared native-token spans as causally unsafe in the returned
    immutable alignment; does NOT assert semantic quality or causality.
    """
    require(isinstance(hidden320,torch.Tensor)
            and hidden320.device.type=="cpu" and hidden320.ndim==2
            and hidden320.shape==(alignment.original_token_count,320)
            and hidden320.dtype==torch.float16
            and bool(torch.isfinite(hidden320).all())
            and not hidden320.requires_grad,
            "S51 genuine original 43M layer8 CPU hidden320 FP16 missing")
    out=[]
    for w in alignment.raw_words:
        indices=torch.tensor(w.native_token_positions,dtype=torch.long)
        weights=torch.tensor(w.native_token_overlap_weights,
                             dtype=torch.float32)
        out.append((hidden320.index_select(0,indices).float()*
                    weights[:,None]).sum(dim=0))
    return torch.stack(out).contiguous()
