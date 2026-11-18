"""S64 GOLD-BLIND native tokenizer prefix boundary stability probe.

Checks ORIGINAL, UNMODIFIED Calibri43M fast tokenizer IDs and offsets
for source text char-prefixes at each S18 word end, comparing to exactly
those native pieces fully contained in the corresponding prefix of the
FULL original templated user message. Reports a diagnostic only:

  - A tokenizer piece shared across two source words is UNSAFE;
  - A tokenizer prefix-stability mismatch is UNSAFE for this probe;
  - Equal token IDs before each boundary is only a *necessary local*
    condition for reproducible incremental tokenization; **not sufficient**
    to certify original hidden320 / GT full-context causal streaming.

Never normalize, repair, retokenize, substitute or patch the genuine
model inputs, original BPE IDs or protected 616 J7 computation. No gold,
no target labels, no training, no weight changes.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s51_fractional_native_bpe_overlap import (
    align_offsets_from_raw_text,
)

def require(ok,why):
    if not bool(ok):raise ValueError("S64 source-only genuine tokenizer prefix audit: "+why)

def native_chat(tokenizer,text):
    template=tokenizer.apply_chat_template(
        [{"role":"user","content":text}],
        tokenize=False,add_generation_prompt=True,
    )
    require(isinstance(template,str) and template.count(text)==1,
            "original unchanged chat template source occurrence changed")
    start=template.index(text)
    full=tokenizer(template,add_special_tokens=False,return_offsets_mapping=True)
    ids=tuple(int(z) for z in full["input_ids"])
    offs=tuple((int(a),int(b)) for a,b in full["offset_mapping"])
    require(len(ids)==len(offs),"original fast tokenizer offset ABI")
    return template,start,ids,offs

def _native_source_pieces(ids,offs,begin,end):
    # Never count chat tokens or a native BPE piece crossing word boundary.
    return tuple((int(ids[j]),a-begin,b-begin)
                 for j,(a,b) in enumerate(offs)
                 if a>=begin and b<=begin+end and b>a)

def source_only_prefix_audit(tokenizer,text,*,expected_ids=None):
    """Audit all source-token boundaries, with no gold/predicate inputs."""
    require(isinstance(text,str) and len(text)>0 and tokenizer is not None,
            "true original tokenizer/text required")
    templ,start,ids,offs=native_chat(tokenizer,text)
    if expected_ids is not None:
        require(ids==tuple(int(z) for z in expected_ids),
                "original protected CPU model BPE IDs not matched")
    whole=align_offsets_from_raw_text(
        text,ids,offs,start,len(templ))
    tokens=raw_surface_grammar(text)
    require(len(tokens)==len(whole.raw_words),"original word alignment mismatch")
    per=[]
    for t,w in zip(tokens,whole.raw_words):
        # Same original chat format, but source truncation is only a
        # diagnostic tokenizer probe: it NEVER enters model forward.
        ptext=text[:t.end]
        ttemplate,pstart,pids,poffs=native_chat(tokenizer,ptext)
        original=_native_source_pieces(ids,offs,start,t.end)
        prefix=_native_source_pieces(pids,poffs,pstart,t.end)
        # Words that have a token extending into the next word already
        # fail strict per-word native causal safety.
        prefix_stable=(prefix==original)
        per.append({
            "source_word_index":t.index-1,
            "raw_start":t.start,"raw_end":t.end,
            "strict_shared_BPE_word_safe":bool(w.causally_safe),
            "prefix_native_token_piece_identical":prefix_stable,
            "source_native_token_piece_count_in_full_prefix":len(original),
            "source_native_token_piece_count_in_prefix_only":len(prefix),
            "original_native_word_positions":list(w.native_token_positions),
            "original_word_uses_shared_native_piece":bool(w.shared_native_tokens),
        })
    assert len(per)==whole.words_recovered
    return {
        "source_text_sha256":whole.original_plaintext_sha256,
        "source_word_count":len(per),
        "full_original_native_BPE_token_count":len(ids),
        "strict_shared_native_words_unsafe":sum(
            not z["strict_shared_BPE_word_safe"] for z in per),
        "prefix_token_identifier_mismatch_words":sum(
            not z["prefix_native_token_piece_identical"] for z in per),
        "strict_word_masks_and_all_prefix_tokens_stable":all(
            z["strict_shared_BPE_word_safe"] and
            z["prefix_native_token_piece_identical"] for z in per),
        "native_source_ids_and_original_tokenizer_untouched":True,
        "no_input_weight_or_gold_mutation":True,
        "prefix_BPE_stability_does_NOT_prove_original_hidden_or_GT_causal_streaming":True,
        "source_word_details":per,
    }
