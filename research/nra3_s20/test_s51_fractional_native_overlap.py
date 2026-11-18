"""S51 fractional native BPE overlap, source-only structural unit tests.

Only synthetic inline text and old sealed S48 TRAIN36 plaintext.
No protected S38, previously inspected S47 manual36 or S48 HELD26.
"""
from __future__ import annotations
import unittest
import torch

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s51_fractional_native_bpe_overlap import (
    align_offsets_from_raw_text,fractional_native_word_pool,
)


def fixture(text,merge:tuple[int,int]|None=None):
    """Simulate native absolute BPE offsets from plain user prompt only."""
    tok=raw_surface_grammar(text)
    offsets=[(t.start,t.end) for t in tok]
    if merge is not None:
        a,b=merge
        if b!=a+1 or tok[a].end!=tok[b].start:
            raise ValueError("fixture can merge only adjacent raw surface tokens")
        offsets[a:b+1]=[(tok[a].start,tok[b].end)]
    ids=tuple(range(100,100+len(offsets)))
    return tok,ids,tuple(offsets)


class S51FractionalNativeBPEOffsets(unittest.TestCase):
    def test_all_disjoint_exact_mapping_remains_causal_safe(self):
        text="Инженер сказал «готово» в отчёте."
        tokens,ids,spans=fixture(text)
        mapping=align_offsets_from_raw_text(text,ids,spans,0,len(text))
        self.assertEqual(mapping.words_recovered,len(tokens))
        self.assertEqual(mapping.shared_native_token_ids,())
        self.assertEqual(mapping.ambiguous_raw_word_ids,())
        self.assertTrue(all(w.causally_safe for w in mapping.raw_words))
        self.assertTrue(all(w.native_token_overlap_weights==(1.0,)
                            for w in mapping.raw_words))
        self.assertFalse(mapping.causal_token_training_allowed)
        self.assertFalse(mapping.source_gold_loaded)
        self.assertEqual(mapping.original_duplicate_native_offset_characters,0)
        self.assertEqual(mapping.original_template_crossing_native_token_positions,())

    def test_native_bpe_crosses_opening_quote_and_first_word(self):
        text="Анна сказала «готово» вчера."
        tokens,ids,spans=fixture(text,merge=(2,3))
        self.assertEqual(tokens[2].form,"«")
        self.assertEqual(tokens[3].form,"готово")
        m=align_offsets_from_raw_text(text,ids,spans,0,len(text))
        self.assertEqual(m.words_recovered,len(tokens))
        self.assertEqual(m.invalid_or_uncovered_characters,0)
        self.assertEqual(len(m.shared_native_token_ids),1)
        self.assertEqual(m.ambiguous_raw_word_ids,(2,3))
        self.assertFalse(m.raw_words[2].causally_safe)
        self.assertFalse(m.raw_words[3].causally_safe)
        self.assertEqual(m.raw_words[2].native_token_overlap_weights,(1.0,))
        self.assertEqual(m.raw_words[3].native_token_overlap_weights,(1.0,))
        # Both words share one original causal token: NO claimed safe
        # separate next-token supervision at either word.
        hidden=torch.arange(len(ids)*320,dtype=torch.float16).reshape(len(ids),320)
        before=hidden.clone()
        out=fractional_native_word_pool(hidden,m)
        self.assertEqual(tuple(out.shape),(len(tokens),320))
        self.assertTrue(torch.equal(out[2],out[3]))
        self.assertTrue(torch.equal(hidden,before))

    def test_one_word_split_across_two_original_bpe_pieces_fractional(self):
        text="Проверяющий заявил «готово» вчера."
        tokens,ids,spans=fixture(text)
        quote=[i for i,t in enumerate(tokens) if t.form=="готово"][0]
        a,b=spans[quote]
        spans=list(spans)
        spans[quote:quote+1]=[(a,a+2),(a+2,b)]
        ids=tuple(range(400,400+len(spans)))
        m=align_offsets_from_raw_text(text,ids,tuple(spans),0,len(text))
        row=m.raw_words[quote]
        self.assertEqual(row.native_token_character_overlap,(2,len("готово")-2))
        self.assertAlmostEqual(sum(row.native_token_overlap_weights),1)
        self.assertTrue(row.causally_safe)
        self.assertTrue(row.spans_multiple_native_pieces)

    def test_native_quote_offsets_in_original_full_template_are_supported(self):
        text="Олег прочёл “документ” утром."
        toks,ids,spans=fixture(text)
        prefix="<|start|>user\n"
        after="\n<|assistant|>"
        full_len=len(prefix)+len(text)+len(after)
        real=tuple((a+len(prefix),b+len(prefix)) for a,b in spans)
        m=align_offsets_from_raw_text(text,ids,real,len(prefix),full_len)
        self.assertEqual(m.words_recovered,len(toks))
        self.assertTrue(m.original_token_ids_unchanged)

    def test_missing_char_coverage_rejected_duplicate_byte_offsets_mass_preserved(self):
        text="Он отметил «всё» вчера."
        tokens,ids,spans=fixture(text)
        faulty=list(spans)
        faulty[0]=(faulty[0][0],faulty[0][1]-1)
        with self.assertRaisesRegex(ValueError,"uncovered"):
            align_offsets_from_raw_text(text,ids,tuple(faulty),0,len(text))
        faulty=list(spans)
        faulty[1]=(faulty[0][1]-1,faulty[1][1])
        # Original byte-level BPE offsets may cover a Unicode character
        # twice. It is accepted only when every character's unit mass
        # is redistributed among all native token pieces.
        pooled=align_offsets_from_raw_text(text,ids,tuple(faulty),0,len(text))
        self.assertGreater(pooled.original_duplicate_native_offset_characters,0)
        self.assertTrue(all(abs(sum(w.native_token_overlap_weights)-1)<1e-8
                            for w in pooled.raw_words))

    def test_unicode_duplicate_native_offsets_and_template_cross_mark_unsafe(self):
        text="“да” сегодня."
        words,ids,orig=fixture(text)
        # Fast tokenizer byte-level encoding may return two pieces with
        # the exact same offset of the opening Unicode curly quote.
        offsets=[orig[0],orig[0],*orig[1:]]
        ids=tuple(range(900,900+len(offsets)))
        mapping=align_offsets_from_raw_text(text,ids,tuple(offsets),0,len(text))
        self.assertGreater(mapping.original_duplicate_native_offset_characters,0)
        self.assertAlmostEqual(mapping.raw_words[0].native_token_overlap_weights[0],0.5)
        self.assertAlmostEqual(mapping.raw_words[0].native_token_overlap_weights[1],0.5)
        self.assertTrue(mapping.raw_words[0].causally_safe)
        # Original chat template prefix may be merged into first quote
        # BPE token; it is represented but declared causally unsafe.
        beginning=10
        shifted=[(a+beginning,b+beginning) for a,b in orig]
        shifted[0]=(beginning-2,shifted[0][1])
        m=align_offsets_from_raw_text(text,tuple(range(700,700+len(shifted))),
                                      tuple(shifted),beginning,beginning+len(text)+8)
        self.assertEqual(m.original_template_crossing_native_token_positions,(0,))
        self.assertIn(0,m.ambiguous_raw_word_ids)
        self.assertFalse(m.raw_words[0].causally_safe)

    def test_no_gold_no_unsafe_weight_or_injected_gradient(self):
        text="Надпись «да» осталась."
        toks,ids,offsets=fixture(text)
        m=align_offsets_from_raw_text(text,ids,offsets,0,len(text))
        bad=torch.zeros(len(ids),320,dtype=torch.float16,requires_grad=True)
        with self.assertRaises(ValueError):
            fractional_native_word_pool(bad,m)
        wrong=torch.zeros(len(ids),319,dtype=torch.float16)
        with self.assertRaises(ValueError):
            fractional_native_word_pool(wrong,m)


if __name__=="__main__":
    unittest.main()
