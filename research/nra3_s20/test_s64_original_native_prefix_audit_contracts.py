"""S64 six synthetic source-only BPE prefix / frozen safety contracts.

Do not import S64 separately locked manual gold here. Tests are *not*
proof online GT streaming correctness or representative performance.
"""
import hashlib,re,unittest
import torch
from research.nra3_s20.s64_native_prefix_causal_boundary_audit import (
    native_chat,source_only_prefix_audit,
)
from research.nra3_s20.s64_new32_raw_only_preregistered import (
    FAMILIES,SOURCE_ROWS,raw_only_inputs,raw_sha256,
)
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import (
    raw_quote_spans,S63QuoteOuterSafe,
)
class FakeOriginalNativeTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        assert len(messages)==1 and messages[0]["role"]=="user"
        assert not tokenize and add_generation_prompt
        return "<user>"+messages[0]["content"]+"</user><assistant>"
    def __call__(self,s,*,add_special_tokens=False,return_offsets_mapping=False):
        # Synthetic tokenizer: split Unicode words, punctuation and HTML
        # sentinels. No live Calibri tokenizer, weights, or source gold.
        assert not add_special_tokens and return_offsets_mapping
        parts=list(re.finditer(r"\w+|[^\w\s]",s,flags=re.UNICODE))
        return {"input_ids":[int(hashlib.sha256(m.group().encode()).hexdigest()[:7],16)
                             for m in parts],
                "offset_mapping":[(m.start(),m.end()) for m in parts]}
    def encode(self,s,add_special_tokens=False):
        return self(s,add_special_tokens=add_special_tokens,
                    return_offsets_mapping=True)["input_ids"]

class S64Contracts(unittest.TestCase):
    def test_frozen_new32_raw_only_has_four_distinct_families(self):
        self.assertEqual(len(raw_only_inputs()),32)
        self.assertEqual([sum(row[1]==f for row in SOURCE_ROWS)
                          for f in FAMILIES],[8]*4)
        self.assertEqual(len(raw_sha256()),64)

    def test_balanced_nested_quote_and_unclosed_are_distinguished(self):
        spans,bad=raw_quote_spans('Наталья прочла «Он сказал: „Врачу не дали акт“»')
        self.assertFalse(bad)
        self.assertEqual(len(spans),2)
        self.assertTrue(raw_quote_spans('«Врачу не дали акт')[1])
        self.assertTrue(raw_quote_spans('“Курьеру не дали акт')[1])

    def test_original_tokenizer_ids_verified_gold_blind(self):
        tok=FakeOriginalNativeTokenizer()
        text="Мастер показал врачу акт."
        prompt,start,ids,offset=native_chat(tok,text)
        audit=source_only_prefix_audit(tok,text,expected_ids=ids)
        self.assertEqual(audit["full_original_native_BPE_token_count"],len(ids))
        self.assertTrue(audit["no_input_weight_or_gold_mutation"])
        self.assertFalse(audit["prefix_BPE_stability_does_NOT_prove_original_hidden_or_GT_causal_streaming"] is False)

    def test_wrong_original_model_prompt_ids_fail_closed(self):
        tok=FakeOriginalNativeTokenizer()
        with self.assertRaisesRegex(ValueError,"original protected CPU model BPE IDs"):
            source_only_prefix_audit(tok,"Врач передал акт.",expected_ids=[9,9])

    def test_all_native_words_report_explicit_prefix_booleans(self):
        report=source_only_prefix_audit(
            FakeOriginalNativeTokenizer(),"Курьер передал секретарю ведомость.")
        self.assertGreaterEqual(report["source_word_count"],4)
        self.assertEqual(len(report["source_word_details"]),
                         report["source_word_count"])
        self.assertTrue(all(isinstance(x["prefix_native_token_piece_identical"],bool)
                            and isinstance(x["strict_shared_BPE_word_safe"],bool)
                            for x in report["source_word_details"]))

    def test_existing_frozen_s63_not_activated_by_s64_audit(self):
        m=S63QuoteOuterSafe()
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.candidate_accepted)
        self.assertFalse(m.activation_allowed)
        self.assertFalse(m.streaming_causality_certified)
        z=m.fp16()
        self.assertTrue(torch.equal(z,S63QuoteOuterSafe.from_fp16(z).fp16()))

if __name__=="__main__":unittest.main()
