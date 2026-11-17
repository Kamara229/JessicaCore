"""S22b source-only tokenizer/capture contracts; no model weight reads here."""
from __future__ import annotations

import ast
import hashlib
import inspect
import unittest

from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s20.s22b_original43m_raw_alignment import (
    s22_train_texts, align_s22_raw_train256,
)
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    manifest as source_manifest, FROZEN_S22_TRAIN256_SHA256,
)


class CharFastTokenizer:
    """Explicit synthetic offset TEST fixture, NOT actual Calibri tokenizer."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<user>" + messages[0]["content"] + "<assistant>"
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"input_ids":self.encode(text),
                "offset_mapping":[(i,i+1) for i in range(len(text))]}


class CrossBoundaryTokenizer(CharFastTokenizer):
    def encode(self,text,add_special_tokens=False):
        return [1]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=True):
        return {"input_ids":[1],"offset_mapping":[(0,len(text))]}


class GenuineCpuNew256SourceGuards(unittest.TestCase):
    def test_only_exact_sha_locked_256_train_plaintexts(self):
        rows=s22_train_texts()
        self.assertEqual(len(rows),256)
        self.assertEqual(len(set(rows)),256)
        m=source_manifest()
        self.assertEqual(m["source_sha256"],FROZEN_S22_TRAIN256_SHA256)
        self.assertEqual(m["rows_synthetic_TRAIN_ONLY"],256)
        self.assertTrue(m["no_model_forward_no_training_no_independent_EVAL"])
        self.assertTrue(m["new_true43m_GT_CPU_not_yet_captured"])
        self.assertFalse(m["candidate_accepted"])

    def test_original_word_alignment_covers_all_256_unicode_spans(self):
        fake=CharFastTokenizer()
        for text in s22_train_texts():
            a=align_s22_raw_train256(fake,text)
            raw=raw_model_input(text)
            self.assertEqual(a.original_prompt_sha256,
                             hashlib.sha256(text.encode("utf-8")).hexdigest())
            self.assertEqual(len(a.word_alignment),len(raw.tokens))
            self.assertEqual([w.word_id for w in a.word_alignment],
                             list(range(1,len(raw.tokens)+1)))
            self.assertTrue(all(w.token_positions for w in a.word_alignment))
            self.assertFalse(a.independent_eval_labels_loaded)
            self.assertEqual(a.complete_prefix_position,a.tokenizer_token_count-1)

    def test_unknown_text_and_modified_native_ids_rejected(self):
        tok=CharFastTokenizer()
        with self.assertRaises(ValueError):
            align_s22_raw_train256(tok,"Лишний текст не из TRAIN256.")
        with self.assertRaises(ValueError):
            align_s22_raw_train256(tok,s22_train_texts()[0],expected_ids=(4,5,6))

    def test_original_bpe_piece_crosses_two_words_rejected(self):
        with self.assertRaises(ValueError):
            align_s22_raw_train256(CrossBoundaryTokenizer(),s22_train_texts()[0])

    def test_original_cpu_capture_uses_only_plain_text_no_optimizer_or_eval(self):
        import research.nra3_s20.s22b_true43m_new256_cpu_capture as capture
        code=inspect.getsource(capture)
        names=[n.module or "" for n in ast.walk(ast.parse(code))
               if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(part in n for n in names
                         for part in ("s18c6","s18c7","nra3_s17",
                                      "nra3_s15","nra3_s12","s3ci")))
        self.assertNotIn("optimizer.step(",code)
        self.assertNotIn("loss.backward(",code)
        self.assertIn("with collector.capture(model,q,ids):",code)
        self.assertIn("align_s22_raw_train256(tok,q,expected_ids=ids)",code)
        self.assertIn("capture_count!=256",code)
        self.assertIn("all_616_native_CPU_generated_token_digests_unchanged",code)


if __name__=="__main__":
    unittest.main()
