"""S19e unit/source guards. No real model or spent EVAL downloaded here."""
from __future__ import annotations

import ast
import hashlib
import inspect
import unittest

from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s19.s19e_original43m_raw_alignment import (
    align_s19e_raw_train9, s19_train_texts,
)
from research.nra3_s19.train_only_event_graph_source import (
    TRAIN_ONLY_FROZEN_SHA256, train_only_manifest,
)


class CharacterFastTokenizer:
    """Test-only fast offset mock, NOT substituted for original CPU tokenizer."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<user>" + messages[0]["content"] + "<assistant>"
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {
            "input_ids":self.encode(text,add_special_tokens=add_special_tokens),
            "offset_mapping":[(i,i+1) for i in range(len(text))]
        }


class CrossBoundaryTokenizer(CharacterFastTokenizer):
    def encode(self,text,add_special_tokens=False):
        return [1]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=True):
        return {"input_ids":[1],"offset_mapping":[(0,len(text))]}


class S19eOriginalCpuSourceGuards(unittest.TestCase):
    def test_only_original_sha_locked_new9_train_texts(self):
        rows=s19_train_texts()
        self.assertEqual(len(rows),9)
        self.assertEqual(len(set(rows)),9)
        m=train_only_manifest()
        self.assertEqual(m["new_train_annotations_sha256"],
                         TRAIN_ONLY_FROZEN_SHA256)
        self.assertEqual(m["new_train_examples"],9)
        self.assertFalse(m["evaluation_reused"])
        self.assertFalse(m["trained_or_inferred"])

    def test_raw_utf8_tokenizer_alignment_covers_every_word(self):
        tok=CharacterFastTokenizer()
        for text in s19_train_texts():
            a=align_s19e_raw_train9(tok,text)
            self.assertEqual(a.original_prompt_sha256,
                             hashlib.sha256(text.encode("utf-8")).hexdigest())
            raw=raw_model_input(text)
            self.assertEqual(len(a.word_alignment),len(raw.tokens))
            self.assertEqual(
                [x.word_id for x in a.word_alignment],
                list(range(1,len(raw.tokens)+1)))
            self.assertTrue(all(len(x.token_positions)>0 for x in a.word_alignment))
            self.assertEqual(a.complete_prefix_position,a.tokenizer_token_count-1)
            self.assertFalse(a.independent_eval_labels_loaded)

    def test_unknown_source_or_changed_native_token_ids_fail_closed(self):
        tok=CharacterFastTokenizer()
        with self.assertRaises(ValueError):
            align_s19e_raw_train9(tok,"НЕИЗВЕСТНЫЙ ТЕКСТ.")
        with self.assertRaises(ValueError):
            align_s19e_raw_train9(tok,s19_train_texts()[0],expected_ids=(0,1,2))

    def test_cross_word_bpe_piece_rejected(self):
        with self.assertRaises(ValueError):
            align_s19e_raw_train9(CrossBoundaryTokenizer(),s19_train_texts()[0])

    def test_cpu_capture_has_no_gold_model_assembly_or_eval_imports(self):
        import research.nra3_s19.s19e_true43m_new9_cpu_capture as source
        code=inspect.getsource(source)
        parsed=ast.parse(code)
        imports=[n.module for n in ast.walk(parsed)
                 if isinstance(n,ast.ImportFrom)]
        imports.extend(alias.name for n in ast.walk(parsed)
                       if isinstance(n,ast.Import) for alias in n.names)
        self.assertFalse(any(x and any(z in x for z in
             ("s18c7","s18c6","nra3_s17","nra3_s15","nra3_s12","s3ci"))
             for x in imports))
        self.assertFalse(any(x in code for x in
             ("optimizer.step()", "loss.backward()", "S18_NEW96_TRAIN_COMPLETE")))
        self.assertIn("with collector.capture(model,q,ids):",code)
        self.assertIn("align_s19e_raw_train9(tok,q,expected_ids=ids)",code)


if __name__=="__main__":
    unittest.main()
