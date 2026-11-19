"""S75 source-only original BPE offset contract, synthetic stand-in for tokenizer.

NEVER uses synthetic tokenizer as actual model evidence. The full
12-layer original physical native BPE test is in S75 GitHub Actions.
"""
from __future__ import annotations
import ast
import unittest
from pathlib import Path

from research.nra3_s20.s74_newtrain120_source_ONLY import (
    new_source_only_train120,source_sha256,
)
from research.nra3_s20.s75_original43m_gt_newtrain120_native_h8 import (
    EXPECTED_S74_SOURCE_SHA,source_offset_alignment,STAGE,require,
)


class SyntheticOffsetTokenizer:
    """Only tests Unicode source offsets; not the real model or true BPE."""
    def apply_chat_template(self,items,tokenize,add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "[USER]"+items[0]["content"]+"[/USER][BOT]"

    def encode(self,whole,add_special_tokens=False):
        return [ord(ch) for ch in whole]

    def __call__(self,whole,add_special_tokens=False,return_offsets_mapping=False):
        assert return_offsets_mapping
        return {"input_ids":self.encode(whole),
                "offset_mapping":[(i,i+1) for i in range(len(whole))]}


class TestS75GenuineNewTrainOnlyContract(unittest.TestCase):
    def test_exact_S74_new_120_input_digest_pinned(self):
        self.assertEqual(source_sha256(),EXPECTED_S74_SOURCE_SHA)
        self.assertEqual(len(new_source_only_train120()),120)
        self.assertEqual(
            STAGE,
            "NRA2_S75_REAL_ORIGINAL43M_GT_PROTECTED616J7_NEW_TRAIN120_NATIVE_BPE_H8_SOURCE_ONLY")

    def test_synthetic_offsets_120_utf8_exact_no_gold(self):
        tok=SyntheticOffsetTokenizer()
        for q in new_source_only_train120():
            text=q["text"]
            native=tok.encode(tok.apply_chat_template(
                [{"role":"user","content":text}],
                tokenize=False,add_generation_prompt=True))
            m=source_offset_alignment(tok,text,native)
            self.assertEqual(m.original_token_ids,tuple(native))
            self.assertEqual(m.words_recovered,len(m.raw_words))
            self.assertFalse(m.source_gold_loaded)
            self.assertFalse(m.causal_token_training_allowed)

    def test_original_token_id_rewrites_are_rejected(self):
        tok=SyntheticOffsetTokenizer()
        text=new_source_only_train120()[0]["text"]
        native=tok.encode(tok.apply_chat_template(
            [{"role":"user","content":text}],
            tokenize=False,add_generation_prompt=True))
        native[3]+=1
        with self.assertRaisesRegex(ValueError,"native offsets drift"):
            source_offset_alignment(tok,text,native)

    def test_never_import_S74_labels_or_EVAL_in_source_capture_module(self):
        tree=ast.parse(Path(
            "research/nra3_s20/s75_original43m_gt_newtrain120_native_h8.py"
            ).read_text(encoding="utf8"))
        imported=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):
                imported.append(n.module or "")
        self.assertTrue(all("gold" not in s.lower()
                            and "eval" not in s.lower()
                            and "s66_" not in s.lower()
                            for s in imported))
        self.assertIn("research.nra3_s20.s74_newtrain120_source_ONLY",imported)

if __name__=="__main__":
    unittest.main()
