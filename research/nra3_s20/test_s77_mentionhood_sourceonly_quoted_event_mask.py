"""NRA2 S77 original 43M situation quote/mention source-vs-loss contracts.

Synthetic masks are only unit tests; not evidence of real model quality.
The physical S75 original native BPE 120-read forensic CI is separate.
"""
import ast
import unittest
from pathlib import Path

import torch

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import PUNCT_FEATURE
from research.nra3_s20.s74_newtrain120_source_ONLY import new_source_only_train120
from research.nra3_s20.s77_real43m_gt_mentionhood_vs_asserted_candidate_audit import (
    STAGE,mention_only_source_mask,
)


class TestS77OriginalSourceOnlyMentionhood(unittest.TestCase):
    def test_quote_inside_mentioned_event_admitted_without_assertion(self):
        text=new_source_only_train120()[1]["text"]
        words=raw_surface_grammar(text)
        n=len(words)
        valid=torch.ones((1,n),dtype=torch.bool)
        unsafe=torch.zeros_like(valid)
        punct=torch.tensor([[bool(z.features[PUNCT_FEATURE]) for z in words]])
        m=mention_only_source_mask(valid,unsafe,punct,(text,))
        predicate=[i for i,z in enumerate(words)
                   if z.form.casefold()=="передал"]
        self.assertEqual(len(predicate),1)
        self.assertTrue(bool(m[0,predicate[0]]))
        self.assertEqual(m.shape,valid.shape)
        self.assertFalse(bool((m & punct).any()))
        self.assertEqual(
            STAGE,
            "NRA2_S77_REAL43M_GT_NEWTRAIN120_QUOTE_INCLUSIVE_MENTION_VS_ASSERTED_PROPOSAL_AUDIT")

    def test_native_unsafe_protects_even_quoted_verb(self):
        text=new_source_only_train120()[1]["text"]
        words=raw_surface_grammar(text)
        n=len(words)
        idx=next(i for i,z in enumerate(words)
                 if z.form.casefold()=="передал")
        valid=torch.ones((1,n),dtype=torch.bool)
        unsafe=torch.zeros_like(valid)
        unsafe[0,idx]=True
        punct=torch.tensor([[bool(z.features[PUNCT_FEATURE]) for z in words]])
        mask=mention_only_source_mask(valid,unsafe,punct,(text,))
        self.assertFalse(bool(mask[0,idx]))

    def test_negation_not_mistaken_for_event(self):
        text=new_source_only_train120()[2]["text"]
        words=raw_surface_grammar(text)
        n=len(words)
        valid=torch.ones((1,n),dtype=torch.bool)
        unsafe=torch.zeros_like(valid)
        punct=torch.tensor([[bool(z.features[PUNCT_FEATURE]) for z in words]])
        mask=mention_only_source_mask(valid,unsafe,punct,(text,))
        indexes=[i for i,z in enumerate(words) if z.form.casefold()=="не"]
        self.assertEqual(len(indexes),1)
        self.assertFalse(bool(mask[0,indexes[0]]))
        predicate=[i for i,z in enumerate(words)
                   if z.form.casefold()=="передал"]
        self.assertEqual(len(predicate),1)
        self.assertTrue(bool(mask[0,predicate[0]]))

    def test_source_cannot_read_constructed_training_labels_before_forward(self):
        p=Path("research/nra3_s20/s77_real43m_gt_mentionhood_vs_asserted_candidate_audit.py")
        tree=ast.parse(p.read_text(encoding="utf-8"))
        top_imports=[
            z.module
            for z in tree.body
            if isinstance(z,ast.ImportFrom)
        ]
        self.assertTrue(all("gold" not in (z or "").lower()
                            and "eval" not in (z or "").lower()
                            for z in top_imports))
        self.assertNotIn("s75_original43m_gt_newtrain120_native_h8",
                         top_imports)
        fun=next(z for z in tree.body if isinstance(z,ast.FunctionDef)
                 and z.name=="posthoc_train_only_coverage")
        self.assertTrue(any(isinstance(z,ast.ImportFrom)
                            and "TRAIN_LOSS_ONLY" in z.module
                            for z in ast.walk(fun)))

    def test_invalid_source_masks_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"source-only native BPE"):
            mention_only_source_mask(torch.ones(1,3,dtype=torch.bool),
                    torch.ones(1,3,dtype=torch.bool),
                    torch.ones(1,2,dtype=torch.bool),("тест",))


if __name__=="__main__":
    unittest.main()
