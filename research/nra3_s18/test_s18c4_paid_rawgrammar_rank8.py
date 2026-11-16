"""Research only S18c4 paid source ABI and raw-granularity no-oracle tests."""
from __future__ import annotations

import hashlib
import unittest

import torch
from torch.nn import functional as F

from research.nra3_s18.s18c4_paid_rawgrammar_rank8 import (
    PaidRawGrammarDirectedValence,PARAMETERS,EXTRA_PARAMETERS,PHYSICAL_BYTES,
    DIM,HEADER,
)
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    raw_surface_grammar,
)


class GrammarPaidRank8Tests(unittest.TestCase):
    def test_all_2597_fp16_parameters_physical_checksum_roundtrip(self):
        model=PaidRawGrammarDirectedValence().eval()
        self.assertEqual(DIM,18)
        self.assertEqual(EXTRA_PARAMETERS,288)
        self.assertEqual(PARAMETERS,2597)
        self.assertEqual(PHYSICAL_BYTES,5286)
        self.assertEqual(sum(t.numel() for t in model.parameters()),PARAMETERS)
        self.assertEqual(model.output_gate,0)
        self.assertFalse(model.candidate_accepted)
        raw=model.physical_bytes()
        self.assertEqual(len(raw),5286)
        self.assertEqual(hashlib.sha256(raw[:-32]).digest(),raw[-32:])
        reload=PaidRawGrammarDirectedValence.from_physical(raw)
        self.assertEqual(reload.physical_bytes(),raw)
        self.assertFalse(any(p.requires_grad for p in reload.parameters()))
        mutable=bytearray(raw)
        mutable[HEADER.size+1]^=0x01
        with self.assertRaises(ValueError):
            PaidRawGrammarDirectedValence.from_physical(bytes(mutable))

    def test_zero_grammar_projections_exact_base_model_regression(self):
        torch.manual_seed(18)
        model=PaidRawGrammarDirectedValence().eval()
        with torch.no_grad():
            model.predicate_grammar.zero_()
            model.argument_grammar.zero_()
        old=model.base
        pred=torch.randn((6,128))
        arg=torch.randn((6,128))
        p=torch.tensor([1,3,5,2,4,6],dtype=torch.long)
        a=torch.tensor([0,2,1,4,6,1],dtype=torch.long)
        current=torch.maximum(p,a)
        grammar=torch.randint(0,2,(6,DIM),dtype=torch.int32).float()
        result=model.score_pairs(pred,arg,p,a,current,grammar,grammar)
        old_result=old.score_pairs(pred,arg,p,a,current)
        self.assertTrue(torch.equal(result,old_result))
        with self.assertRaises(ValueError):
            model.score_pairs(pred,arg,p,a,current,grammar[:5],grammar)

    def test_every_physical_new_projection_and_old_role_receives_train_gradient(self):
        torch.manual_seed(19)
        model=PaidRawGrammarDirectedValence().eval()
        n=18
        pred=torch.randn((n,128))
        arg=torch.randn((n,128))
        p=torch.arange(2,n+2,dtype=torch.long)
        a=torch.arange(n,dtype=torch.long)
        current=torch.maximum(p,a)
        rawp=torch.randint(0,2,(n,DIM),dtype=torch.int32).float()
        rawa=torch.randint(0,2,(n,DIM),dtype=torch.int32).float()
        target=torch.tensor([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2],
                            dtype=torch.long)
        logits=model.score_pairs(pred,arg,p,a,current,rawp,rawa)
        loss=F.cross_entropy(logits,target)
        loss.backward()
        self.assertEqual(tuple(logits.shape),(n,5))
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all()
                            for x in model.parameters()))
        self.assertGreater(model.predicate_grammar.grad.abs().sum().item(),0.)
        self.assertGreater(model.argument_grammar.grad.abs().sum().item(),0.)
        model.candidate_accepted=True
        with self.assertRaises(ValueError):
            model.score_pairs(pred,arg,p,a,current,rawp,rawa)

    def test_no_gold_returned_by_raw_unicode_feature_extraction(self):
        words=raw_surface_grammar("Книга была отправлена Антоном Софье.")
        self.assertTrue(all(len(w.features)==DIM for w in words))
        self.assertTrue(all(set(w.features).issubset({0,1}) for w in words))
        self.assertFalse(any(hasattr(w,attr) for w in words for attr in
            ("semantic_role","upos","deprel","lemma","head","gold")))
