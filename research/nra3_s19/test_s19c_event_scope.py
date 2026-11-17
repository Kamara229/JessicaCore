"""S19c source-only and synthetic-unit checks. NO model-quality evaluation."""
from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import unittest

import torch
from torch.nn import functional as F

from research.nra3_s18.ru96_multiverb_train_source import build_s18_new_train96
from research.nra3_s19.paid_grammar_parser import (
    CompactWordGrammar, INPUT_DIM, trainable_parameter_count
)
from research.nra3_s19.s19b_train96_true43m_grammar import WordGrammarTrainData
from research.nra3_s19.s19c_paid_event_factuality import (
    PaidEventFactuality, STATUS, HEAD_PARAMETERS, HEADER
)
from research.nra3_s19.s19c_train96_real43m_event_factuality import (
    build_train_targets, aggregate, evaluate_frozen
)


def synthetic_batch():
    torch.manual_seed(319113)
    x = torch.randn(2,5,INPUT_DIM,dtype=torch.float32)
    valid = torch.tensor([[1,1,1,0,0],[1,1,1,1,0]],dtype=torch.bool)
    x[~valid] = 0.
    return x,valid


class S19cEventScopeContracts(unittest.TestCase):
    def test_joint_physical_count_all_grammar_and_event_fp16_paid(self):
        grammar=CompactWordGrammar().eval()
        grammar.requires_grad_(False)
        event=PaidEventFactuality(grammar).eval()
        self.assertEqual(trainable_parameter_count(grammar),6528)
        self.assertEqual(HEAD_PARAMETERS,68)
        self.assertEqual(len(event.head_physical()),136)
        raw=event.physical_bytes()
        restored=PaidEventFactuality.from_physical(raw)
        self.assertEqual(restored.head_physical(),event.head_physical())
        self.assertEqual(restored.output_gate,0)
        self.assertFalse(restored.candidate_accepted)
        self.assertFalse(restored.activation_allowed)
        self.assertFalse(any(p.requires_grad for p in restored.parameters()))
        self.assertEqual(hashlib.sha256(raw[:-32]).digest(),raw[-32:])

    def test_corrupted_joint_physical_bytes_rejected(self):
        frozen=CompactWordGrammar().eval().requires_grad_(False)
        blob=PaidEventFactuality(frozen).physical_bytes()
        with self.assertRaises(ValueError):
            PaidEventFactuality.from_physical(blob[:-2]+b"xx")
        with self.assertRaises(ValueError):
            PaidEventFactuality.from_physical(blob[:-4])
        with self.assertRaises(ValueError):
            PaidEventFactuality.from_physical(b"FALSEHDR"+blob[8:])

    def test_all_words_scored_without_oracle_verb_or_gold_tags(self):
        frozen=CompactWordGrammar().eval().requires_grad_(False)
        model=PaidEventFactuality(frozen)
        x,valid=synthetic_batch()
        out=model(x,valid)
        self.assertEqual(out.shape,(2,5,len(STATUS)))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertEqual(out[0,3].tolist(),[-1e4]*len(STATUS))
        self.assertTrue(all(p.requires_grad for p in model.event_head.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.grammar.parameters()))

    def test_event_training_gradient_frozen_grammar(self):
        frozen=CompactWordGrammar().eval().requires_grad_(False)
        model=PaidEventFactuality(frozen)
        x,valid=synthetic_batch()
        targets=torch.tensor([[0,2,1,-1,-1],[0,1,3,0,-1]],dtype=torch.long)
        loss=F.cross_entropy(model(x,valid)[valid],targets[valid])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.event_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.grammar.parameters()))

    def test_forbid_live_activation_and_wrong_input(self):
        frozen=CompactWordGrammar().eval().requires_grad_(False)
        model=PaidEventFactuality(frozen)
        x,valid=synthetic_batch()
        with self.assertRaises(ValueError):
            model(x[...,:320],valid)
        with self.assertRaises(ValueError):
            model(x,valid.float())
        model.output_gate=1
        with self.assertRaises(ValueError):
            model(x,valid)
        with self.assertRaises(ValueError):
            model.physical_bytes()

    def test_original_train96_event_status_only_on_verbs(self):
        rows=build_s18_new_train96()
        n=max(len(r.words) for r in rows)
        x=torch.zeros(96,n,INPUT_DIM)
        valid=torch.zeros(96,n,dtype=torch.bool)
        upos=torch.full((96,n),-1,dtype=torch.long)
        for k,row in enumerate(rows):
            valid[k,:len(row.words)]=True
        b=WordGrammarTrainData(x,valid,upos,upos,upos,upos,"source96","mock")
        result=build_train_targets(tuple(SimpleNamespace(row=r) for r in rows),b)
        v=result.status[valid]
        self.assertEqual((v==STATUS.index("ASSERTED")).sum().item(),84)
        self.assertEqual((v==STATUS.index("NEGATED")).sum().item(),12)
        self.assertEqual((v==STATUS.index("FUTURE_UNREALIZED")).sum().item(),12)
        self.assertEqual((v>0).sum().item(),108)
        self.assertTrue(bool((result.status[~valid]==-1).all()))
        self.assertEqual(result.source_sha,"source96")

    def test_aggregate_joint_wrong_factuality_counts_both_fp_and_fn(self):
        metrics={
            "all_raw_words":10,"event_gold_count":2,
            "correct_predicate_and_factuality_TP":1,
            "false_predicate_or_wrong_factuality_FP":2,
            "missed_or_wrong_factuality_FN":1,
            "per_status":{name:{
                "TP":0,"FP":0,"FN":0,"TRAIN_internal_gold_support":0
            } for name in STATUS[1:]}
        }
        m=aggregate([metrics,metrics])
        self.assertEqual((m["TP"],m["FP"],m["FN"]),(2,4,2))
        self.assertAlmostEqual(m["joint_predicate_factuality_micro_F1"],4/10)

    def test_frozen_synthetic_output_metrics_no_gold_as_input(self):
        frozen=CompactWordGrammar().eval().requires_grad_(False)
        model=PaidEventFactuality(frozen).eval()
        x,valid=synthetic_batch()
        labels=torch.tensor([[0,2,1,-1,-1],[0,1,3,0,-1]])
        data=WordGrammarTrainData(x,valid,labels,labels,labels,labels,"synthetic","synthetic")
        gold=SimpleNamespace(status=labels)
        m=evaluate_frozen(model,data,gold)
        self.assertEqual(m["all_raw_words"],7)
        self.assertEqual(m["event_gold_count"],4)
        self.assertTrue(0<=m["joint_predicate_factuality_F1"]<=1)


if __name__=="__main__":
    unittest.main()
