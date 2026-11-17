"""S19d synthetic/shadow contracts. These are NOT independent evaluations."""
from __future__ import annotations
import hashlib
import unittest

import torch
from torch.nn import functional as F

from research.nra3_s18.s18c3_raw_russian_grammar_cues import FEATURES
from research.nra3_s19.paid_grammar_parser import (
    CompactWordGrammar, INPUT_DIM, RANK, RAW_DIM
)
from research.nra3_s19.s19c_paid_event_factuality import STATUS
from research.nra3_s19.s19d_paid_contextual_scope import (
    PaidContextualEventScope, EXTRA_PARAMETERS, KERNEL_WIDTH,
)
from research.nra3_s19.s19d_train96_real43m_contextual_scope import (
    fixed_train_context, STEPS
)


def synthetic():
    torch.manual_seed(319104)
    x=torch.randn(2,7,INPUT_DIM,dtype=torch.float32)
    x[:,:,-RAW_DIM:]=torch.randint(0,2,(2,7,RAW_DIM)).float()
    mask=torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,0]],dtype=torch.bool)
    x[~mask]=0.
    return x,mask


def model():
    return PaidContextualEventScope(
        CompactWordGrammar().eval().requires_grad_(False))


class ContextualScopeContracts(unittest.TestCase):
    def test_all_raw_words_scored_and_no_gold_predicate_or_UPos_in_input(self):
        m=model()
        x,valid=synthetic()
        pred=m(x,valid)
        self.assertEqual(pred.shape,(2,7,len(STATUS)))
        self.assertTrue(bool(torch.isfinite(pred).all()))
        self.assertTrue(bool((pred[0,4:]==-1e4).all()))
        self.assertTrue(all(p.requires_grad for p in m.event_head.parameters()))
        self.assertTrue(all(not p.requires_grad for p in m.grammar.parameters()))

    def test_extra_2804_fp16_and_whole_frozen_grammar_billed(self):
        m=model()
        self.assertEqual(EXTRA_PARAMETERS,2804)
        self.assertEqual(len(m.head_physical()),5608)
        paid=m.physical_bytes()
        self.assertEqual(hashlib.sha256(paid[:-32]).digest(),paid[-32:])
        restored=PaidContextualEventScope.from_physical(paid)
        self.assertEqual(restored.head_physical(),m.head_physical())
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        self.assertEqual(restored.output_gate,0)
        self.assertFalse(restored.activation_allowed)

    def test_scoped_context_can_change_verb_but_not_nonlocal_past_sentence(self):
        # Construct explicit weights solely for this synthetic unit test,
        # proving the convolution's raw scope channel can observe previous НЕ.
        m=model()
        with torch.no_grad():
            m.context.weight.zero_()
            m.context.bias.zero_()
            m.event_head.weight.zero_()
            m.event_head.bias.zero_()
            not_channel=RANK+FEATURES.index("negative_particle")
            m.context.weight[0,not_channel,1]=2.
            m.event_head.weight[STATUS.index("NEGATED"),0]=2.
        x,valid=synthetic()
        x[:,:,-RAW_DIM:]=0.
        baseline=m(x,valid)
        x[0,1,320+FEATURES.index("negative_particle")]=1.
        changed=m(x,valid)
        self.assertGreater(
            changed[0,2,STATUS.index("NEGATED")],
            baseline[0,2,STATUS.index("NEGATED")])
        self.assertTrue(torch.equal(
            changed[0,0],baseline[0,0]))
        self.assertTrue(torch.equal(
            changed[0,3],baseline[0,3]))

    def test_grammar_gradients_absent_event_and_context_gradients_present(self):
        m=model()
        x,valid=synthetic()
        gold=torch.tensor([[0,0,2,1,-1,-1,-1],[0,1,3,0,1,2,-1]])
        loss=F.cross_entropy(m(x,valid)[valid],gold[valid])
        loss.backward()
        self.assertTrue(all(p.grad is None for p in m.grammar.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in [*m.context.parameters(),*m.event_head.parameters()]))

    def test_activate_or_nonbinary_surface_or_wrong_shape_rejected(self):
        m=model()
        x,valid=synthetic()
        with self.assertRaises(ValueError):
            m(x[:,:,:320],valid)
        with self.assertRaises(ValueError):
            m(x,valid.float())
        with self.assertRaises(ValueError):
            m(x,~valid)
        broken=x.clone()
        broken[0,0,-1]=0.3
        with self.assertRaises(ValueError):
            m(broken,valid)
        m.candidate_accepted=True
        with self.assertRaises(ValueError):
            m(x,valid)
        with self.assertRaises(ValueError):
            m.physical_bytes()

    def test_corrupted_physical_fp16_checksum_and_header_rejected(self):
        paid=model().physical_bytes()
        with self.assertRaises(ValueError):
            PaidContextualEventScope.from_physical(paid[:-10])
        with self.assertRaises(ValueError):
            PaidContextualEventScope.from_physical(paid[:-1]+b"x")
        with self.assertRaises(ValueError):
            PaidContextualEventScope.from_physical(b"NOPEHERE"+paid[8:])


if __name__=="__main__":
    unittest.main()
