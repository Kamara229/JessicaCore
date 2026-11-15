"""Physical S16 2309-FP16 and existing TRAIN72 source-only tests."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import torch

from research.nra3_s16.paid_directed_valence import (
    ROLES,MAGIC,PARAMETERS,PHYSICAL_BYTES,
    PaidDirectedValence,
)
from research.nra3_s16.direct_train72_edges import (
    SPLIT,build_directed_train72,canonical_directed_train72,
    sealed_directed_train72,validate_train72,
)
from research.nra3_s9.new_ru_train72 import build_s9_train72


class PaidDirectedValenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(316012)
        torch.set_num_threads(1)

    def test_exact_physical_fp16_charges_all_2309_parameters_and_full_SHA(self):
        m=PaidDirectedValence()
        self.assertEqual(PARAMETERS,2309)
        self.assertEqual(PHYSICAL_BYTES,4665)
        self.assertEqual(sum(p.numel() for p in m.parameters()),PARAMETERS)
        physical=m.physical_bytes()
        self.assertEqual(len(physical),PHYSICAL_BYTES)
        self.assertEqual(physical[:8],MAGIC)
        reloaded=PaidDirectedValence.from_physical(physical)
        self.assertEqual(reloaded.physical_bytes(),physical)
        self.assertFalse(any(p.requires_grad for p in reloaded.parameters()))
        self.assertEqual(reloaded.output_gate,0)
        self.assertFalse(reloaded.candidate_accepted)
        for bad in (physical[:-1],physical+b"\\x00",
                    bytes([physical[0]^1])+physical[1:],
                    physical[:130]+bytes([physical[130]^1])+physical[131:]):
            with self.assertRaises(ValueError):
                PaidDirectedValence.from_physical(bad)

    def test_both_orientations_preserve_predicate_to_argument_direction(self):
        m=PaidDirectedValence().eval()
        relations=torch.randn((1,8,128))
        valid=torch.ones((1,8),dtype=torch.bool)
        original=m(relations,valid)
        reloaded=PaidDirectedValence.from_physical(m.physical_bytes())
        roundtrip=reloaded(relations,valid)
        self.assertTrue(torch.equal(original.predicate_now,roundtrip.predicate_now))
        self.assertTrue(torch.equal(original.argument_now,roundtrip.argument_now))
        self.assertEqual(original.predicate_now.shape,(1,8,8,5))
        self.assertEqual(original.argument_now.shape,(1,8,8,5))
        self.assertTrue(bool((~original.allowed[0].tril(-1)).all()))
        self.assertTrue(bool((original.predicate_now[0].triu(0)==-1e4).all()))
        self.assertTrue(bool((original.argument_now[0].triu(0)==-1e4).all()))
        self.assertTrue(bool((original.predicate_now[0,3,1]!=-1e4).all()))
        self.assertTrue(bool((original.argument_now[0,3,1]!=-1e4).all()))

    def test_strict_prefix_causal_no_future_sentence_summary_or_padding(self):
        m=PaidDirectedValence().eval()
        r=torch.randn((1,9,128))
        v=torch.ones((1,9),dtype=torch.bool)
        with torch.no_grad():
            a=m(r[:,:6],v[:,:6])
            b=m(r,v)
        self.assertTrue(torch.allclose(a.predicate_now,
                    b.predicate_now[:,:6,:6],atol=2e-5,rtol=2e-5))
        self.assertTrue(torch.allclose(a.argument_now,
                    b.argument_now[:,:6,:6],atol=2e-5,rtol=2e-5))
        vp=v.clone();vp[:,6:]=False
        z=m(r,vp)
        self.assertTrue(torch.allclose(a.predicate_now,
                    z.predicate_now[:,:6,:6],atol=2e-5,rtol=2e-5))
        self.assertTrue(torch.allclose(a.argument_now,
                    z.argument_now[:,:6,:6],atol=2e-5,rtol=2e-5))
        self.assertFalse(bool(z.allowed[0,6:].any()))

    def test_trainable_causal_score_gradients_and_zero_activation(self):
        m=PaidDirectedValence()
        r=torch.randn((2,6,128))
        v=torch.ones((2,6),dtype=torch.bool)
        z=m(r,v)
        total=torch.nn.functional.cross_entropy(
            z.predicate_now[:,3,1,:],torch.tensor([1,2]))+(
            torch.nn.functional.cross_entropy(
            z.argument_now[:,4,2,:],torch.tensor([3,4])))
        total.backward()
        for x in m.parameters():
            self.assertIsNotNone(x.grad)
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
            self.assertTrue(bool((x.grad!=0).any()))
        m.output_gate=1
        with self.assertRaises(RuntimeError):
            m(r,v)
        m.output_gate=0
        m.candidate_accepted=True
        with self.assertRaises(RuntimeError):
            m.physical_bytes()
        m.candidate_accepted=False
        with self.assertRaises(ValueError):
            m(r,v.float())
        with self.assertRaises(ValueError):
            m(r[:,:,:-1],v)

    def test_original_72_TRAIN_passive_and_receiver_role_direction_only(self):
        old=build_s9_train72()
        rows=build_directed_train72()
        self.assertEqual(len(rows),len(old)==72)
        self.assertEqual(len({r.text_sha256 for r in rows}),72)
        self.assertEqual(set(r.source_split for r in rows),{SPLIT})
        all_links=[z for row in rows for z in row.links]
        self.assertGreaterEqual(len(all_links),180)
        self.assertEqual(set(x.role for x in all_links),set(ROLES[1:]))
        self.assertEqual(set(x.orientation for x in all_links),
                         {"predicate_now","argument_now"})
        passive=[r for r in rows if r.construction=="passive_agent"]
        self.assertEqual(len(passive),12)
        for r in passive:
            self.assertEqual(sum(x.role=="AGENT" for x in r.links),1)
            self.assertEqual(sum(x.role=="PATIENT" for x in r.links),1)
            patient=next(x for x in r.links if x.role=="PATIENT")
            self.assertEqual(patient.orientation,"predicate_now")
        for r in rows:
            self.assertTrue(all(x.current_word>x.prior_word for x in r.links))
        x=sealed_directed_train72()
        source=canonical_directed_train72(rows)
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         x["S16_complete_predicate_to_argument_gold_source_sha256"])
        self.assertTrue(x["S16_existing_train72_only"])
        self.assertFalse(x["S16_used_consumed_S15c_EVAL24"])
        self.assertEqual(x["S16_optimizer_steps"],0)

    def test_gold_causal_direction_and_source_guard_fail_closed(self):
        rows=build_directed_train72()
        with self.assertRaises(ValueError):
            validate_train72(rows[:-1])
        first=rows[0]
        changed=replace(first,source_split="EVAL")
        with self.assertRaises(ValueError):
            validate_train72((changed,)+rows[1:])
        edge=first.links[0]
        wrong=replace(edge,current_word=edge.prior_word)
        changed=replace(first,links=(wrong,)+first.links[1:])
        with self.assertRaises(ValueError):
            validate_train72((changed,)+rows[1:])


if __name__=="__main__":
    unittest.main()
