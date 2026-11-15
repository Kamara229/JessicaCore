"""Synthetic gradients + real precommitted TRAIN-only pair-layout contracts for S3CE."""
from __future__ import annotations
import unittest
import torch
from research.nra2_bridge.stage3ce_order_paired_fit192_objective import (
    build_fit_pairs,objective,protocol_report,_sym_kl,FIT_COUNT,PAIR_COUNT,
)
from research.nra2_bridge.stage3bx_blocked_family_topology_protocol import split_data


class Stage3CEPairLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout=build_fit_pairs()

    def test_exact_fit192_and_no_old_train_guards(self):
        g=split_data()
        self.assertEqual(len(self.layout.pairs),96)
        self.assertEqual(len(self.layout.fit_prompt_sha256),192)
        self.assertEqual(sum(self.layout.yes_labels),96)
        a=self.layout.pairs
        self.assertEqual(sorted(x for p in a for x in p),list(range(192)))
        for first,reverse in a:
            r1,r2=g["fit"][first],g["fit"][reverse]
            self.assertEqual((r1.query_style,r2.query_style),
                             (r1.query_style//2*2,r1.query_style//2*2+1))
            self.assertEqual((r1.src,r1.dst,r1.reachable),
                             (r2.src,r2.dst,r2.reachable))
            self.assertEqual(r1.edges,r2.edges)
            self.assertNotEqual(r1.prompt_sha256,r2.prompt_sha256)
        blocked={r.prompt_sha256 for k in ("grammar_guard","topology_guard",
                                          "double_guard") for r in g[k]}
        self.assertTrue(not set(self.layout.fit_prompt_sha256)&blocked)
        report=protocol_report(self.layout)
        self.assertFalse(report["candidate_accepted"])
        self.assertEqual(report["output_gate"],0.)
        self.assertTrue(report["all_five_independent_EVAL128_excluded_from_gradient_scoring_selection"])

    def _perfect_logits(self):
        src,dst=self.layout.gold()
        logits_a=torch.zeros((192,6),dtype=torch.float32)
        logits_b=torch.zeros_like(logits_a)
        logits_a.scatter_(1,src.unsqueeze(1),8.0)
        logits_b.scatter_(1,dst.unsqueeze(1),8.0)
        return logits_a,logits_b

    def test_mirrored_correct_wording_zero_consistency_gap(self):
        a,b=self._perfect_logits()
        result=objective(a,b,self.layout,(a,b))
        self.assertLess(float(result["supervised"]),0.002)
        self.assertLess(abs(float(result["semantic_pair_symmetric_kl"])),1e-7)
        self.assertLess(abs(float(result["direction_ce_gap_squared"])),1e-7)
        self.assertLess(abs(float(result["frozen_S3BN_teacher_kl"])),1e-7)

    def test_destination_first_errors_increase_loss_and_gradients(self):
        clean_s,clean_d=self._perfect_logits()
        baseline=objective(clean_s,clean_d,self.layout)
        src_gold,dst_gold=self.layout.gold()
        distorted_s=clean_s.clone()
        distorted_d=clean_d.clone()
        odd=torch.tensor([odd for _,odd in self.layout.pairs])
        distorted_s[odd]=0.0
        distorted_d[odd]=0.0
        wrong=(src_gold[odd]+1)%6
        distorted_s[odd,wrong]=8.
        distorted_d[odd,(dst_gold[odd]+1)%6]=8.
        distorted_s=distorted_s.detach().requires_grad_()
        distorted_d=distorted_d.detach().requires_grad_()
        measured=objective(distorted_s,distorted_d,self.layout)
        self.assertGreater(float(measured["total"]),float(baseline["total"]))
        self.assertGreater(float(measured["destination_first_ce"]),
                           float(measured["source_first_ce"]))
        self.assertGreater(float(measured["semantic_pair_symmetric_kl"]),0.1)
        self.assertGreater(float(measured["direction_ce_gap_squared"]),0.1)
        measured["total"].backward()
        self.assertTrue(bool(torch.isfinite(distorted_s.grad).all()))
        self.assertTrue(bool(torch.isfinite(distorted_d.grad).all()))
        self.assertGreater(float(distorted_s.grad.abs().sum()),0.)

    def test_reverse_order_pairwise_loss_is_symmetric(self):
        a=torch.tensor([[1.,2.,0.],[2.,0.,1.]],dtype=torch.float32)
        b=torch.tensor([[0.,2.,1.],[1.,1.,3.]],dtype=torch.float32)
        self.assertAlmostEqual(float(_sym_kl(a,b)),float(_sym_kl(b,a)),places=6)

    def test_invalid_and_nonfinite_fail_closed(self):
        a,b=self._perfect_logits()
        with self.assertRaisesRegex(AssertionError,"fit192"):
            objective(a[:191],b,self.layout)
        bad=a.clone()
        bad[3,0]=float("nan")
        with self.assertRaisesRegex(AssertionError,"finite"):
            objective(bad,b,self.layout)
        teacher=a.clone()[:100]
        with self.assertRaisesRegex(AssertionError,"teacher"):
            objective(a,b,self.layout,(teacher,b))

    def test_true_semantic_reverse_must_not_be_enforced_equal(self):
        g=split_data()["fit"]
        by_graph={}
        for i,r in enumerate(g):
            by_graph.setdefault(r.graph_id,{})[r.query_style]=i
        for styles in by_graph.values():
            # Style0/1 is one direction; style2/3 is the opposite.
            p=g[styles[0]]
            q=g[styles[2]]
            self.assertEqual((p.src,p.dst),(q.dst,q.src))
            self.assertNotEqual((p.src,p.dst),(q.src,q.dst))
        self.assertEqual(len(by_graph),48)


if __name__=="__main__":
    unittest.main()
