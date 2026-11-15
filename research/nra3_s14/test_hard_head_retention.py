"""NRA3 S14 contracts: strictly causal, paid FP16, hard-head retention.

All synthetic tests are clearly source-level, NOT original43M performance.
The only original 43M experiment is the GitHub Actions native-C job.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import torch

from research.nra3_s13.paid_relation_feedback import (
    PaidRelationFeedback,PHYSICAL_BYTES,PARAMETERS,UPOS_DIM,
)
from research.nra3_s13.real43m_train72_readouts import FixedNative72
from research.nra3_s14.hard_head_retention_train72 import (
    FAMILY_TRAIN,FAMILY_INTERNAL,
    MODES,STEPS,SNAP_EVERY,
    _lock_upos_,_margin,task_aligned_objective,
    baseline_safe_comparison,metrics,split72,
)
from research.nra3_s9.new_ru_train72 import build_s9_train72


class S14CausalHardHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(314014)
        torch.set_num_threads(1)

    def mock(self,n=72,t=8)->FixedNative72:
        r=torch.randn(n,t,128)*.10
        up=torch.randn(n,t,17)*.15
        role=torch.randn(n,t,9)*.15
        dep=torch.randn(n,t,t)*.02
        valid=torch.ones((n,t),dtype=torch.bool)
        gu=torch.full((n,t),-100,dtype=torch.long)
        gr=torch.full((n,t),-100,dtype=torch.long)
        gd=torch.full((n,t),-100,dtype=torch.long)
        candidates=torch.zeros((n,t,t),dtype=torch.bool)
        for i in range(n):
            gu[i,:5]=torch.tensor([3,15,11,7,12])
            gr[i,:5]=torch.tensor([1,0,3,2,0])
            gd[i,1:5]=torch.tensor([0,0,1,2])
            for t0 in range(t):
                candidates[i,t0,:t0+1]=True
        dep=dep.masked_fill(~torch.tril(torch.ones(t,t,dtype=torch.bool)), -1e4)
        return FixedNative72(r,up,role,dep,valid,gu,gr,gd,candidates,n*5,n*4)

    def test_8_4_lexical_family_internal_split_never_independent_eval(self):
        samples=[type("Sample",(),{"row":r}) for r in build_s9_train72()]
        self.assertEqual(len(samples),72)
        self.assertEqual(FAMILY_TRAIN,tuple(range(8)))
        self.assertEqual(FAMILY_INTERNAL,tuple(range(8,12)))
        tr,dev=split72(tuple(samples),self.mock())
        self.assertEqual(tr.relation.shape[0],48)
        self.assertEqual(dev.relation.shape[0],24)
        self.assertEqual(tr.num_words,48*5)
        self.assertEqual(dev.num_words,24*5)
        self.assertEqual(tr.num_past_head_arcs,48*4)

    def test_physically_zero_UPOS_head_is_exact_original_under_any_causal_input(self):
        m=PaidRelationFeedback().eval()
        _lock_upos_(m)
        with torch.no_grad():
            m.output_weight[UPOS_DIM:,:].normal_(0,.05)
            m.output_bias[UPOS_DIM:].normal_(0,.02)
            m.dependency_diagonal.fill_(.04)
        physical=m.physical_bytes()
        self.assertEqual(len(physical),PHYSICAL_BYTES)
        self.assertEqual(PARAMETERS,1266)
        reload=PaidRelationFeedback.from_physical(physical)
        self.assertTrue(bool((reload.output_weight[:UPOS_DIM]==0).all()))
        self.assertTrue(bool((reload.output_bias[:UPOS_DIM]==0).all()))
        x=self.mock(n=2)
        o=reload(x.relation,x.upos,x.role,x.dep,x.valid)
        self.assertTrue(torch.equal(o.upos_logits,x.upos))
        self.assertFalse(torch.equal(o.semantic_role_logits,x.role))
        self.assertFalse(torch.equal(o.dependency_logits,x.dep))
        self.assertFalse(reload.candidate_accepted)
        self.assertEqual(reload.output_gate,0)

    def test_unweighted_hard_margin_affects_true_top_class(self):
        logits=torch.tensor([[.2,.8,.1],[.9,.1,.2]],requires_grad=True)
        labels=torch.tensor([0,1])
        base=_margin(logits,labels)
        self.assertGreater(float(base.detach()),0.)
        base.backward()
        self.assertTrue(bool((logits.grad!=0).any()))
        improved=logits.detach()+torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        self.assertLess(float(_margin(improved,labels)),float(base.detach()))

    def test_joint_and_locked_objectives_have_real_physical_gradients(self):
        x=self.mock(n=2)
        for mode in MODES:
            model=PaidRelationFeedback().eval()
            if mode=="upos_locked":
                _lock_upos_(model)
            loss=task_aligned_objective(model,x,mode)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertTrue(bool((model.output_weight.grad[UPOS_DIM:]!=0).any()))
            self.assertTrue(bool((model.output_bias.grad[UPOS_DIM:]!=0).any()))
            with self.assertRaises(ValueError):
                task_aligned_objective(model,x,"EVAL")

    def test_hard_safety_is_priority_over_soft_CE(self):
        old={"upos":{"correct":245,"total":408},
             "roles":{"correct":247,"total":408},
             "past_heads":{"correct":147,"total":192}}
        improved={k:dict(v) for k,v in old.items()}
        self.assertTrue(baseline_safe_comparison(improved,old))
        improved["roles"]["correct"]=260
        self.assertTrue(baseline_safe_comparison(improved,old))
        improved["upos"]["correct"]=244
        self.assertFalse(baseline_safe_comparison(improved,old))
        improved["upos"]["correct"]=245
        improved["past_heads"]["correct"]=146
        self.assertFalse(baseline_safe_comparison(improved,old))

    def test_shape_and_physical_regression_refuses_old_weak_evals(self):
        m=PaidRelationFeedback().eval()
        self.assertEqual(PHYSICAL_BYTES,2581)
        self.assertEqual(len(m.physical_bytes()),2581)
        self.assertEqual(STEPS,160)
        self.assertEqual(SNAP_EVERY,20)
        self.assertEqual(MODES,("joint_margin","upos_locked"))
        self.assertEqual(len(FAMILY_TRAIN)+len(FAMILY_INTERNAL),12)
        self.assertTrue(set(FAMILY_TRAIN).isdisjoint(FAMILY_INTERNAL))
        metrics_sample=metrics(m,self.mock(n=2))
        self.assertEqual(metrics_sample["upos"]["total"],10)
        self.assertEqual(metrics_sample["roles"]["total"],10)
        self.assertEqual(metrics_sample["past_heads"]["total"],8)


if __name__=="__main__":
    unittest.main()
