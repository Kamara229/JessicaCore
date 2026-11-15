"""S11 tests: physical 39-param source, causal features, no gold/EVAL.

These are CPU unit tests with synthetic hidden; the genuine 43M 72 TRAIN
source and CPU native C are checked separately in the research Actions job.
"""
from __future__ import annotations

import dataclasses
import hashlib
import unittest

import torch
from torch.nn import functional as F

from research.nra3_s0.neural_area import NRA3Config,NRA3NeuralArea,SEMANTIC_ROLES
from research.nra3_s7.parallel_hypotheses import parallel_three_relations
from research.nra3_s11.paid_router import (
    PaidThreeWayRouter,PHYSICAL_BYTES,SOURCE_PARAMETERS,D,K,
    train_balanced_router,
)
from research.nra3_s11.weak_ru72_train_labels import (
    weak_train_router_labels,BRANCHES,
)
from research.nra3_s9.new_ru_train72 import build_s9_train72
from research.nra3_s4.complete_prefix_training_contract import (
    CausalAlignment,AlignedWord,
)


class RouterS11PhysicalContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(311103)
        torch.set_num_threads(1)

    def test_physical_39_fp16_plus_sha_roundtrip_and_tamper_fail(self):
        p=PaidThreeWayRouter()
        with torch.no_grad():
            p.weights.copy_(torch.randn(3,12)/10)
            p.bias.copy_(torch.tensor([.01,-.02,.015]))
        x=torch.randn(4,7,12)
        data=p.physical_bytes()
        self.assertEqual(len(data),PHYSICAL_BYTES)
        self.assertEqual(SOURCE_PARAMETERS,39)
        self.assertEqual(PHYSICAL_BYTES,123)
        q=PaidThreeWayRouter.from_physical(data)
        self.assertEqual(data,q.physical_bytes())
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         hashlib.sha256(q.physical_bytes()).hexdigest())
        self.assertTrue(torch.equal(p(x),q(x)))
        with self.assertRaises(ValueError):
            PaidThreeWayRouter.from_physical(data[:-1])
        corrupted=bytearray(data)
        corrupted[30]^=2
        with self.assertRaises(ValueError):
            PaidThreeWayRouter.from_physical(bytes(corrupted))
        with self.assertRaises(ValueError):
            PaidThreeWayRouter.from_physical(data+b"\x00")
        self.assertFalse(q.candidate_accepted)
        self.assertEqual(q.output_gate,0)
        self.assertFalse(any(z.requires_grad for z in q.parameters()))

    def test_zero_router_is_uniform_and_paid_bias_really_controls_gate(self):
        b,t,r=1,6,128
        q=torch.randn(b,t,r)
        v=torch.randn(b,t,r)
        allowed=torch.tril(torch.ones(t,t,dtype=torch.bool)).unsqueeze(0)
        valid=torch.ones(b,t,dtype=torch.bool)
        scores=q@q.transpose(1,2)/(r**.5)
        roles=torch.randn(b,t,len(SEMANTIC_ROLES))
        empty=PaidThreeWayRouter()
        a=parallel_three_relations(query=q,value=v,scores=scores,
          allowed=allowed,valid=valid,semantic_role_logits=roles,
          local_window=4,selector_version=3,paid_router=empty)
        self.assertEqual(a.router_features.shape,(b,t,12))
        self.assertTrue(torch.allclose(a.hypothesis_gates,
                         torch.full((b,t,3),1/3),atol=1e-6))
        with torch.no_grad():
            empty.bias.copy_(torch.tensor([-3.,4.,-3.]))
        z=parallel_three_relations(query=q,value=v,scores=scores,
          allowed=allowed,valid=valid,semantic_role_logits=roles,
          local_window=4,selector_version=3,paid_router=empty)
        self.assertTrue((z.hypothesis_gates.argmax(-1)==1).all())
        self.assertFalse(torch.allclose(z.message,a.message))
        self.assertTrue(torch.allclose(z.hypothesis_gates.sum(-1),
                                       torch.ones((b,t)),atol=1e-6))
        self.assertFalse(torch.any(torch.isnan(z.message)))

    def test_route_is_unchanged_under_future_suffix(self):
        b,t,r=1,9,128
        query=torch.randn(b,t,r)
        value=torch.randn(b,t,r)
        role=torch.randn(b,t,len(SEMANTIC_ROLES))
        scores=(query@query.transpose(1,2))/(r**.5)
        router=PaidThreeWayRouter()
        with torch.no_grad():
            router.weights.copy_(torch.randn(3,12)/5)
            router.bias.copy_(torch.tensor([.04,.01,-.06]))
        def forward(n):
            valid=torch.ones(b,n,dtype=torch.bool)
            return parallel_three_relations(
                query=query[:,:n],value=value[:,:n],
                scores=scores[:,:n,:n],
                allowed=torch.tril(torch.ones(n,n,dtype=torch.bool))[None],
                valid=valid,semantic_role_logits=role[:,:n],
                local_window=4,selector_version=3,paid_router=router)
        shorter=forward(5)
        longer=forward(9)
        self.assertTrue(torch.allclose(shorter.hypothesis_gates,
                         longer.hypothesis_gates[:,:5],atol=2e-6))
        self.assertTrue(torch.allclose(shorter.router_features,
                         longer.router_features[:,:5],atol=2e-6))
        self.assertTrue(torch.allclose(shorter.message,
                         longer.message[:,:5],atol=2e-6))

    def test_no_unbilled_router_for_original_versions(self):
        config=NRA3Config(parallel_hypotheses=3,parallel_selector_version=3)
        area=NRA3NeuralArea(config).eval()
        self.assertEqual(area.parameter_count(),6_106_154)
        with self.assertRaises(ValueError):
            area(torch.randn(1,4,320))
        q=torch.randn(1,4,128)
        scores=q@q.transpose(1,2)/(128**.5)
        allow=torch.tril(torch.ones(4,4,dtype=torch.bool))[None]
        valid=torch.ones(1,4,dtype=torch.bool)
        role=torch.randn(1,4,len(SEMANTIC_ROLES))
        with self.assertRaises(ValueError):
            parallel_three_relations(query=q,value=q,scores=scores,
                allowed=allow,valid=valid,semantic_role_logits=role,
                selector_version=3)
        with self.assertRaises(ValueError):
            parallel_three_relations(query=q,value=q,scores=scores,
                allowed=allow,valid=valid,semantic_role_logits=role,
                selector_version=2,paid_router=PaidThreeWayRouter())
        with self.assertRaises(ValueError):
            NRA3Config(parallel_selector_version=3,parallel_hypotheses=1)

    def test_training_optimizer_exclusively_small_physical_selector(self):
        x=torch.randn(90,12)
        teacher=(x[:,0]+x[:,1]*.5).bucketize(torch.tensor([-.5,.5]))
        physical,report=train_balanced_router(x,teacher,
                                              optimizer_steps=120,lr=.02)
        self.assertEqual(len(physical),123)
        self.assertEqual(report["extra_physical_parameters"],39)
        self.assertEqual(report["candidate_accepted"],False)
        self.assertEqual(report["output_gate"],0)
        self.assertLess(report["router_train_only_after_class_balanced_ce"],
                        report["router_train_only_before_class_balanced_ce"])
        native=PaidThreeWayRouter.from_physical(physical)
        with torch.inference_mode():
            self.assertGreater(float((native(x).argmax(-1)==teacher).float().mean()),
                               .5)

    def test_s9_72_only_teacher_all_three_with_no_future_head_gold(self):
        rows=build_s9_train72()
        counts=torch.zeros(3,dtype=torch.long)
        for row in rows:
            wordmap=tuple(AlignedWord(i+1,(i+1,),i+1)
                          for i in range(len(row.words)))
            align=CausalAlignment(row.sha256(),len(row.words)+2,
                tuple(range(len(row.words)+2)),wordmap,
                len(row.words)+1,False)
            labels=weak_train_router_labels(row,align)
            counts+=torch.bincount(labels,minlength=3)
            self.assertEqual(len(labels),len(row.words))
            with self.assertRaises(ValueError):
                weak_train_router_labels(dataclasses.replace(row,split="EVAL"),align)
            with self.assertRaises(ValueError):
                weak_train_router_labels(
                    dataclasses.replace(row,roles=("NONE",)+row.roles[1:]),align)
        self.assertEqual(len(rows),72)
        self.assertEqual(len(BRANCHES),3)
        self.assertTrue(bool((counts>0).all()))

    def test_weak_label_rejects_explicit_eval_and_future_offsets(self):
        row=build_s9_train72()[1]
        words=tuple(AlignedWord(i+1,(i+1,),i+1)
                    for i in range(len(row.words)))
        align=CausalAlignment(row.sha256(),len(row.words)+2,
            tuple(range(len(row.words)+2)),words,len(row.words)+1,False)
        with self.assertRaises(ValueError):
            weak_train_router_labels(
                row,dataclasses.replace(align,independent_eval_labels_loaded=True))
        bad=dataclasses.replace(align,
              word_alignment=(dataclasses.replace(words[0],last_token=10_000),)+words[1:])
        with self.assertRaises(ValueError):
            weak_train_router_labels(row,bad)


if __name__=="__main__":
    unittest.main()
