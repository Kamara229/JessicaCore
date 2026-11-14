"""S3BM architectural invariants only. NO S3AQ/S3BJ spent EVAL data."""
from __future__ import annotations
from dataclasses import replace
import unittest
import torch
import torch.nn.functional as F

from research.nra2_bridge.test_stage3j_token_entity_alignment import CharTokenizer
from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
from research.nra2_bridge.stage3o_neural_query_roles import build_query_text_mask
from research.nra2_bridge.stage3k_neural_entity_bridge import bind_mentions_to_context
from research.nra2_bridge.stage3bm_joint_semantic_pair_role_head import (
    JointSemanticPairRoleHead,S3BMGraphWithJointRoles,
)

class JointRoleArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries=[
            "Connection log: R0 -> R1; R1 -> R2. Can R2 be reached from R0 along directed paths? Reply yes or no.",
            "Connection log: R0 -> R1; R1 -> R2. Starting at R0, can a path reach R2? Reply yes or no.",
            "Connection log: R0 -> R1; R1 -> R2. From R0 is R2 accessible along the arrows? Reply yes or no.",
            "Connection log: R0 -> R1; R1 -> R2. Does arrival at R2 follow from departure at R0? Reply yes or no.",
        ]
        cls.tok=CharTokenizer()
        cls.align=validate_without_gold(cls.tok,cls.queries)
        cls.t=max(r["tokens"] for r in cls.align)
        cls.q=build_query_text_mask(cls.tok,cls.queries,cls.align,cls.t)
        cls.binding=bind_mentions_to_context(cls.align,cls.t)
        cls.mask=torch.zeros((4,cls.t),dtype=torch.bool)
        for i,a in enumerate(cls.align):
            cls.mask[i,:a["tokens"]]=True
        torch.manual_seed(2026111518)
        cls.tokens=torch.randn((4,cls.t,320))

    @staticmethod
    def swap_candidates(q):
        flip=lambda x:x[:,[1,0],:] if x.ndim==3 else x[:,[1,0]]
        return replace(q,anchor_mask=flip(q.anchor_mask),
                       left_context_mask=flip(q.left_context_mask),
                       right_context_mask=flip(q.right_context_mask),
                       node_order=flip(q.node_order),
                       relative_positions=flip(q.relative_positions))

    def test_exact_frozen_baseline_at_initialization(self):
        h=JointSemanticPairRoleHead().eval()
        self.assertTrue(all(not p.requires_grad for p in h.base.parameters()))
        with torch.no_grad():
            a,b=h.base(self.tokens,self.mask,self.q)
            c,d=h(self.tokens,self.mask,self.q)
        self.assertTrue(torch.equal(a,c))
        self.assertTrue(torch.equal(b,d))

    def test_swapping_candidate_indices_is_true_equivariance(self):
        h=JointSemanticPairRoleHead().eval()
        with torch.no_grad():
            torch.nn.init.normal_(h.pair_score[-1].weight,mean=0.0,std=0.075)
            torch.nn.init.normal_(h.pair_score[-1].bias,mean=0.0,std=0.03)
            src,dst=h(self.tokens,self.mask,self.q)
            swapped=self.swap_candidates(self.q)
            src2,dst2=h(self.tokens,self.mask,swapped)
        self.assertTrue(torch.allclose(src,src2,atol=2e-5,rtol=1e-5))
        self.assertTrue(torch.allclose(dst,dst2,atol=2e-5,rtol=1e-5))
        self.assertTrue(torch.isfinite(src).all())

    def test_new_joint_layers_receive_label_supervision_not_legacy(self):
        h=JointSemanticPairRoleHead()
        src,dst=h(self.tokens,self.mask,self.q)
        truth_src=torch.zeros(4,dtype=torch.long)
        truth_dst=torch.full((4,),2,dtype=torch.long)
        loss=F.cross_entropy(src,truth_src)+F.cross_entropy(dst,truth_dst)
        loss.backward()
        self.assertTrue(h.pair_score[-1].weight.grad is not None)
        self.assertGreater(h.pair_score[-1].weight.grad.abs().sum().item(),0.0)
        self.assertTrue(all(p.grad is None for p in h.base.parameters()))

    def test_graph_branch_independent_and_frozen(self):
        torch.manual_seed(2026111519)
        m=S3BMGraphWithJointRoles().eval()
        m.graph.requires_grad_(False)
        with torch.no_grad():
            edges,src,dst=m(self.tokens,self.mask,self.binding,self.q)
        self.assertEqual(tuple(edges.shape),(4,6,6))
        self.assertEqual(tuple(src.shape),(4,6))
        self.assertEqual(tuple(dst.shape),(4,6))
        self.assertTrue(torch.isfinite(edges).all())
        self.assertFalse(any(p.requires_grad for p in m.graph.parameters()))

    def test_reject_gold_like_unbound_text_masks(self):
        h=JointSemanticPairRoleHead()
        q=replace(self.q,node_order=self.q.node_order.clone())
        q.node_order[0,1]=q.node_order[0,0]
        with self.assertRaises(AssertionError):
            h(self.tokens,self.mask,q)
        q=replace(self.q,query_mask=self.q.query_mask.clone())
        q.query_mask[0,-1]=True
        bad_mask=self.mask.clone()
        bad_mask[0,-1]=False
        with self.assertRaises(AssertionError):
            h(self.tokens,bad_mask,q)

if __name__=="__main__":
    unittest.main()
