"""S3O neural query-role language grounding tests, no model or user CHAT."""
from __future__ import annotations
import unittest
import torch
from research.nra2_bridge.test_stage3j_token_entity_alignment import CharTokenizer
from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
from research.nra2_bridge.stage3k_neural_entity_bridge import bind_mentions_to_context
from research.nra2_bridge.stage3o_neural_query_roles import (
    build_query_text_mask,LearnedQueryRoleHead,S3KGraphWithLearnedRoles,
)
from research.nra2_bridge.stage3m_role_balanced_graph_data import build_rows


class Stage3ORoleBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.question_list=[
            "Connection log: R0 -> R1; R1 -> R2. Is there a directed path from R0 to R2? Reply yes or no.",
            "Connection log: R0 -> R1; R1 -> R2. Can R2 be reached from R0 following the arrows? Reply yes or no.",
            "Connection log: R0 -> R1; R1 -> R2. Starting at R0, can one travel to R2? Reply yes or no.",
            "Connection log: R0 -> R1; R1 -> R2. Is R2 downstream of R0 along one-way links? Reply yes or no.",
        ]
        cls.tok=CharTokenizer()
        cls.aligned=validate_without_gold(cls.tok,cls.question_list)
        cls.t=max(a["tokens"] for a in cls.aligned)
        cls.query=build_query_text_mask(cls.tok,cls.question_list,cls.aligned,cls.t)
        cls.bind=bind_mentions_to_context(cls.aligned,cls.t)
        cls.token_mask=torch.zeros((4,cls.t),dtype=torch.bool)
        for i,a in enumerate(cls.aligned):
            cls.token_mask[i,:a["tokens"]]=True

    def test_query_candidates_not_semantic_source_oracle(self):
        q=self.query
        self.assertEqual(q.node_order.tolist(),[[0,2],[2,0],[0,2],[2,0]])
        self.assertEqual(q.query_mask.shape,(4,self.t))
        self.assertEqual(q.anchor_mask.shape,(4,2,self.t))
        self.assertTrue(q.query_mask.any(1).all())
        self.assertTrue(q.anchor_mask.any(2).all())
        self.assertFalse((q.anchor_mask & ~q.query_mask[:,None,:]).any())
        self.assertFalse(hasattr(q,"source"))
        self.assertFalse(hasattr(q,"target"))
        self.assertFalse(hasattr(q,"truth"))

    def test_neural_query_role_gradients_and_differing_orders(self):
        torch.manual_seed(111)
        model=LearnedQueryRoleHead()
        states=torch.randn(4,self.t,320)
        src,dst=model(states,self.token_mask,self.query)
        self.assertEqual(src.shape,(4,6))
        self.assertEqual(dst.shape,(4,6))
        self.assertTrue(torch.isfinite(src).all())
        self.assertTrue(torch.isfinite(dst).all())
        loss=torch.nn.functional.cross_entropy(src,torch.tensor([0,0,0,0]))
        loss=loss+torch.nn.functional.cross_entropy(dst,torch.tensor([2,2,2,2]))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_joint_graph_edge_and_semantic_role_gradients(self):
        torch.manual_seed(177)
        model=S3KGraphWithLearnedRoles()
        tokens=torch.randn((4,self.t,320))
        edge,src,dst=model(tokens,self.token_mask,self.bind,self.query)
        self.assertEqual(edge.shape,(4,6,6))
        self.assertTrue(torch.isfinite(edge).all())
        truth=torch.zeros((4,6,6))
        truth[:,0,1]=1
        truth[:,1,2]=1
        offdiag=~torch.eye(6,dtype=torch.bool)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(
            edge[:,offdiag],truth[:,offdiag]
        )+torch.nn.functional.cross_entropy(src,torch.zeros(4,dtype=torch.long))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.graph.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.query.parameters()))

    def test_no_gold_or_role_order_from_dataset_in_query_inference(self):
        rows=build_rows()
        tok=CharTokenizer()
        aligned=validate_without_gold(tok,[r.question for r in rows[:128]])
        maximum=max(a["tokens"] for a in aligned)
        query=build_query_text_mask(tok,[r.question for r in rows[:128]],aligned,maximum)
        self.assertEqual(query.node_order.shape,(128,2))
        self.assertTrue(query.anchor_mask.any(-1).all())
        self.assertTrue(((query.relative_positions>=0)
                         &(query.relative_positions<=1)).all())
        self.assertFalse(hasattr(query,"reachable"))

    def test_fail_closed_on_more_than_two_query_entities(self):
        tok=CharTokenizer()
        prompts=["Graph: R0 -> R1. Is there a path from R0 through R2 to R1? Reply yes or no."]
        a=validate_without_gold(tok,prompts)
        with self.assertRaisesRegex(AssertionError,"exactly 2"):
            build_query_text_mask(tok,prompts,a,a[0]["tokens"])


if __name__=="__main__":
    unittest.main()
