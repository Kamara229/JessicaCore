"""S3K self-contained CPU tests: entity anchoring, no oracle or LIVE."""
from __future__ import annotations
import unittest
import torch
from research.nra2_bridge.stage3j_token_entity_alignment import node_mention_alignment
from research.nra2_bridge.test_stage3j_token_entity_alignment import CharTokenizer
from research.nra2_bridge.stage3k_neural_entity_bridge import (
    bind_mentions_to_context,NodeSpanBoundGraph,NODES,WIDTH,
)


class Stage3KNeuralBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=CharTokenizer()
        cls.prompts=[
            "Directed hops: R0 -> R1; R1 -> R2. Can R0 reach R2? Reply yes or no.",
            "Directed hops: R0 -> R1; R1 -> R2. Can R2 reach R0? Reply yes or no.",
            "Transfer registry: R3 -> R2; R0 -> R3. Is R2 downstream of R0? Reply yes or no.",
        ]
        cls.aligned=[node_mention_alignment(cls.tok,q) for q in cls.prompts]
        cls.t=max(x["tokens"] for x in cls.aligned)
        cls.binding=bind_mentions_to_context(cls.aligned,cls.t)
        cls.mask=torch.zeros((3,cls.t),dtype=torch.bool)
        for n,a in enumerate(cls.aligned):
            cls.mask[n,:a["tokens"]]=True

    def test_alignment_only_in_no_graph_gold(self):
        b=self.binding
        self.assertEqual(b.mention_mask.shape,(3,self.t,NODES))
        self.assertEqual(b.event_pairs.shape[0],3)
        self.assertEqual(b.event_mask.shape[2],self.t)
        self.assertEqual(b.query_order.tolist(),[[0,2],[2,0],[2,0]])
        self.assertTrue((b.mention_mask.sum(-1)<=1).all())
        self.assertFalse((b.event_mask & b.mention_mask.any(-1).unsqueeze(1)).any())
        for k,v in vars(b).items():
            self.assertTrue(isinstance(v,torch.Tensor),k)

    def test_neural_forward_real_320_abi_and_trainable_gradient(self):
        torch.manual_seed(20261114)
        model=NodeSpanBoundGraph()
        states=torch.randn((3,self.t,WIDTH),dtype=torch.float32)
        edge,src,dst=model(states,self.mask,self.binding)
        self.assertEqual(edge.shape,(3,6,6))
        self.assertEqual(src.shape,(3,6))
        self.assertEqual(dst.shape,(3,6))
        self.assertTrue(torch.isfinite(edge).all())
        self.assertTrue(torch.isfinite(src).all())
        self.assertTrue(torch.isfinite(dst).all())
        truth=torch.zeros((3,NODES,NODES))
        truth[0,0,1]=1.
        truth[0,1,2]=1.
        truth[1,0,1]=1.
        truth[1,1,2]=1.
        truth[2,0,3]=1.
        truth[2,3,2]=1.
        offdiag=~torch.eye(NODES,dtype=torch.bool)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(
            edge[:,offdiag],truth[:,offdiag]) + (
            torch.nn.functional.cross_entropy(src,torch.tensor([0,2,0]))
        ) + torch.nn.functional.cross_entropy(dst,torch.tensor([2,0,2]))
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_same_graph_reversed_query_not_a_gold_edge(self):
        b=self.binding
        # Both graph presentations have the SAME edge description and
        # DIFFERENT queries; the prompt parser may not use answer gold.
        first={(int(a),int(c)) for a,c in b.event_pairs[0][b.event_valid[0]].tolist()}
        second={(int(a),int(c)) for a,c in b.event_pairs[1][b.event_valid[1]].tolist()}
        self.assertIn((0,1),first&second)
        self.assertIn((1,2),first&second)
        self.assertIn((2,0),second)
        self.assertEqual(b.query_order[:2].tolist(),[[0,2],[2,0]])

    def test_mask_and_padding_fail_closed(self):
        model=NodeSpanBoundGraph().eval()
        states=torch.zeros((3,self.t,WIDTH))
        bad_mask=self.mask.clone()
        bad_mask[0,-1]=False
        # Mark a literal mention position as invalid.
        idx=int(self.binding.mention_mask[0].any(-1).nonzero()[0])
        bad_mask[0,idx]=False
        with self.assertRaisesRegex(AssertionError,"invalid or padded"):
            model(states,bad_mask,self.binding)
        with self.assertRaisesRegex(AssertionError,"token length mismatch"):
            bind_mentions_to_context(self.aligned,self.t-40)


if __name__=="__main__":
    unittest.main()
