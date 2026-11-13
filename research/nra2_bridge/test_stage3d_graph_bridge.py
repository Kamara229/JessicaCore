"""S3D CPU-only independent unit tests. No Calibri weights, LIVE or QA."""
import unittest
import torch
import torch.nn.functional as F

from research.nra2_standalone.model import RelationalNRA
from research.nra2_bridge.stage3d_graph_bridge import (
    Hidden320GraphBridge,normalized_hidden,logits_from_bridge,
    WIDTH,N,TRAIN,HELD,
)


class GraphBridgeS3DTests(unittest.TestCase):
    def test_hidden_and_graph_shape_guard(self):
        bridge=Hidden320GraphBridge()
        edges,src,dst=bridge(torch.randn(3,WIDTH))
        self.assertEqual(tuple(edges.shape),(3,N,N))
        self.assertEqual(tuple(src.shape),(3,N))
        self.assertEqual(tuple(dst.shape),(3,N))
        with self.assertRaises(AssertionError):
            bridge(torch.zeros(3,WIDTH-1))
        with self.assertRaises(AssertionError):
            bridge(torch.full((3,WIDTH),float("nan")))

    def test_gradients_reach_only_shadow_bridge(self):
        torch.manual_seed(137)
        bridge=Hidden320GraphBridge()
        frozen=RelationalNRA(width=16,hops=5).eval().requires_grad_(False)
        old={k:v.clone() for k,v in frozen.state_dict().items()}
        x=torch.randn(TRAIN,WIDTH)
        e,src,dst,nra=logits_from_bridge(bridge,frozen,x)
        labels=torch.randint(0,2,(TRAIN,),dtype=torch.float32)
        loss=(e.square().mean()+src.square().mean()+dst.square().mean()
              +F.binary_cross_entropy_with_logits(nra,labels))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in bridge.parameters()))
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))
        self.assertTrue(all(torch.equal(v,old[k])
                            for k,v in frozen.state_dict().items()))

    def test_train_mean_and_std_are_explicit(self):
        tr=torch.arange(TRAIN*WIDTH,dtype=torch.float32).reshape(TRAIN,WIDTH)
        mean=tr.mean(0)
        scale=tr.std(0,unbiased=False).clamp(min=.05)
        train=normalized_hidden(tr,mean,scale)
        self.assertEqual(tuple(train.shape),(TRAIN,WIDTH))
        self.assertTrue(torch.allclose(train.mean(0),torch.zeros(WIDTH),atol=1e-5))
        with self.assertRaises(AssertionError):
            normalized_hidden(tr,mean,torch.zeros_like(scale))


if __name__=="__main__":
    unittest.main()
