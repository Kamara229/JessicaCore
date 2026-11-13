"""S3F CPU-only tests; no Calibri weights or data artefacts loaded."""
import unittest
import torch
import torch.nn.functional as F

from research.nra2_standalone.model import RelationalNRA
from research.nra2_bridge.stage3f_token_attention import (
    TokenAttentionToGraph,graph_forward,normalize,N,WIDTH,
)


class S3FTokenAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1411)
        torch.set_num_threads(2)

    def test_exact_shape_and_padding_independence(self):
        m=TokenAttentionToGraph().eval()
        x=torch.randn(2,12,WIDTH)
        mask=torch.ones(2,12,dtype=torch.bool)
        with torch.no_grad():
            a=m(x,mask)
            z=torch.cat([x,torch.randn(2,4,WIDTH)],dim=1)
            mz=torch.cat([mask,torch.zeros(2,4,dtype=torch.bool)],dim=1)
            b=m(z,mz)
        self.assertEqual(tuple(a[0].shape),(2,N,N))
        self.assertEqual(tuple(a[1].shape),(2,N))
        self.assertEqual(tuple(a[2].shape),(2,N))
        for aa,bb in zip(a,b):
            self.assertTrue(torch.allclose(aa,bb,atol=1e-5,rtol=1e-5))

    def test_no_nra_gradients_zero_gate_compatible(self):
        network=TokenAttentionToGraph()
        frozen=RelationalNRA(width=16,hops=5).eval().requires_grad_(False)
        copy={k:v.clone() for k,v in frozen.state_dict().items()}
        x=torch.randn(3,10,WIDTH)
        mask=torch.ones(3,10,dtype=torch.bool)
        e,s,t,r=graph_forward(network,frozen,x,mask)
        loss=e.square().mean()+s.square().mean()+t.square().mean()+F.binary_cross_entropy_with_logits(r,torch.tensor([0.,1.,0.]))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in network.parameters()))
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))
        self.assertTrue(all(torch.equal(v,copy[k]) for k,v in frozen.state_dict().items()))

    def test_fail_closed_bad_mask_shape_or_nonfinite(self):
        net=TokenAttentionToGraph()
        x=torch.randn(2,7,WIDTH)
        with self.assertRaises(AssertionError):
            net(x,torch.ones(2,6,dtype=torch.bool))
        with self.assertRaises(AssertionError):
            net(x,torch.zeros(2,7,dtype=torch.bool))
        with self.assertRaises(AssertionError):
            net(x,torch.ones(2,7,dtype=torch.float32))
        with self.assertRaises(AssertionError):
            net(torch.full_like(x,float("nan")),torch.ones(2,7,dtype=torch.bool))
        with self.assertRaises(AssertionError):
            normalize(x,torch.zeros(WIDTH),torch.zeros(WIDTH),torch.ones(2,7,dtype=torch.bool))


if __name__=="__main__":
    unittest.main()
