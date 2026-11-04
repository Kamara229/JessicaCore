"""No new policy weights for 6-layer, 60-layer or 3000-layer roles."""
import unittest
import torch
from torch import nn
from types import SimpleNamespace
from v123.neox_adapter import NeoXAdapter
from v122.policy import SharedGreenPolicy

class FakeBlock(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.attention=nn.Module()
        self.attention.dense=nn.Linear(n,n,bias=False)
        self.mlp=nn.Module()
        self.mlp.dense_4h_to_h=nn.Linear(n,n,bias=False)
class FakeNeoX(nn.Module):
    def __init__(self,depth,width):
        super().__init__()
        self.config=SimpleNamespace(num_hidden_layers=depth)
        self.gpt_neox=nn.Module()
        self.gpt_neox.layers=nn.ModuleList(FakeBlock(width) for _ in range(depth))

class Tests(unittest.TestCase):
    def test_two_different_host_geometry_same_neural_param_count(self):
        torch.set_num_threads(2)
        policy=SharedGreenPolicy().eval()
        n=sum(p.numel() for p in policy.parameters())
        for d,w in ((2,8),(6,32),(12,64)):
            model=FakeNeoX(d,w)
            adapter=NeoXAdapter(model)
            source=torch.randn(2,w)
            orig=model.gpt_neox.layers[0].attention.dense.source
            adapter.neutral()
            a=model.gpt_neox.layers[0].attention.dense(source)
            torch.testing.assert_close(a,orig(source))
            with torch.inference_mode():
                stats=adapter.set_question(policy,"Test sample prompt")
            self.assertEqual(stats["number_of_host_operator_roles"],2*d)
            self.assertEqual(sum(p.numel() for p in policy.parameters()),n)
            adapter.restore()
            self.assertIs(model.gpt_neox.layers[0].attention.dense,orig)
if __name__=="__main__":
    unittest.main()
