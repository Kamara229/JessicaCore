"""CPU-only synthetic tests: no trained CALIBRI weights and NO heldout QA."""
import io
import unittest
import torch
from torch import nn
from safetensors.torch import save_file,load_file
from tempfile import TemporaryDirectory
from pathlib import Path
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel,physical_bpp


class FakeBlock(nn.Module):
    def __init__(self,index):
        super().__init__()
        self.index=index
    def forward(self,h):
        # Original layer has a deterministic residual and tuple ABI.
        return (h+0.01*(self.index+1)*torch.tanh(h),)


class TwelveLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([FakeBlock(i) for i in range(12)])
    def forward(self,x):
        for layer in self.model.layers:
            x=layer(x)[0]
        return x


class S126FullLayerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(126)
        self.model=TwelveLayerModel().eval()
        self.gt=FullLayerStateTunnel(width=320,depth=12,rank=8)
        self.x=torch.randn(2,5,320)

    def tearDown(self):
        self.gt.uninstall()

    def test_zero_gate_exact_original_all_12(self):
        original=self.model(self.x).detach()
        self.gt.install(self.model)
        with self.gt.scope():
            actual=self.model(self.x)
        self.assertTrue(torch.equal(original,actual))
        self.assertEqual(self.gt.last_forward_coverage,tuple(range(12)))
        self.assertEqual(self.gt.new_parameter_count(),2*320*8+24)
        self.assertIsNone(self.gt._carry)

    def test_carry_changes_each_layer_with_nonzero_gates(self):
        original=self.model(self.x).detach()
        self.gt.install(self.model)
        with torch.no_grad():
            self.gt.gates.fill_(0.8)
        with self.gt.scope():
            out=self.model(self.x)
        self.assertFalse(torch.equal(original,out))
        self.assertEqual(self.gt.last_forward_coverage,tuple(range(12)))
        # Invocation cannot leak state between independent user requests.
        with self.gt.scope():
            again=self.model(self.x)
        self.assertTrue(torch.equal(out,again))
        with self.gt.scope(False):
            bypass=self.model(self.x)
        self.assertTrue(torch.equal(original,bypass))

    def test_true_gradients_and_all_original_frozen(self):
        self.gt.install(self.model)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),0)
        with self.gt.scope():
            out=self.model(self.x)
            out.square().mean().backward()
        self.assertIsNotNone(self.gt.gates.grad)
        self.assertGreater(int(torch.count_nonzero(self.gt.gates.grad)),0)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),0)

    def test_fp16_roundtrip_and_budget_is_full_physical(self):
        self.gt.install(self.model)
        with TemporaryDirectory() as temp:
            path=Path(temp)/"new_tunnel.safetensors"
            save_file(self.gt.export_fp16(),str(path))
            weights=load_file(str(path))
            second=FullLayerStateTunnel(width=320,depth=12,rank=8)
            second.load_fp16(weights)
            self.assertTrue(all(torch.equal(v,second.export_fp16()[k])
                                for k,v in weights.items()))
            self.assertEqual(set(weights),{
                "full_layer_tunnel.encoder","full_layer_tunnel.decoder",
                "full_layer_tunnel.gates","full_layer_tunnel.mix_logits"})
            # 36,380,480 original source parameters; this synthetic
            # calculation does NOT establish a safe S101 physical checkpoint.
            bpp=physical_bpp(3_000_000,path.stat().st_size,200,
                             36_380_480,second.new_parameter_count())
            self.assertLess(bpp,0.7)
            # Explicit accounting for two separate runtime components.
            self.assertGreater(bpp,8*3_000_000/(36_380_480+second.new_parameter_count()))

    def test_reject_invalid_depth_and_fp16_abi(self):
        with self.assertRaises(ValueError):
            FullLayerStateTunnel(depth=10)
        with self.assertRaises(ValueError):
            self.gt.load_fp16({"bad":torch.zeros(1,dtype=torch.float16)})
        with self.assertRaises(ValueError):
            physical_bpp(10,-1,3,36_380_480,0)


if __name__=="__main__":
    unittest.main()
