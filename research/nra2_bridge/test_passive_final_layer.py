"""Stage-2 bounded tests for a single *passive* 12th-layer read port."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from .passive_final_layer import NRA2PassiveFinalLayer
from research.nra2_standalone.model import RelationalNRA


class Toy12(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Identity() for _ in range(12)])
    def forward(self,x):
        for layer in self.model.layers:
            x=layer(x)
        return x


class NRA2Stage2Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261113)
        torch.set_num_threads(1)

    def fixture(self,folder):
        checkpoint=Path(folder)/"separate_fp16.pt"
        model=RelationalNRA(width=16,hops=5)
        torch.save({k:v.detach().half().cpu() for k,v in model.state_dict().items()},
                   checkpoint)
        return checkpoint,hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def test_passive_final_read_preserves_exact_320d_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,digest=self.fixture(tmp)
            core=Toy12()
            h=torch.randn(2,7,320)
            before=core(h).clone()
            tap=NRA2PassiveFinalLayer(file,digest).install(core)
            out=core(h)
            self.assertTrue(torch.equal(before,out))
            self.assertEqual(out.data_ptr(),h.data_ptr())
            self.assertEqual(tap.forward_calls,1)
            self.assertEqual(tap.last_shape,(2,7,320))
            self.assertEqual(tap.report["read_layer_index"],11)
            self.assertEqual(tap.report["connected_decoder_layers"],1)
            self.assertEqual(tap.report["standalone_trainable_parameters"],2241)
            self.assertTrue(tap.report["standalone_weights_frozen"])
            self.assertFalse(tap.report["relation_inference_enabled"])
            tap.uninstall()
            self.assertTrue(torch.equal(core(h),before))

    def test_no_implicit_nonzero_output_and_no_duplicate_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,digest=self.fixture(tmp)
            core=Toy12()
            tap=NRA2PassiveFinalLayer(file,digest).install(core)
            with self.assertRaises(RuntimeError):
                tap.install(core)
            tap.output_gate=.5
            with self.assertRaises(AssertionError):
                core(torch.randn(1,2,320))
            tap.output_gate=0
            tap.uninstall()
            self.assertEqual(core(torch.zeros(1,2,320)).shape,(1,2,320))

    def test_checksum_and_shape_contract_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,digest=self.fixture(tmp)
            with self.assertRaises(AssertionError):
                NRA2PassiveFinalLayer(file,"0"*64)
            tap=NRA2PassiveFinalLayer(file,digest)
            with self.assertRaises(AssertionError):
                tap.install(nn.Sequential(nn.Identity()))
            core=Toy12()
            tap.install(core)
            with self.assertRaises(AssertionError):
                core(torch.randn(1,4,319))
            tap.uninstall()


if __name__=="__main__":
    unittest.main()
