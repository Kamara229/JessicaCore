"""Unit tests for exact passive 320D read-port capture, no target leakage."""
import hashlib
import unittest
from pathlib import Path
import tempfile

import torch
from torch import nn
from safetensors.torch import save_file,load_file
from .frozen_features import FrozenFinalFeatures,prompt_sha
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


class FrozenFeaturesContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(352026)
        torch.set_num_threads(1)

    def test_exact_frozen_output_and_token_320_features(self):
        m=Toy12().eval()
        tap=FrozenFinalFeatures().install(m)
        a,b="Unseen TRAIN example A","Unseen TRAIN example B"
        x=torch.randn(1,8,320)
        before=m(x).clone()
        with tap.prompt_scope(a):
            after=m(x)
            # This mimics step1/step2 cached tokens from same input identity.
            _=m(torch.randn(1,1,320))
        self.assertTrue(torch.equal(after,before))
        self.assertEqual(after.data_ptr(),x.data_ptr())
        y=x+0.25
        with tap.prompt_scope(b):
            self.assertTrue(torch.equal(m(y),y))
        h=tap.select(sorted([prompt_sha(a),prompt_sha(b)]))
        self.assertEqual(tuple(h.shape),(2,320))
        self.assertEqual(h.dtype,torch.float16)
        self.assertEqual(tap.observed_tagged_calls,3)
        self.assertEqual(len(tap.features),2)
        self.assertTrue(torch.equal(tap.features[prompt_sha(a)],x[0,-1].half()))
        self.assertTrue(torch.equal(tap.features[prompt_sha(b)],y[0,-1].half()))
        tap.uninstall()
        self.assertTrue(torch.equal(m(x),x))

    def test_unscoped_inputs_not_captured_and_fail_closed(self):
        m=Toy12()
        tap=FrozenFinalFeatures().install(m)
        m(torch.randn(1,3,320))
        self.assertFalse(tap.features)
        with tap.prompt_scope("train only"):
            with self.assertRaises(RuntimeError):
                with tap.prompt_scope("nested"):
                    pass
            with self.assertRaises(RuntimeError):
                tap.uninstall()
        with tap.prompt_scope("train only"):
            with self.assertRaises(AssertionError):
                m(torch.randn(1,3,319))
        with self.assertRaises(AssertionError):
            tap.select([prompt_sha("never seen")])
        tap.uninstall()

    def test_single_layer_learned_NRA_attached_in_parallel_no_model_change(self):
        with tempfile.TemporaryDirectory() as td:
            ck=Path(td)/"standalone_nra.pt"
            nra=RelationalNRA()
            torch.save({k:v.detach().half() for k,v in nra.state_dict().items()},ck)
            digest=hashlib.sha256(ck.read_bytes()).hexdigest()
            core=Toy12()
            read=NRA2PassiveFinalLayer(ck,digest).install(core)
            tap=FrozenFinalFeatures().install(core)
            sample=torch.randn(1,6,320)
            with tap.prompt_scope("NRA2 real bridge SAMPLE TRAIN"):
                got=core(sample)
            self.assertTrue(torch.equal(got,sample))
            self.assertEqual(read.report["connected_decoder_layers"],1)
            self.assertFalse(read.report["relation_inference_enabled"])
            h=tap.select([prompt_sha("NRA2 real bridge SAMPLE TRAIN")])
            path=Path(td)/"features.safetensors"
            save_file({"final_hidden_320":h},str(path))
            physical=load_file(str(path))["final_hidden_320"]
            self.assertTrue(torch.equal(h,physical))
            tap.uninstall()
            read.uninstall()


if __name__=="__main__":
    unittest.main()
