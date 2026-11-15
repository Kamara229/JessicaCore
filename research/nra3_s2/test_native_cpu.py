"""S2: real Calibri C CB59 kernel executes all NRA3 matrices, offline only.

No pretrained Calibri 43M loads, no TRAIN/DEV/EVAL/J7, no read/write gain.
Comparisons are against the reconstructed *same quantized* source, not FP32
quality. Green Tunnel modules are original S87 and S103, on a toy decoder.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from torch import nn

from research.nra3_s0.neural_area import NRA3Config,NRA3NeuralArea
from research.nra3_s1.physical_cb59 import encode_area,reconstruct_state
from research.nra3_s1.green_tunnel_compatibility import (
    NRA3WithExistingGreenTunnel,module_fingerprint,
)
from research.nra3_s2.native_cpu import NRA3NativePackedCPU,expected_shapes
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel
from calibri2.s103.shared_all_layer_green_bridge import SharedInterLayerGreenBridge


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Identity() for _ in range(12)])
    def forward(self,x):
        for layer in self.model.layers:
            x=layer(x)
        return x


class NativeCB59NRA3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(30202)
        cls.td=tempfile.TemporaryDirectory()
        cls.library=Path(cls.td.name)/"lib_nra3_cb59_native.so"
        source=Path("research/calibri/direct_kernel_v59.c")
        subprocess.run(["cc","-O3","-fPIC","-shared","-std=c11","-Wall",
                        "-Wextra","-Werror",str(source),"-o",str(cls.library)],
                       check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    @staticmethod
    def small_config():
        return NRA3Config(hidden=32,heads=4,grammar_blocks=1,
                          semantic_blocks=1,relation_width=24,relation_steps=2)

    def packed(self,config):
        parent=NRA3NeuralArea(config).eval()
        source,meta,_=encode_area(parent)
        decoded=NRA3NeuralArea(config).eval()
        decoded.load_state_dict(reconstruct_state(source,parent),strict=True)
        native=NRA3NativePackedCPU(source,config,self.library).eval()
        self.assertEqual(set(expected_shapes(config)),set(parent.state_dict()))
        self.assertEqual(native.unique_parameters,parent.parameter_count())
        self.assertFalse(native.native_cpu_weights_dense_resident)
        self.assertEqual(len(native.state_dict()),0)
        self.assertEqual(len(list(native.parameters())),0)
        self.assertEqual(native.source_sha256,hashlib.sha256(source).hexdigest())
        self.assertGreater(native.packed_matrix_count(),10)
        return source,decoded,native

    def check_feature_agreement(self,decoded,native,x,valid=None):
        with torch.inference_mode():
            reference=decoded(x,valid)
            observed=native(x,valid)
        for name in ("grammar","semantics","relations","upos_logits",
                     "morph_logits","dependency_logits","semantic_role_logits",
                     "antecedent_logits","readout"):
            old=getattr(reference,name)
            new=getattr(observed,name)
            self.assertEqual(tuple(old.shape),tuple(new.shape),name)
            self.assertTrue(torch.allclose(old,new,atol=3e-4,rtol=4e-4),
                            name+f": maxerr={float((old-new).abs().max())}")

    def test_all_packed_operators_match_their_own_quantized_dense_reference(self):
        _,dense,native=self.packed(self.small_config())
        self.check_feature_agreement(dense,native,torch.randn(1,5,320))
        self.check_feature_agreement(dense,native,torch.randn(2,6,320),
                                      torch.tensor([[True]*6,
                                                    [True]*4+[False]*2]))

    def test_causal_prefix_and_no_fp32_fallback(self):
        _,dense,native=self.packed(self.small_config())
        x=torch.randn(1,5,320)
        y=x.clone();y[:,3:]+=30
        with torch.inference_mode():
            a=native(x)
            b=native(y)
        for n in ("grammar","semantics","relations","semantic_role_logits"):
            self.assertTrue(torch.allclose(getattr(a,n)[:,:3],
                                           getattr(b,n)[:,:3],
                                           atol=1e-5,rtol=0),n)
        self.assertFalse(native.native_cpu_weights_dense_resident)
        self.assertEqual(len(list(native.named_parameters())),0)
        with self.assertRaises(ValueError):
            native(torch.randn(1,3,321))
        with self.assertRaises(ValueError):
            native(torch.randn(1,3,320).half())
        with self.assertRaises(RuntimeError):
            native.forward_train(torch.randn(1,3,320))

    def test_green_tunnel_and_shared_all_layer_routes_exact_zero_gate(self):
        _,dense,native=self.packed(self.small_config())
        model=ToyDecoder()
        tunnel=IntegratedGreenTunnel(width=320,rank=8,layers=(3,9)).install(model)
        bridge=SharedInterLayerGreenBridge(tunnel).install(model)
        with torch.no_grad():
            tunnel.blocks["3"].up.weight.fill_(0.04)
            bridge.layer_gains[bridge.additional_layers.index(7)]=0.3
        gt_sha=module_fingerprint(tunnel)
        bp_sha=module_fingerprint(bridge)
        combined=NRA3WithExistingGreenTunnel(native,green_tunnel=tunnel,
                                              shared_bridge=bridge).install(model)
        x=torch.randn(1,3,320)
        try:
            with torch.inference_mode():
                for route in (None,"S87","S91","S93"):
                    with bridge.profile(route):
                        baseline=model(x)
                        with combined.port.activated():
                            shadow=model(x)
                        self.assertTrue(torch.equal(baseline,shadow),
                                        f"GT route {route}")
                        self.assertEqual(combined.port.last_features["roles"],
                                         (1,3,9))
            self.assertEqual(gt_sha,module_fingerprint(tunnel))
            self.assertEqual(bp_sha,module_fingerprint(bridge))
            self.assertTrue(combined.integrity()["existing_green_tunnel_unchanged"])
            with self.assertRaises(RuntimeError):
                combined.port.set_output_gate(1.0)
        finally:
            combined.uninstall()
            bridge.uninstall()
            tunnel.uninstall()

    def test_complete_6m_area_native_cpu_one_token_vs_quantized_reference(self):
        cfg=NRA3Config()
        source,decoded,native=self.packed(cfg)
        self.assertEqual(native.unique_parameters,6_106_154)
        self.assertEqual(native.packed_matrix_count(),43)
        self.assertLess(len(source),750_000)
        self.check_feature_agreement(decoded,native,torch.randn(1,1,320))


if __name__=="__main__":
    unittest.main()
