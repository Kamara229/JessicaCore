#!/usr/bin/env python3
"""Real native-C toy operator and source packaging; no network or 135M download."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch
from direct_codebook_v59 import CB59,encode
from direct_runtime_v59 import NativeKernel,PackedLinear,compile_kernel

from v84.archive import rewrite_codebooks
from v85 import physical
from v85.gate_train import fit_activation_gate
from v85.native_ops import AdaptivePackedLinear
from v85.validation import variant_sets

NAME="model.layers.11.mlp.down_proj"


class AdaptiveOriginalOperatorTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(8501)
        cls.original,_,_=encode({NAME+".weight":torch.randn(16,16).float()})
        c=CB59(cls.original)
        cls.original_handle=c.handles[NAME+".weight"]
        cls.modified_centers=cls.original_handle.centroids.copy()*1.2
        cls.modified=rewrite_codebooks(cls.original,{NAME:cls.modified_centers})

    def test_entire_coded_source_and_trailing_gates_integrity(self):
        w=np.arange(16,dtype=np.float32)*.01
        gates={NAME:(w,np.array([-.33],dtype=np.float32))}
        with patch.object(physical,"SOURCE_BYTES",len(self.original)):
            packed=physical.create_packed(self.original,self.modified,gates)
            decoded=physical.CB85(packed)
            self.assertEqual(decoded.original.hash(),CB59(self.original).hash())
            self.assertEqual(len(decoded.extra),1)
            self.assertTrue(np.array_equal(decoded.extra[NAME][1],
                w.astype("<f2").astype(np.float32)))
            self.assertEqual(len(decoded.blob),len(packed))
            self.assertGreater(len(packed),len(self.original))
            corrupted=bytearray(packed)
            corrupted[-50]^=1
            with self.assertRaises(ValueError):
                physical.CB85(bytes(corrupted))

    def test_dynamic_native_operator_matches_two_real_C_branches(self):
        with TemporaryDirectory() as td:
            ker=NativeKernel(compile_kernel(
                Path(__file__).with_name("direct_kernel_v59.c"),
                Path(td)/"adaptive_native_test.so"))
            basemodule=PackedLinear(self.original_handle,ker)
            w=np.ones(16,dtype=np.float32)*.12
            gate=AdaptivePackedLinear(basemodule,self.modified_centers,w,-.15)
            x=torch.randn(5,16)
            with torch.inference_mode():
                result=gate(x)
                norm=x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-6)
                alpha=torch.sigmoid((norm*gate.gate_w).sum(-1,keepdim=True)+gate.gate_b)
                expected=gate.base(x)+(gate.experimental(x)-gate.base(x))*alpha
            self.assertLess(float((result-expected).abs().max()),1e-5)
            self.assertGreater(float(alpha.std()),0.001)

    def test_actual_student_activation_gate_optimizer(self):
        torch.manual_seed(8502)
        n=256
        x=torch.randn(n,16)
        y0=torch.randn(n,4)
        delta=torch.randn(n,4)
        norm=x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-6)
        truth=torch.sigmoid(norm[:,0]*0.5+0.2)
        target=y0+truth[:,None]*delta
        (_,b),stats=fit_activation_gate(x,target,y0,y0+delta)
        self.assertGreater(stats["TRAIN_student_activation_examples"],200)
        self.assertLess(stats["trained_gated_TRAIN_relative_squared_error"],
                        stats["original_student_TRAIN_relative_squared_error"])
        self.assertTrue(np.isfinite(b).all())

    def test_variant_sets_predeclared_not_looked_up_from_test(self):
        paths=("a","b","c")
        variants=variant_sets(paths)
        self.assertEqual(len(variants),7)
        self.assertEqual(variants[-1][1],paths)


if __name__=="__main__":
    main()
