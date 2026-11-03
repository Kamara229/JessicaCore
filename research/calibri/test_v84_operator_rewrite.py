#!/usr/bin/env python3
"""Unit tests of actual paid archive integrity, trainable operator and guard."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from direct_codebook_v59 import CB59, encode
from direct_runtime_v59 import NativeKernel, compile_kernel, PackedLinear
from v84.archive import rewrite_codebooks
from v84.fit import Projection, train_codebook, local_relative_error, CHECKPOINTS
from v84.valid import choose_guarded_variant


TEST_NAME = "model.layers.11.mlp.down_proj"


class RealCB59Rewrites(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(8401)
        cls.original,_,_ = encode({
            TEST_NAME+".weight": torch.randn(16, 16).float()})
        cls.decoder = CB59(cls.original)

    def test_rewrite_is_physically_same_size_and_reparsable(self):
        a=self.decoder.handles[TEST_NAME+".weight"]
        changed=a.centroids.copy()
        changed[0,0] += 0.25
        packed=rewrite_codebooks(self.original,{TEST_NAME:changed})
        new=CB59(packed)
        h=new.handles[TEST_NAME+".weight"]
        self.assertEqual(len(packed),len(self.original))
        self.assertNotEqual(packed,self.original)
        self.assertTrue(np.array_equal(h.packed,a.packed))
        self.assertEqual(h.centroids.shape,(32,8))
        self.assertTrue(np.isfinite(h.centroids).all())
        edited=bytearray(packed)
        edited[len(edited)//2]^=1
        with self.assertRaises(ValueError):
            CB59(bytes(edited))

    def test_native_operator_and_trainable_surrogate_agree(self):
        h=self.decoder.handles[TEST_NAME+".weight"]
        with tempfile.TemporaryDirectory() as td:
            c=Path(__file__).with_name("direct_kernel_v59.c")
            ker=NativeKernel(compile_kernel(c,Path(td)/"test_v84.so"))
            native=PackedLinear(h,ker)
            op=Projection.from_native(TEST_NAME,native)
            x=torch.randn(6,16)
            with torch.inference_mode():
                a=native(x)
                b=op.output(x,op.base_codebook)
            self.assertLess(float((a-b).norm()/(a.norm()+1e-8)),2e-4)

    def test_training_rewrites_native_centroids_not_weights_shape(self):
        h=self.decoder.handles[TEST_NAME+".weight"]
        op=Projection.from_native(TEST_NAME,SimpleNamespace(handle=h,bias=None))
        x=torch.randn(64,16)
        true_centers=op.base_codebook * 1.15
        with torch.no_grad():
            y=op.output(x,true_centers)
        before=local_relative_error(op,x,y,op.base_codebook.numpy())
        variants=train_codebook(op,x,y)
        self.assertEqual(tuple(variants),CHECKPOINTS)
        after=min(local_relative_error(op,x,y,v) for v in variants.values())
        self.assertLess(after,before)
        self.assertTrue(np.array_equal(h.centroids,op.base_codebook.numpy()))

    def test_per_corpus_validation_guard_is_strict(self):
        def entry(w,n):
            return {"mean_KL":(w+n)/2,"per_train_valid_corpus":{
                "wiki_valid":{"real_original_teacher_student_full_vocab_KL":w},
                "news_valid":{"real_original_teacher_student_full_vocab_KL":n}}}
        baseline=entry(10,10)
        variants={"bad":entry(8,11),"good":entry(9.8,9.9)}
        winner,note=choose_guarded_variant(baseline,variants)
        self.assertEqual(winner,"good")
        self.assertTrue(note["newly_rewritten_source_valid_guard_pass"])
        winner,note=choose_guarded_variant(baseline,{"bad":entry(8,11)})
        self.assertEqual(winner,"baseline")
        self.assertFalse(note["newly_rewritten_source_valid_guard_pass"])


if __name__ == "__main__":
    unittest.main()
