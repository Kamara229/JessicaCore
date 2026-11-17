"""Oscar v1K genuine differentiable original K32V8 geometry / GT contracts.

Synthetic matrices only; actual source binary CRC, full 5bit indices,
teacher and independent pretrained model validation happens in real CI.
"""
import unittest
import numpy as np
import torch

from direct_codebook_v59 import QuantHandle,pack5
from oscar.v1k.joint_original_codebooks import (
    ExactOriginalK32V8Matrix,TARGETS,
)
from oscar.v1f.green_tunnel import FullLayerGreenTunnel
from oscar.v1k.real_joint_source_gt8 import source_gt_anchor


class OriginalPhysicallyPaidSourceOnlyTests(unittest.TestCase):
    def test_exact_quantized_original_no_extra_fp32_source_tensor(self):
        rng=np.random.default_rng(818)
        codes=rng.integers(0,32,size=8*64//8,dtype=np.uint8)
        scale=np.linspace(.13,.27,8,dtype=np.float16)
        centers=rng.standard_normal((32,8)).astype(np.float16)
        handle=QuantHandle("layers.1.feed_forward.w2.weight",
                           (8,64),
                           np.frombuffer(pack5(codes),dtype=np.uint8),
                           scale.astype(np.float32),
                           centers.astype(np.float32))
        module=ExactOriginalK32V8Matrix(handle)
        expected=torch.from_numpy(
            centers.astype(np.float32)[codes].reshape(8,64)*
            scale.astype(np.float32)[:,None])
        self.assertTrue(torch.equal(module.weight(),expected))
        self.assertEqual(module.fixed_original_5bit_indices.numel(),64)
        self.assertEqual(len(module.state_dict()),5)
        self.assertEqual(
            module.changed_fp16()["centroid_fp16_changed"],0)

    def test_gradients_reach_original_centroids_and_scales(self):
        rng=np.random.default_rng(423)
        codes=rng.integers(0,32,size=8*64//8,dtype=np.uint8)
        centers=rng.standard_normal((32,8)).astype(np.float16)
        h=QuantHandle("matrix",(8,64),
            np.frombuffer(pack5(codes),dtype=np.uint8),
            np.full(8,.25,dtype=np.float32),
            centers.astype(np.float32))
        mod=ExactOriginalK32V8Matrix(h)
        loss=mod.weight().square().mean()+mod.penalty()
        loss.backward()
        self.assertGreater(float(mod.center.grad.norm()),0)
        self.assertGreater(float(mod.log_scale.grad.norm()),0)
        with torch.no_grad():
            # Normalized synthetic update large enough to survive actual
            # FP16 physical rounding, regardless of random gradient scale.
            mod.center.sub_(.12*mod.center.grad/
                            mod.center.grad.norm().clamp_min(1e-12))
            mod.log_scale.sub_(.12*mod.log_scale.grad/
                               mod.log_scale.grad.norm().clamp_min(1e-12))
        self.assertGreater(mod.changed_fp16()["centroid_fp16_changed"],0)
        self.assertGreater(mod.changed_fp16()["row_scale_fp16_changed"],0)
        self.assertEqual(mod.changed_fp16()["5bit_codeword_indices_changed"],0)

    def test_joint_GT_anchor_has_nonzero_autograd(self):
        gt=FullLayerGreenTunnel()
        anchors={k:v.clone().float() for k,v
                 in gt.source_paid_tensors().items()}
        with torch.no_grad():
            gt.up[0,0,0]=.1
            gt.down[1,0,0]+=.2
        val=source_gt_anchor(gt,anchors)
        self.assertGreater(float(val.detach()),0)
        val.backward()
        self.assertGreater(float(gt.up.grad.norm()),0)
        self.assertGreater(float(gt.down.grad.norm()),0)

    def test_selected_sensitive_matrices_are_predeclared(self):
        self.assertEqual(TARGETS,(
          "layers.7.feed_forward.w2.weight",
          "layers.1.feed_forward.w2.weight",
          "layers.6.attention.wq.weight",
          "layers.1.feed_forward.w3.weight",
        ))


if __name__=="__main__":
    unittest.main()
