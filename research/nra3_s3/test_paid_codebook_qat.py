"""S3 synthetic-only gradient/repack/C-native tests, NOT language evaluation."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config,NRA3NeuralArea
from research.nra3_s1.physical_cb59 import encode_area
from research.nra3_s2.native_cpu import NativeCB59Kernel
from research.nra3_s3.paid_codebook_qat import (
    CB59PaidCodebook, SyntheticTrainingGuard,train_one_synthetic_codebook,
    repack_paid_single_operator,train_sha_for,
)


class NRA3S3PaidQATTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(303003)
        torch.set_num_threads(1)
        cls.config=NRA3Config(hidden=32,heads=4,grammar_blocks=1,
                              semantic_blocks=1,relation_width=24,relation_steps=2)
        cls.model=NRA3NeuralArea(cls.config).eval()
        cls.archive,_,_=encode_area(cls.model)
        cls.name="grammar.0.ffn.0.weight"
        cls.label="SYNTHETIC_NRA3_S3_TRAIN:FFN_CODEBOOK_TINY"
        cls.allow=frozenset({train_sha_for(cls.label)})
        cls.guard=SyntheticTrainingGuard(cls.allow,frozenset())

    def test_provenance_guard_blocks_unapproved_and_denied_every_step(self):
        self.guard.admit_optimizer_step((self.label,))
        with self.assertRaises(ValueError):
            self.guard.admit_optimizer_step(("SYNTHETIC_NRA3_S3_TRAIN:UNKNOWN",))
        with self.assertRaises(ValueError):
            self.guard.admit_optimizer_step(("actual Russian QA text",))
        with self.assertRaises(ValueError):
            self.guard.admit_optimizer_step((self.label,self.label))
        with self.assertRaises(ValueError):
            SyntheticTrainingGuard(self.allow,self.allow)
        denied="SYNTHETIC_NRA3_S3_TRAIN:REJECTED"
        strict=SyntheticTrainingGuard(self.allow,frozenset({train_sha_for(denied)}))
        with self.assertRaises(ValueError):
            strict.admit_optimizer_step((denied,))

    def test_true_paid_qat_gradients_and_physical_repack(self):
        x=torch.randn(64,self.model.state_dict()[self.name].shape[1])
        # Teacher exists ONLY in the synthetic TRAIN process; never saved as
        # inference weights or used as a claimed semantic benchmark.
        teacher=self.model.state_dict()[self.name].detach().clone()
        y=F.linear(x,teacher)
        saved,report=train_one_synthetic_codebook(
            self.archive,self.name,x,y,self.guard,self.label,steps=70,lr=.04)
        self.assertEqual(report.optimizer_steps,70)
        self.assertEqual(report.actual_source_bytes_before,len(self.archive))
        self.assertEqual(report.actual_source_bytes_after,len(self.archive))
        self.assertNotEqual(report.source_sha_after,report.source_sha_before)
        self.assertLess(report.after_train_mse,report.before_train_mse*.98)
        self.assertAlmostEqual(report.after_physical_reload_mse,
                               report.after_train_mse,delta=2e-5)
        self.assertFalse(report.candidate_accepted)
        self.assertEqual(report.output_gate,0)
        before=CB59(self.archive);after=CB59(saved)
        self.assertEqual(before.meta,after.meta)
        self.assertEqual(set(before.handles),set(after.handles))
        for key in before.handles:
            self.assertEqual(before.handles[key].packed.tobytes(),
                             after.handles[key].packed.tobytes())
            if key!=self.name:
                self.assertTrue(np.array_equal(before.handles[key].centroids,
                                               after.handles[key].centroids))
                self.assertTrue(np.array_equal(before.handles[key].row_scales,
                                               after.handles[key].row_scales))
        self.assertEqual(set(before.raw),set(after.raw))
        self.assertTrue(all(np.array_equal(before.raw[k],after.raw[k])
                            for k in before.raw))
        with tempfile.TemporaryDirectory() as td:
            library=Path(td)/"libcalibri_cb59.so"
            subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                            "research/calibri/direct_kernel_v59.c",
                            "-o",str(library)],check=True)
            kernel=NativeCB59Kernel(library)
            actual=kernel.matmul(after.handles[self.name],x)
            reference=CB59PaidCodebook(after,self.name)(x).detach()
            self.assertTrue(torch.allclose(actual,reference,atol=3e-5,rtol=2e-4))
            self.assertLess(float(F.mse_loss(actual,y)),
                            report.before_train_mse*.98)

    def test_no_repack_without_exact_immutable_code_indices(self):
        source=CB59(self.archive)
        m=CB59PaidCodebook(source,self.name)
        codes=m.original_packed_codes
        m.original_packed_codes=codes[:-1]+bytes([codes[-1]^1])
        with self.assertRaises(ValueError):
            repack_paid_single_operator(self.archive,self.name,m)
        with self.assertRaises(ValueError):
            train_one_synthetic_codebook(
                self.archive,self.name,
                torch.randn(2,32),torch.randn(2,128),
                SyntheticTrainingGuard(
                    frozenset({train_sha_for("SYNTHETIC_NRA3_S3_TRAIN:OTHER")}),
                    frozenset()),self.label,steps=1)
        damaged=bytearray(self.archive)
        damaged[len(damaged)//2]^=1
        with self.assertRaises(ValueError):
            repack_paid_single_operator(bytes(damaged),self.name,
                                        CB59PaidCodebook(source,self.name))


if __name__=="__main__":
    unittest.main()
