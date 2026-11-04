"""Integrity tests for full physical 135M all-in bpp; synthetic bytes are NOT pretrained."""
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from v117.protocol import (SOURCE_BYTES,MAX_SOURCE_BYTES,TOTAL_UNIQUE,
                           declared_factor_bytes,roles)
from v117.archive import Archive,encode
from v117.runtime import PaidResidual

class Guard(unittest.TestCase):
    def test_full_source_and_budget(self):
        synthetic=bytes(SOURCE_BYTES)
        fake_sha=hashlib.sha256(synthetic).hexdigest()
        with patch("v117.archive.ORIGINAL_CB59_SHA",fake_sha):
            raw=encode(synthetic)
            candidate=Archive(raw)
            info=candidate.stats()
            self.assertEqual(info["native_CB59_base_bytes"],SOURCE_BYTES)
            self.assertEqual(info["physical_all_roles"],60)
            self.assertEqual(info["physical_factor_bytes"],declared_factor_bytes())
            self.assertLessEqual(info["complete_source_bytes"],MAX_SOURCE_BYTES)
            self.assertLess(info["all_in_bpp"],.7)
            self.assertEqual(candidate.base,synthetic)
            self.assertEqual(len(set(roles())),60)
            tampered=bytearray(raw);tampered[-40]^=1
            with self.assertRaises(ValueError):
                Archive(tampered)
            with self.assertRaises(ValueError):
                encode(synthetic+b"\0")
    def test_forward_nonzero_and_identity(self):
        class Base(nn.Module):
            def __init__(self):
                super().__init__();self.out_features=4;self.in_features=3
            def forward(self,x):return 2*x[:,0:1].expand(-1,4)
        baseline=Base()
        u=np.zeros((4,2),dtype=np.float16)
        v=np.zeros((2,3),dtype=np.float16)
        x=torch.tensor([[1.,2.,3.]],dtype=torch.float32)
        zero=PaidResidual(baseline,u,v)
        self.assertTrue(torch.equal(zero(x),baseline(x)))
        u[1,0]=3;v[0,2]=2
        corrected=PaidResidual(baseline,u,v)
        expected=baseline(x)
        expected[:,1]+=18
        torch.testing.assert_close(corrected(x),expected)
        self.assertTrue(torch.isfinite(corrected(x)).all())

if __name__=="__main__":
    unittest.main()
