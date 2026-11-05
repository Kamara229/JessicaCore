"""Check literal source reconstruction vs original native C on toy model."""
import unittest
import numpy as np
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from direct_codebook_v59 import encode,CB59
from direct_runtime_v59 import NativeKernel,compile_kernel
from v173.recover import exact_dense_matrix
from v173.protocol import check,EXPECTED_BYTE_SIZE,ARMS

class OriginalCodecForensics(unittest.TestCase):
    def test_precommitted_shape_and_sources(self):
        self.assertTrue(check())
        self.assertEqual(ARMS[1],"native_original_CB59")
        self.assertEqual(ARMS[2],"UNPAID_dense_decoder_of_exact_CB59_weights")
        self.assertEqual(EXPECTED_BYTE_SIZE,11229303)

    def test_fp32_recovered_matmul_matches_real_native_C(self):
        torch.manual_seed(173)
        source={"proj.weight":torch.randn(24,16,dtype=torch.float32),
                "proj.bias":torch.randn(24,dtype=torch.float32)}
        blob,manifest,_=encode(source)
        archive=CB59(blob)
        handle=archive.handles["proj.weight"]
        recovered=exact_dense_matrix(handle)
        self.assertEqual(recovered.shape,(24,16))
        self.assertEqual(recovered.dtype,np.dtype(np.float32))
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"libv173_toy.so"
            lib=NativeKernel(compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"),path))
            x=np.random.default_rng(173).normal(size=(3,16)).astype(np.float32)
            native=lib.linear(handle,x)
            decoded=x@recovered.T
            self.assertTrue(np.allclose(native,decoded,atol=2e-5,rtol=1e-5))

    def test_tied_source_demand_embedding_exact(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed=torch.nn.Embedding(32,16)
                self.head=torch.nn.Linear(16,32,bias=False)
                self.head.weight=self.embed.weight
        torch.manual_seed(174)
        toy=Toy()
        blob,_,_=encode(toy.state_dict())
        archive=CB59(blob)
        self.assertIs(archive.handles["embed.weight"],archive.handles["head.weight"])
        h=archive.handles["embed.weight"]
        with TemporaryDirectory() as tmp:
            kernel=NativeKernel(compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"),Path(tmp)/"native.so"))
            ix=np.array([0,2,17,31],dtype=np.int64)
            actual=kernel.embedding(h,ix)
            ref=exact_dense_matrix(h)[ix]
            self.assertTrue(np.allclose(actual,ref,rtol=0,atol=1e-7))

if __name__=="__main__":
    unittest.main()
