"""Real numerical VJP must match explicit dense W on tiny synthetic TEST only."""
import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from direct_codebook_v59 import pack5
from v119.native import build,PackedVJP
class VJPGuard(unittest.TestCase):
    def test_packed_backward_full_vs_dense_toy(self):
        rng=np.random.default_rng(119)
        rows,cols=8,16
        codes=rng.integers(0,32,(rows,cols//8),dtype=np.uint8)
        centers=rng.normal(size=(32,8)).astype(np.float32)
        scales=rng.uniform(.1,2,size=rows).astype(np.float32)
        h=SimpleNamespace(packed=np.frombuffer(pack5(codes),dtype=np.uint8),
                          row_scales=scales,centroids=centers,shape=(rows,cols))
        W=centers[codes].reshape(rows,cols)*scales[:,None]
        with tempfile.TemporaryDirectory() as folder:
            forward,backward=build(folder)
            x=torch.randn(3,cols,dtype=torch.float32,requires_grad=True)
            y=PackedVJP.apply(x,h,forward,backward)
            torch.testing.assert_close(y,torch.from_numpy(x.detach().numpy()@W.T),atol=1e-4,rtol=1e-4)
            dy=torch.randn_like(y)
            y.backward(dy)
            expected=torch.from_numpy(dy.numpy()@W)
            torch.testing.assert_close(x.grad,expected,atol=1e-4,rtol=1e-4)
if __name__=="__main__":unittest.main()
