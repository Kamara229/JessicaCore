"""Local policy unit test; synthetic sources, not a real model/GPU benchmark."""
import torch
from torch import nn
from torch.nn import functional as F
from .cache import frozen_cache

class Source(nn.Module):
    def __init__(self,trainable=False):
        super().__init__();self.centroids=nn.Parameter(torch.randn(8,8),requires_grad=trainable)
        self.log_scales=nn.Parameter(torch.zeros(4),requires_grad=trainable)
        self.register_buffer('codes',torch.randint(8,(8,)))
        self.register_buffer('base_scales',torch.ones(4))
        self.cached=None
    def forward(self):
        if self.cached is None:self.cached=self.centroids[self.codes].reshape(4,16)*(self.base_scales*self.log_scales.exp())[:,None]
        return self.cached
    def clear(self):self.cached=None

def run():
    torch.manual_seed(22);a,b,t=Source(),Source(),Source(True)
    x=torch.randn(3,16,requires_grad=True)
    old=Source.forward
    ref=F.linear(x,a());g=torch.autograd.grad(ref.square().sum(),x)[0]
    with frozen_cache(Source,[a,b,t],byte_cap=256) as stats:
        y=F.linear(x,a());a.clear();assert torch.equal(y,F.linear(x,a()))
        assert torch.equal(g,torch.autograd.grad(y.square().sum(),x)[0])
        assert stats['hits']==1 and stats['misses']==1
        old_matrix=a().clone()
        with torch.no_grad():a.centroids.add_(1)
        z=a();assert torch.allclose(z,old_matrix+1) and stats['invalidations']==1
        with torch.no_grad():a.codes[0]=(a.codes[0]+1)%8
        assert not torch.equal(a(),z) and stats['invalidations']==2
        b();assert stats['evictions']==1 and stats['resident_bytes']<=256
        for _ in range(2):
            loss=F.linear(x,t()).sum();gt=torch.autograd.grad(loss,t.centroids)[0];assert gt.abs().sum()>0;t.clear()
        assert stats['bypasses']==2
        a();a.centroids.requires_grad_(True)
        assert torch.autograd.grad(a().sum(),a.centroids)[0].abs().sum()>0
        a.clear();a.centroids.requires_grad_(False)
    assert Source.forward is old and stats['resident_bytes']==0
    assert all(s.cached is None for s in (a,b,t))
    try:
        with frozen_cache(Source,[a],byte_cap=0):raise RuntimeError('scope cleanup')
    except RuntimeError:pass
    assert Source.forward is old and a.cached is None
    print('PASS',stats)

if __name__=='__main__':run()
