"""Synthetic test, not real CALIBRI quality or speed certification."""
import torch
from .proposals import propose
from .transaction import trial

class Source:
    pass

def run():
    torch.manual_seed(73);o,i,k=7,24,8
    s=Source();s.centroids=torch.randn(k,8);s.codes=torch.randint(k,(o*i//8,));scales=torch.rand(o)+.1
    x=torch.randn(2,5,i);dy=torch.randn(2,5,o)
    def loss():return (torch.nn.functional.linear(x,s.centroids[s.codes].reshape(o,i)*scales[:,None])*dy).sum()
    old=s.codes.clone();before=float(loss());p=propose(x,dy,s.centroids,scales,s.codes,(o,i),budget=5,rows=2)
    assert len(p)==5 and len({v[0] for v in p})==5
    result=trial(s,p,loss,lambda:None);assert result['accepted']
    assert abs((before-float(loss()))-sum(v[3] for v in p))<2e-4
    assert int((s.codes!=old).sum())==5
    current=s.codes.clone();reverse=[(i,new,old,gain) for i,old,new,gain in p]
    result=trial(s,reverse,loss,lambda:None);assert not result['accepted'] and torch.equal(s.codes,current)
    calls=[0]
    def fail():
        calls[0]+=1
        if calls[0]>1:raise ValueError('injected failure')
        return loss()
    try:trial(s,reverse,fail,lambda:None)
    except ValueError:pass
    else:raise AssertionError('expected failure')
    assert torch.equal(s.codes,current)
    print('PASS: compact score matches exact linear objective; budget, rejection and exception rollback')

if __name__=='__main__':run()
