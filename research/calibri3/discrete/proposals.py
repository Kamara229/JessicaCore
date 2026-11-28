"""Compact first-order code-index proposals; no dense weight or weight gradient."""
import torch

@torch.no_grad()
def propose(x,dy,centers,scales,codes,shape,budget=16,rows=16):
    o,i=shape;g=i//8;k=centers.shape[0]
    assert i%8==0 and centers.shape==(k,8) and codes.numel()==o*g
    assert x.shape[:-1]==dy.shape[:-1] and x.shape[-1]==i and dy.shape[-1]==o
    assert scales.shape==(o,) and budget>0 and rows>0
    assert codes.min()>=0 and codes.max()<k
    x=x.detach().float().reshape(-1,g,8);dy=dy.detach().float().reshape(-1,o)
    table=x@centers.detach().float().T
    assert torch.isfinite(table).all() and torch.isfinite(dy).all()
    ids=codes.reshape(o,g);best=[]
    for start in range(0,o,rows):
        end=min(o,start+rows)
        # dL/dy times scaled codebook dot products, summed over actual tokens.
        costs=torch.einsum('tr,tgk->rgk',dy[:,start:end]*scales[start:end],table)
        old=costs.gather(-1,ids[start:end,:,None]).squeeze(-1)
        value,choice=costs.min(-1);gain=(old-value).reshape(-1)
        ix=torch.argsort(gain,descending=True,stable=True)[:budget]
        for j in ix.tolist():
            if float(gain[j])>0:
                best.append((start*g+j,int(ids.reshape(-1)[start*g+j]),int(choice.reshape(-1)[j]),float(gain[j])))
        best=sorted(best,key=lambda v:(-v[3],v[0]))[:budget]
    return best
