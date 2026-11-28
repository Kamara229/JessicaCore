"""Synthetic unit tests only; not CALIBRI full model or MPS measurements."""
import torch
from torch import nn
from .schedule import plan,metrics
from .target_head import last_position_readout

def run():
    rows=[{"p":[1,3,4],"g":[6,2]}, {"p":[1,5],"g":[7,8,2]}, {"p":[1,9,4],"g":[9,2]}]
    indices=[0,1,2,0]
    for b in (1,2,4,16):
        groups=plan(rows,indices,b)
        actual=sorted((occ,i,pos) for _,xs in groups for occ,i,pos in xs)
        expected=[(occ,i,pos) for occ,i in enumerate(indices) for pos in range(len(rows[i]["g"]))]
        assert actual==expected
        assert all(len(xs)<=b and all(len(rows[i]["p"])+pos==length for _,i,pos in xs) for length,xs in groups)
        print("SCHEDULE",b,metrics(rows,indices,b))
    try:plan([dict(rows[0],split="DEV")],[0],4)
    except ValueError:pass
    else:raise AssertionError("DEV leak")
    torch.manual_seed(7)
    model=nn.Module();model.lm_head=nn.Linear(12,31)
    h=torch.randn(4,9,12,requires_grad=True)
    params=(h,*model.lm_head.parameters())
    a=model.lm_head(h)[:,-1:,:]
    ga=torch.autograd.grad(a.square().mean(),params)
    with last_position_readout(model) as counts:
        b=model.lm_head(h)
        gb=torch.autograd.grad(b.square().mean(),params)
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
    assert all(torch.allclose(x,y,atol=1e-6,rtol=1e-5) for x,y in zip(ga,gb))
    assert counts=={"calls":1,"input_positions":36,"projected_positions":4}
    assert model.lm_head(h).shape==(4,9,31)
    try:
        with last_position_readout(model):raise RuntimeError("cleanup")
    except RuntimeError:pass
    assert model.lm_head(h).shape==(4,9,31)
    print("HEAD_PASS",counts)
if __name__=="__main__":run()
