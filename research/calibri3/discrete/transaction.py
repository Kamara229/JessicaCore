"""Trial sparse reassignments with exact rollback; selection only on TRAIN."""
import math
import torch

@torch.no_grad()
def trial(source,proposals,loss,clear_all,*,min_gain=1e-7):
    assert proposals and min_gain>=0
    flat=source.codes.reshape(-1)
    positions=[p[0] for p in proposals]
    assert len(set(positions))==len(positions)
    assert all(0<=i<flat.numel() and 0<=new<source.centroids.shape[0] and old!=new for i,old,new,_ in proposals)
    ix=torch.tensor(positions,device=flat.device,dtype=torch.long)
    old=flat[ix].clone();new=torch.tensor([p[2] for p in proposals],device=flat.device,dtype=flat.dtype)
    assert old.tolist()==[p[1] for p in proposals]
    clear_all();before=float(loss());assert math.isfinite(before)
    accepted=False
    try:
        flat[ix]=new;clear_all();after=float(loss())
        accepted=math.isfinite(after) and after+min_gain<before
        return {'before':before,'after_trial':after,'accepted':accepted,'changes':len(proposals) if accepted else 0}
    finally:
        if not accepted:flat[ix]=old
        clear_all()
