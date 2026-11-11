"""S132 genuine additive rank growth preserving immutable S129 tensors.

Rank448 original down is reconstructed with the exact historical S129
seed and tensor shape; further 448-channel down chunks have separate
deterministic seeds. No preexisting learned coefficient changes.
New up columns start at exact zero. All nonzero new coefficients MUST be
serialized FP16 with explicit active index list. A zero-derived/fixed
parameter counts toward structural architecture, but is NOT reported
as an independently learned weight or novel competence.

Only active selected channels consume inference FLOPs. At each stage
the new growth chunk enables 24 zero-initialized channels. All existing
active indices and learned physical weights are carried forward.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from calibri2.s130.route_protected_expansion import RouteIsolatedAdditiveCore
from calibri2.s129.additive_core_expansion import (
    WIDTH,DEPTH,EXPANSION_RANK,ACTIVE_RANK,SEED,UP_KEY)
from calibri2.s132.growth_schedule import (
    START_RANK,RANK_STEP,MAX_RANK,PER_RANK,architecture_params)

PREFIX="s132_progressive."
IDS_KEY=PREFIX+"active_indices_i32"
WEIGHTS_KEY=PREFIX+"active_up_fp16"


def fixed_down_chunks(rank:int)->torch.Tensor:
    if rank<START_RANK or rank>MAX_RANK:
        raise ValueError("S132 invalid growth rank")
    remaining=rank
    chunks=[]
    index=0
    while remaining>0:
        width=min(RANK_STEP,remaining)
        g=torch.Generator(device="cpu").manual_seed(SEED+index)
        chunks.append(torch.randn((DEPTH,width,WIDTH),generator=g)*0.02)
        remaining-=width
        index+=1
    return torch.cat(chunks,dim=1).contiguous()


def initial_active_indices(rank:int)->tuple[int,...]:
    if rank<START_RANK:
        raise ValueError("Expected S129 448 base")
    x=list(range(ACTIVE_RANK))
    # Only grow activation when a *new structural chunk* is added.
    for start in range(START_RANK,rank,RANK_STEP):
        width=min(RANK_STEP,rank-start)
        x.extend(range(start,start+min(ACTIVE_RANK,width)))
    return tuple(x)


class ProgressiveRouteCore(RouteIsolatedAdditiveCore):
    def __init__(self,*,rank:int,active_indices:tuple[int,...]|None=None):
        super().__init__(active_rank=ACTIVE_RANK)
        if rank<START_RANK or rank>MAX_RANK:
            raise ValueError("S132 rank outside original-to-target 135M interval")
        self.rank=int(rank)
        self.down=nn.Parameter(fixed_down_chunks(rank),requires_grad=False)
        self.up=nn.Parameter(torch.zeros((DEPTH,WIDTH,rank)))
        if active_indices is None:
            active_indices=initial_active_indices(rank)
        if (len(active_indices)!=len(set(active_indices))
                or not active_indices
                or any(not isinstance(x,int) or x<0 or x>=rank for x in active_indices)
                or list(active_indices)!=sorted(active_indices)):
            raise ValueError("S132 active columns must be unique, ordered, in range")
        self.active_indices=tuple(active_indices)

    def _layer_hook(self,index:int):
        def hook(_layer,_inputs,output):
            if not self._active:
                return output
            h=output[0] if isinstance(output,tuple) else output
            if not isinstance(h,torch.Tensor) or h.ndim!=3 or h.shape[-1]!=WIDTH:
                raise AssertionError("S132 original 320 hidden ABI changed")
            if h.device!=self.up.device or h.device!=self.down.device:
                raise AssertionError("S132 core and new adapter device changed")
            if len(self._visited)!=index:
                raise AssertionError("S132 decoder hook order changed")
            self._visited.append(index)
            bridge=getattr(self,"_bridge_for_route_only",None)
            if bridge is not None and getattr(bridge,"_active_profile",None)!="S93":
                return output
            idx=torch.as_tensor(self.active_indices,device=h.device,dtype=torch.long)
            down=self.down[index].index_select(0,idx)
            up=self.up[index].index_select(1,idx)
            z=F.gelu(F.linear(F.layer_norm(h,(WIDTH,)),down))
            updated=h+F.linear(z,up)
            return (updated,*output[1:]) if isinstance(output,tuple) else updated
        return hook

    def architectural_new_parameter_count(self):
        count=sum(int(x.numel()) for x in self.parameters())
        if count!=PER_RANK*self.rank:
            raise AssertionError("S132 new counted architecture inconsistent")
        return count

    def trainable_initial_parameters(self):
        if self.down.requires_grad or not self.up.requires_grad:
            raise AssertionError("S132 only new up receives gradients")
        self.assert_inactive_zero()
        return [self.up]

    def assert_inactive_zero(self):
        with torch.no_grad():
            active=torch.zeros((self.rank,),dtype=torch.bool,device=self.up.device)
            active[list(self.active_indices)]=True
            if torch.count_nonzero(self.up[:,:,~active]).item():
                raise AssertionError("S132 omitted learned nonzero weights from sidecar")
            if not bool(torch.isfinite(self.up).all()):
                raise AssertionError("S132 new weights nonfinite")

    def export_active_fp16(self):
        self.assert_inactive_zero()
        return {
            IDS_KEY:torch.tensor(self.active_indices,dtype=torch.int32),
            WEIGHTS_KEY:self.up.detach().cpu()[:,:,list(self.active_indices)].half().contiguous(),
        }

    def load_active_fp16(self,sidecar):
        if set(sidecar)!={IDS_KEY,WEIGHTS_KEY}:
            raise ValueError("S132 incomplete physical learned-weight sidecar")
        ids=sidecar[IDS_KEY]
        weights=sidecar[WEIGHTS_KEY]
        if (ids.dtype!=torch.int32
            or ids.ndim!=1
            or tuple(int(i) for i in ids.tolist())!=self.active_indices
            or weights.dtype!=torch.float16
            or weights.shape!=(DEPTH,WIDTH,len(self.active_indices))
            or not bool(torch.isfinite(weights).all())):
            raise ValueError("S132 active learned-weight payload inconsistent")
        # Apple MPS lacks aten::index_copy.out in the pinned PyTorch build.
        # Assemble the physical FP16 sidecar on CPU once, then perform a
        # single GPU copy; all training and gradients still run on MPS.
        with torch.no_grad():
            dense=torch.zeros(self.up.shape,dtype=self.up.dtype,device="cpu")
            ids=torch.as_tensor(self.active_indices,dtype=torch.long,device="cpu")
            dense.index_copy_(2,ids,weights.to(device="cpu",dtype=dense.dtype))
            self.up.copy_(dense.to(device=self.up.device))
        self.assert_inactive_zero()


def expand_preserving_previous(old:ProgressiveRouteCore,new_rank:int)->ProgressiveRouteCore:
    if not (old.rank<new_rank<=min(old.rank+RANK_STEP,MAX_RANK)):
        raise ValueError("S132 rank growth must be one bounded new chunk")
    if not old.active_indices==initial_active_indices(old.rank):
        raise AssertionError("S132 old active lineage differs")
    old.assert_inactive_zero()
    new=ProgressiveRouteCore(rank=new_rank)
    with torch.no_grad():
        if not torch.equal(new.down[:,:old.rank,:].cpu(),old.down.detach().cpu()):
            raise AssertionError("S132 new expansion changed inherited fixed down matrices")
        new.up[:,:,:old.rank].copy_(old.up.detach().cpu())
    new.assert_inactive_zero()
    if not torch.equal(new.up[:,:,:old.rank],old.up.detach().cpu()):
        raise AssertionError("S132 growth modified old learned weights")
    return new
