"""NRA3 S11: explicitly physically PAID 39-parameter 3way causal router.

Three-way messages and all 43 native CB59 5-bit matrices are frozen and
unchanged. This auxiliary tiny 12->3 FP16 operator is NOT free: complete
bytes (header+39 FP16 scalars+SHA256) are charged explicitly. Forward uses
the physically snapped FP16 source, not an unbilled FP32 optimizer copy.
It is research-only, no EVAL or main/LIVE integration, zero output gate.

The 12 input features are *only causal model-internal evidence*. No UD
gold, weak target, user-specific task label, future token or answers enter
inference. TRAIN labels are an imperfect heuristic, not semantic truth.
"""
from __future__ import annotations

import hashlib
import struct
import math
import torch
from torch import nn
from torch.nn import functional as F

MAGIC=b"N3R11V1\x00"  # 8 bytes
HEADER=struct.Struct("<8sBHH")
D=12
K=3
SOURCE_PARAMETERS=D*K+K
PHYSICAL_BYTES=HEADER.size+SOURCE_PARAMETERS*2+32


def _check(ok:bool,reason:str):
    if not ok:
        raise ValueError("NRA3 S11 paid selector "+reason)


class PaidThreeWayRouter(nn.Module):
    """Fixed 39-parameter FP16 physically charged selector, no extra bias."""

    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(torch.zeros((K,D),dtype=torch.float32))
        self.bias=nn.Parameter(torch.zeros((K,),dtype=torch.float32))
        self.physical_v1=True
        self.output_gate=0
        self.candidate_accepted=False

    def forward(self,x:torch.Tensor)->torch.Tensor:
        _check(isinstance(x,torch.Tensor) and x.shape[-1]==D and
               x.dtype==torch.float32 and torch.isfinite(x).all(),
               "causal evidence ABI nonfinite or not [*,12] FP32")
        if self.output_gate!=0 or self.candidate_accepted:
            raise RuntimeError("S11 never activates original Calibri outputs")
        # Straight-through FP16 cast during TRAIN. On native CPU / reload,
        # source parameters are exact persisted half precision values.
        weight=self.weights+(self.weights.half().float()-self.weights).detach()
        bias=self.bias+(self.bias.half().float()-self.bias).detach()
        return F.linear(x,weight,bias)

    def physical_bytes(self)->bytes:
        if self.output_gate!=0 or self.candidate_accepted:
            raise RuntimeError("S11 activated selector cannot be saved")
        with torch.no_grad():
            values=torch.cat((self.weights.flatten(),self.bias)).detach().cpu()
            _check(bool(torch.isfinite(values).all()),"nonfinite physical source")
            raw=values.half().numpy().astype("<f2",copy=False).tobytes()
        hdr=HEADER.pack(MAGIC,1,D,K)
        payload=hdr+raw
        physical=payload+hashlib.sha256(payload).digest()
        _check(len(physical)==PHYSICAL_BYTES,
               "expected all FP16 header and digest physically charged")
        return physical

    @classmethod
    def from_physical(cls,physical:bytes)->"PaidThreeWayRouter":
        _check(isinstance(physical,bytes) and len(physical)==PHYSICAL_BYTES,
               "missing entire physically paid selector blob")
        _check(HEADER.unpack_from(physical,0)==(MAGIC,1,D,K),
               "wrong ABI/version or 3-route capacity")
        _check(hashlib.sha256(physical[:-32]).digest()==physical[-32:],
               "selector complete byte SHA256 invalid")
        import numpy as np
        raw=physical[HEADER.size:-32]
        a=np.frombuffer(raw,dtype="<f2").astype("float32")
        _check(len(a)==SOURCE_PARAMETERS and bool(np.isfinite(a).all()),
               "FP16 physical fields invalid")
        m=cls().eval()
        with torch.no_grad():
            m.weights.copy_(torch.from_numpy(a[:D*K].reshape(K,D).copy()))
            m.bias.copy_(torch.from_numpy(a[D*K:].copy()))
        for p in m.parameters():p.requires_grad_(False)
        _check(m.physical_bytes()==physical,
               "physical selector cannot be exactly reconstructed")
        return m


def train_balanced_router(x:torch.Tensor,y:torch.Tensor, *,
                          optimizer_steps:int=200,lr:float=.025):
    """TRAIN only, original packed CB59 6M state completely frozen.

    Weak semantic teacher labels are supplied from a separately validated
    TRAIN loader and never embedded in the exported model.
    """
    _check(x.ndim==2 and x.shape[1]==D and
           y.ndim==1 and y.shape==(x.shape[0],) and
           x.shape[0]>=60 and y.dtype==torch.long and
           bool(torch.isfinite(x).all()) and
           set(y.tolist())=={0,1,2} and
           1<=optimizer_steps<=400 and 0<lr<=.1,
           "requires all three classes of registered TRAIN-only real features")
    torch.manual_seed(311111)
    torch.set_num_threads(2)
    router=PaidThreeWayRouter().eval()
    count=torch.bincount(y,minlength=K).float()
    weight=count.sum()/(K*count)
    opt=torch.optim.Adam(router.parameters(),lr=lr)
    with torch.no_grad():
        logits=router(x)
        prior_loss=float(F.cross_entropy(logits,y,weight=weight))
        prior_acc=float((logits.argmax(-1)==y).float().mean())
    trace=[]
    best_state=None
    best_score=prior_loss
    for i in range(optimizer_steps):
        opt.zero_grad(set_to_none=True)
        logits=router(x)
        ce=F.cross_entropy(logits,y,weight=weight)
        penalty=.0005*(router.weights.square().sum()+
                       router.bias.square().sum())
        loss=ce+penalty
        _check(bool(torch.isfinite(loss)), "nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(),1.0)
        opt.step()
        if (i+1)%25==0 or i==optimizer_steps-1:
            with torch.no_grad():
                at=float(F.cross_entropy(router(x),y,weight=weight))
            trace.append({"step":i+1,"balanced_train_ce":at})
            if at<best_score:
                best_score=at
                best_state=router.physical_bytes()
    _check(best_state is not None and best_score<prior_loss,
           "TRAIN-only fit not better than unbiased zero-weight router")
    loaded=PaidThreeWayRouter.from_physical(best_state)
    with torch.inference_mode():
        actual=loaded(x)
        after_loss=float(F.cross_entropy(actual,y,weight=weight))
        after_acc=float((actual.argmax(-1)==y).float().mean())
        routes=torch.bincount(actual.argmax(-1),minlength=K).tolist()
    _check(after_loss<prior_loss and after_loss<=best_score+.001,
           "real physically paid FP16 selector did not survive reload")
    report={
        "router_train_only_before_class_balanced_ce":prior_loss,
        "router_train_only_after_class_balanced_ce":after_loss,
        "router_train_accuracy_before":prior_acc,
        "router_train_accuracy_after":after_acc,
        "router_train_class_counts":count.int().tolist(),
        "router_train_hard_argmax_counts":routes,
        "router_optimizer_steps":optimizer_steps,
        "router_loss_trace":trace,
        "extra_physical_parameters":SOURCE_PARAMETERS,
        "auxiliary_selector_matrix_shape":[K,D],
        "extra_physically_paid_selector_bytes":len(best_state),
        "extra_selector_SHA256":hashlib.sha256(best_state).hexdigest(),
        "router_semantic_accuracy_is_NOT_proven":True,
        "full_model_parent_NRA_all_inclusive_bpp_NOT_certified":True,
        "candidate_accepted":False,
        "output_gate":0,
    }
    return best_state,report
