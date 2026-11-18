"""S58 TRAIN-only event and transfer residual over FROZEN S57 readout.

83 additional FP16 parameters: 41 event residual + 41 transfer
residual + 1 transfer bias. Use ALREADY INSPECTED S53-S56 native96
old TRAIN and gold only. Freeze original 43M+GT, old S26, S57 paid231.
No new S58 raw/gold32 or S57 spent24 DEV data allowed in fitting.
Full-context offline research, NOT online/causal or accepted.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    EVENT_DIM, EventFirstInputs, S57EventFirstRecipient,
)
N_PAID=2*EVENT_DIM+1
assert EVENT_DIM==41 and N_PAID==83

def require(ok,msg):
    if not bool(ok): raise ValueError("S58 old96 TRAIN residual: "+msg)


class S58EventTransferResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.event_delta=nn.Parameter(torch.zeros(EVENT_DIM))
        self.transfer_delta=nn.Parameter(torch.zeros(EVENT_DIM))
        self.transfer_bias=nn.Parameter(torch.zeros(1))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.online_causality_certified=False

    def forward(self,data:EventFirstInputs,frozen_s57):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and not self.online_causality_certified,
                "cannot activate research residual")
        require(data.event.shape[-1]==EVENT_DIM and
                not data.event.requires_grad, "original frozen features drift")
        e0,t0,r0,n0=frozen_s57
        b,w,_=data.event.shape
        require(e0.shape==t0.shape==(b,w)
                and r0.shape==n0.shape==(b,w,w)
                and all(not q.requires_grad for q in frozen_s57),
                "old paid 231 must be frozen")
        return (
            e0+data.event@self.event_delta,
            t0+data.event@self.transfer_delta+self.transfer_bias,
            r0,n0,
        )

    def decode(self,data:EventFirstInputs,frozen_s57):
        with torch.inference_mode():
            event,transfer,recipient,rej=self(data,frozen_s57)
            event=event.masked_fill(~data.candidate_event,float("-inf"))
            chosen=event.argmax(-1)
            b,w=chosen.shape[0],data.candidate_event.shape[1]
            result=torch.zeros((b,w,w),dtype=torch.bool)
            rejected=torch.zeros_like(result)
            stats={"nontransfer_abstentions":0,"unsafe_target_abstentions":0}
            for i in range(b):
                src=int(chosen[i])
                if float(transfer[i,src])<=0:
                    stats["nontransfer_abstentions"]+=1
                    continue
                safe=data.valid_pair[i,src]
                if not bool(safe.any()):
                    stats["unsafe_target_abstentions"]+=1
                    continue
                dest=int(recipient[i,src].masked_fill(~safe,float("-inf")).argmax())
                result[i,src,dest]=True
                rejected[i,src,dest]=float(rej[i,src,dest])>0
            require(int(result.sum())<=b
                    and not bool((result&~data.valid_pair).any())
                    and not bool((rejected&~result).any()),
                    "new S58 residual emitted unsafe directed pair")
            return {"chosen_event":chosen,
                    "directed_recipient":result,
                    "directed_rejected":rejected,
                    "reason_counts":stats,
                    "output_gate":0,
                    "candidate_accepted":False,
                    "activation_allowed":False}

    def fp16(self):
        with torch.no_grad():
            v=torch.cat((self.event_delta.detach(),
                         self.transfer_delta.detach(),
                         self.transfer_bias.detach())).to(torch.float16)
            require(v.shape==(N_PAID,) and bool(torch.isfinite(v).all()),
                    "invalid 83 FP16 frozen residual")
            return v.contiguous()

    @classmethod
    def from_fp16(cls,p):
        require(p.shape==(N_PAID,) and p.dtype==torch.float16
                and bool(torch.isfinite(p).all()) and not p.requires_grad,
                "saved S58 83FP16 ABI mismatch")
        out=cls()
        with torch.no_grad():
            out.event_delta.copy_(p[:EVENT_DIM].float())
            out.transfer_delta.copy_(p[EVENT_DIM:2*EVENT_DIM].float())
            out.transfer_bias.copy_(p[-1:].float())
        out.eval().requires_grad_(False)
        require(torch.equal(out.fp16(),p),"FP16 physical reload drift")
        return out


def train_on_old_S53_S54_S55_S56_ONLY(
    features:list[tuple[EventFirstInputs,torch.Tensor,torch.Tensor]],
    old:S57EventFirstRecipient
):
    """Gold main predicate + exists(recipient) ONLY old96 supervision."""
    require(len(features)==4 and old.output_gate==0
            and all(not p.requires_grad for p in old.parameters()),
            "TRAIN provenance: old four 24-case native sets and frozen S57")
    require(sum(int(t.sum()) for _,_,t in features)==60,
            "exact spent old96 directed recipient totals drift")
    for inp,idx,has in features:
        b,w,_=inp.event.shape
        require(b==24 and idx.shape==(b,) and has.shape==(b,)
                and has.dtype==torch.bool
                and bool(inp.candidate_event[torch.arange(b),idx].all())
                and not inp.event.requires_grad,
                "old96 event or transfer supervised ABI failed")
    torch.manual_seed(581118)
    weights=S58EventTransferResidual()
    optimizer=torch.optim.Adam(weights.parameters(),lr=.022)
    with torch.no_grad():
        cached=[tuple(x.detach().clone() for x in old(inp))
                for inp,_,_ in features]
    for step in range(320):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for (inp,idx,has),base in zip(features,cached):
            ev,tr,_,_=weights(inp,base)
            events=ev.masked_fill(~inp.candidate_event,-1e9)
            i=torch.arange(idx.shape[0])
            # All hyperparameters fixed without any S57 DEV inspection.
            loss=F.cross_entropy(events,idx) + 1.2*F.binary_cross_entropy_with_logits(
                tr[i,idx],has.float(),pos_weight=torch.tensor(.60))
            losses.append(loss)
        total=sum(losses)/4 + .002*sum(p.square().sum() for p in weights.parameters())
        require(bool(torch.isfinite(total)), "loss not finite")
        total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in weights.parameters()),"gradient missing")
        optimizer.step()
    out=S58EventTransferResidual.from_fp16(weights.fp16())
    return out,out.fp16(),{
        "source":"S53_S54_S55_S56_ALREADY_SPENT_REAL43M_GT_OLD96_TRAIN",
        "training_rows":96,"positive_recipient_rows":60,
        "parameters":83,"training_steps":320,"lr":.022,"L2":.002,
        "new_32_HELD_source_or_gold_used":False,
        "S57_spent24_DEV_used_in_fitting":False,
        "original_model_and_old231_trainable":False,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False
    }
