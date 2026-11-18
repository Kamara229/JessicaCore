"""S59 research-only ORIGINAL Calibri43M+GreenTunnel semantic word heads.

S57/S58 only exposed 41 surface + logits event features; true original
post-layer8 native320 survived primarily as pair cosine. S59 exposes the
unmodified native320 itself (normalized + sentence centered) to three
independent new supervised small heads: outer event, actual transfer/not,
and selected-event-local REJECTED polarity. This is a targeted architecture
change, NOT a proof of general Russian understanding or online causality.

Only genuine original CPU post-GT frozen word339 + old S26+S57 logits
enter forward; manual source/recipient/polarity labels enter TRAIN loss only.
S59 produces <=1 directed pair per sentence and never modifies original
Calibri43M/GreenTunnel or old S26/S57/S58 weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    EVENT_DIM, EventFirstInputs, S57EventFirstRecipient,
)

D=2*320+EVENT_DIM
P=3*(D+1)  # 2,046 FP16
assert EVENT_DIM==41 and D==681 and P==2046

def require(cond,why):
    if not bool(cond): raise ValueError("S59 true43M GT semantic research: "+why)

@dataclass(frozen=True)
class SemanticInputs:
    event:EventFirstInputs
    semantic:torch.Tensor        # B,W,681 source-only frozen semantic vectors

def semantic_features(native339:torch.Tensor,inp:EventFirstInputs)->SemanticInputs:
    require(native339.ndim==3 and native339.shape[-1]==339 and
            not native339.requires_grad and
            native339.dtype==torch.float32 and
            inp.event.shape==native339.shape[:2]+(EVENT_DIM,) and
            inp.candidate_event.dtype==torch.bool and
            inp.valid_pair.dtype==torch.bool,
            "original CPU source ABI or trainable native hidden")
    b,w,_=native339.shape
    # Eligible words inside source are encoded in the original43M native
    # word339 features. Pure source-only normalization, NEVER gold tokens.
    mask=native339[...,338:339].new_zeros(b,w,1)
    # Native validity MUST be provided by frozen parent EventFirstInputs.
    # Padding is excluded via candidate or safe-pair participant mask,
    # including any quoted words; this is NOT manual event supervision.
    valid=(inp.candidate_event |
           inp.valid_pair.any(dim=-1) |
           inp.valid_pair.any(dim=-2)).unsqueeze(-1)
    require(bool(valid.any(-2).all()),"no original model word rows")
    mask=valid.float()
    raw=native339[...,:320].detach()
    avg=(raw*mask).sum(-2,keepdim=True)/mask.sum(-2,keepdim=True).clamp(min=1)
    # Native 43M values are frozen, scale normalization for robust FP16
    # linear adapter; no oracle or manual POS is included.
    a=F.normalize(raw.float(),dim=-1,eps=1e-8)*4.
    centered=F.normalize((raw-avg).float(),dim=-1,eps=1e-8)*4.
    out=torch.cat((a,centered,inp.event.float()),dim=-1).detach()
    require(out.shape==(b,w,D) and not out.requires_grad and
            bool(torch.isfinite(out).all()),"nonfinite semantic frozen ABI")
    return SemanticInputs(inp,out)

class S59NativeEventSemantic(nn.Module):
    def __init__(self):
        super().__init__()
        self.event=nn.Parameter(torch.zeros(D))
        self.transfer=nn.Parameter(torch.zeros(D))
        self.reject=nn.Parameter(torch.zeros(D))
        self.event_bias=nn.Parameter(torch.zeros(1))
        self.transfer_bias=nn.Parameter(torch.zeros(1))
        self.reject_bias=nn.Parameter(torch.zeros(1))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.streaming_causality_certified=False

    def forward(self,x:SemanticInputs,old57:tuple[torch.Tensor,...]):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed
                and not self.streaming_causality_certified,
                "S59 output gate fail-closed")
        d=x.semantic; b,w,_=d.shape
        require(d.shape[-1]==D and len(old57)==4 and
                old57[0].shape==old57[1].shape==(b,w) and
                old57[2].shape==old57[3].shape==(b,w,w) and
                all(not q.requires_grad for q in old57) and
                not d.requires_grad, "old frozen S57 or new native ABI drift")
        # Preserve old S57 recipient ranking as separate frozen graph head.
        # Supervise transfer + rejection with TRUE native representations
        # which S58's 83-dimensional surface-only head did NOT expose.
        return (old57[0]+d@self.event+self.event_bias,
                old57[1]+d@self.transfer+self.transfer_bias,
                old57[2],
                old57[3])

    def rejected_logit(self,d:torch.Tensor,source:torch.Tensor,
                       old57_reject:torch.Tensor,dest:torch.Tensor):
        i=torch.arange(len(source),device=source.device)
        return (old57_reject[i,source,dest] +
                (d[i,source]@self.reject)+self.reject_bias)

    def decode(self,x:SemanticInputs,old57:tuple[torch.Tensor,...]):
        with torch.inference_mode():
            event,transfer,recipient,reject=self(x,old57)
            event=event.masked_fill(~x.event.candidate_event,float('-inf'))
            chosen=event.argmax(-1)
            b,w=x.event.candidate_event.shape
            links=torch.zeros((b,w,w),dtype=torch.bool,device=event.device)
            rejected=torch.zeros_like(links)
            abstain_nontransfer=abstain_unsafe=0
            for row in range(b):
                src=int(chosen[row])
                if float(transfer[row,src])<=0.:
                    abstain_nontransfer+=1
                    continue
                allowed=x.event.valid_pair[row,src]
                if not bool(allowed.any()):
                    abstain_unsafe+=1
                    continue
                dst=int(recipient[row,src].masked_fill(~allowed,float('-inf')).argmax())
                links[row,src,dst]=True
                rej=(reject[row,src,dst]+
                     x.semantic[row,src]@self.reject+self.reject_bias)
                rejected[row,src,dst]=bool(rej>0)
            require(int(links.sum())<=b and
                    not bool((links&~x.event.valid_pair).any()) and
                    not bool((rejected&~links).any()),
                    "new candidate violated quote/BPE/directed safety")
            return {
                "chosen_event":chosen,
                "directed_recipient":links,
                "directed_rejected":rejected,
                "abstain_nontransfer":abstain_nontransfer,
                "abstain_no_safe_target":abstain_unsafe,
                "output_gate":0,
                "candidate_accepted":False,
                "activation_allowed":False,
                "streaming_causality_certified":False,
            }

    def fp16(self):
        with torch.no_grad():
            params=tuple(z.detach().reshape(-1) for z in self.parameters())
            v=torch.cat(params).half().contiguous()
            require(v.shape==(P,) and bool(torch.isfinite(v).all()),"physical FP16 semantic")
            return v

    @classmethod
    def from_fp16(cls,v:torch.Tensor):
        require(v.shape==(P,) and v.dtype==torch.float16 and
                not v.requires_grad and bool(torch.isfinite(v).all()),
                "native semantic readout ABI")
        obj=cls()
        at=0
        with torch.no_grad():
            for p in obj.parameters():
                n=p.numel()
                p.copy_(v[at:at+n].float().reshape(p.shape))
                at+=n
        obj.eval().requires_grad_(False)
        require(torch.equal(obj.fp16(),v),"saved physical semantic FP16 differs")
        return obj


def fit_only_PREVIOUSLY_INSPECTED_real43m_cases(
        groups:list[tuple[SemanticInputs,torch.Tensor,torch.Tensor,
                          torch.Tensor,torch.Tensor]],
        old:S57EventFirstRecipient,
):
    """Six sealed SOURCE groups S53..S58 old 24/24/24/24/24/32 =152.

    Each gold event index, optional recipient and rejection only used
    in supervised loss; model forward has NO labels.
    """
    require(len(groups)==6 and [z[0].semantic.shape[0] for z in groups] ==
            [24,24,24,24,24,32], "S59 old-only six native groups required")
    require(all(not p.requires_grad for p in old.parameters()) and
            old.output_gate==0 and not old.candidate_accepted, "S57 not frozen")
    torch.manual_seed(591118)
    model=S59NativeEventSemantic()
    opt=torch.optim.AdamW(model.parameters(),lr=.016,weight_decay=.014)
    cached=[]
    for inp,idx,has,recipient,rejected in groups:
        b,w,_=inp.semantic.shape
        require(idx.shape==(b,) and has.shape==(b,) and has.dtype==torch.bool
                and recipient.shape==rejected.shape==(b,w,w)
                and torch.equal(recipient.sum((1,2))>0,has)
                and not bool((rejected&~recipient).any())
                and bool(inp.event.candidate_event[
                    torch.arange(b),idx].all()),
                "old inspected manual labels not source aligned")
        with torch.no_grad():
            base=tuple(z.detach().clone() for z in old(inp.event))
        cached.append(base)
    count_links=sum(int(g[3].sum()) for g in groups)
    count_rejected=sum(int(g[4].sum()) for g in groups)
    require(count_links>=80 and count_rejected>=45,
            "S59 original prior TRAIN semantics too sparse")
    # Scale each group by its size, not by manually-authored family gold.
    # Fixed hyperparameters; no NEW S59 HELD or protected EVAL touched.
    for step in range(260):
        opt.zero_grad(set_to_none=True)
        total=None
        for (inp,idx,has,rec,rej),base in zip(groups,cached):
            b,w,_=inp.semantic.shape
            i=torch.arange(b)
            event,transfer,_,old_rej=model(inp,base)
            evloss=F.cross_entropy(
                event.masked_fill(~inp.event.candidate_event,-1e9),idx)
            transloss=F.binary_cross_entropy_with_logits(
                transfer[i,idx],has.float())
            pos=torch.where(has)[0]
            gold_dst=rec[pos,idx[pos]].float().argmax(-1)
            reject_logits=model.rejected_logit(
                inp.semantic,idx[pos],
                old_rej[pos],gold_dst) if False else None
            # Index selected subset correctly; no gold enters model forward.
            rej_logits=old_rej[pos,idx[pos],gold_dst]+(
                inp.semantic[pos,idx[pos]]@model.reject)+model.reject_bias
            rej_label=rej[pos,idx[pos],gold_dst].float()
            rloss=F.binary_cross_entropy_with_logits(rej_logits,rej_label)
            loss=(evloss*.75+transloss*1.8+rloss*1.5)*(b/152.)
            total=loss if total is None else total+loss
        total=total+.0012*sum(p.square().sum() for p in model.parameters())
        require(bool(torch.isfinite(total)), "S59 semantic loss nonfinite")
        total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),"S59 missing/invalid gradient")
        opt.step()
    restored=S59NativeEventSemantic.from_fp16(model.fp16())
    return restored,restored.fp16(),{
        "old_152_groups":[24,24,24,24,24,32],
        "old_real43M_GT_directed_recipient_gold":count_links,
        "old_real43M_GT_directed_rejected_gold":count_rejected,
        "new_FP16_parameters":P,
        "optimizer":"AdamW lr .016 weight_decay .014 260 updates L2 .0012",
        "inputs_used":"frozen original native 320+320 centered and S57 41D",
        "nontransfer_loss":"BCE on OLD gold event (TRAIN only)",
        "new_S59_first_held_OR_protected_EVAL_used":False,
        "S53_to_S58_ALL_spent_train_only_not_independent_evaluation":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False
    }
