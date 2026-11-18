"""S61 original Calibri43M+Green Tunnel JOINT recipient-vs-NONE and polarity.

Unlike S60 (frozen S57 pair rank), S61 learns a NEW target-aware source /
recipient relation using frozen original genuine post-layer8 native320
BOTH at the source and candidate target + elementwise interaction,
original S56 pair54, local S60 scope12, and old S57 logits (1028 dims).
An independent source-specific NONE decision (693 dims) is learned
JOINTLY with recipient ranking using CE over W targets + NONE. Second
new pair-aware logit corrects REJECTED polarity for the chosen target.

Event predicate selection is deliberately FROZEN S59 (not oracle).
No gold predicate/target/scope in model forward or inference. Old216
gold labels enter TRAIN loss only; new S61 raw32/gold32 excluded.
No source/original GT/S26/S57/S59/S60 weights modified.
Full-context offline research only, not autoregressive causal.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s20.s60_native43m_local_triclass_model import (
    S60Input,source_local_scope,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    S59NativeEventSemantic,
)

PAIR_D=3*320+54+12+2 # original src native320, tgt native320, elementwise, oldpair54, scope12, old2
NULL_D=681+12          # original S59 semantic681 and S60 scope12
PAID=2*(PAIR_D+1)+(NULL_D+1) # 2752 FP16
assert PAIR_D==1028 and NULL_D==693 and PAID==2752

def require(flag,message):
    if not bool(flag):
        raise ValueError("S61 original43M GT target-aware 2752FP16: "+message)

class S61NativePairNull(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair=nn.Parameter(torch.zeros(PAIR_D))
        self.pair_bias=nn.Parameter(torch.zeros(1))
        self.null=nn.Parameter(torch.zeros(NULL_D))
        self.null_bias=nn.Parameter(torch.zeros(1))
        self.polar=nn.Parameter(torch.zeros(PAIR_D))
        self.polar_bias=nn.Parameter(torch.zeros(1))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.streaming_causality_certified=False

    def score_sources(self,x:S60Input,old57:tuple[torch.Tensor,...],
                      old59:S59NativeEventSemantic,
                      source:torch.Tensor):
        """Source indices may come from old S59 inference OR OLD TRAIN labels.

        No gold role/class access here. Return B×(W+1) target/NONE scores,
        B×W candidate-local REJECTED logit, and source-safe mask.
        """
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and
                not self.streaming_causality_certified,
                "research gate fail-closed")
        b,w,_=x.fused.shape
        require(x.base.semantic.shape==(b,w,681) and
                x.scope.shape==(b,w,12) and
                len(old57)==4 and
                old57[0].shape==old57[1].shape==(b,w) and
                old57[2].shape==old57[3].shape==(b,w,w) and
                source.shape==(b,) and source.dtype==torch.long and
                bool(((source>=0)&(source<w)).all()) and
                bool(x.base.event.candidate_event[
                    torch.arange(b,device=source.device),source].all()) and
                not x.fused.requires_grad and
                all(not z.requires_grad for z in old57) and
                all(not p.requires_grad for p in old59.parameters()) and
                old59.output_gate==0,"frozen source native/paid ABI drift")
        bi=torch.arange(b,device=source.device)
        sem=x.base.semantic
        ss=sem[bi,source,:320].unsqueeze(1).expand(-1,w,-1)
        target=sem[:,:,:320]
        product=ss*target
        pair54=x.base.event.pair[bi,source]
        scope=x.scope[bi,source].unsqueeze(1).expand(-1,w,-1)
        old_pair=old57[2][bi,source]
        old_rej=old57[3][bi,source]
        combined=torch.cat((ss,target,product,pair54,scope,
                            old_pair.unsqueeze(-1),old_rej.unsqueeze(-1)),dim=-1)
        require(combined.shape==(b,w,PAIR_D) and
                bool(torch.isfinite(combined).all()),
                "S61 source-target original native pair construction nonfinite")
        _,old_transfer,_,old_polarity=old59(x.base,old57)
        safe=x.base.event.valid_pair[bi,source]
        valid_for_max=old_pair.masked_fill(~safe,-1e9)
        # Per-source center preserves frozen S57 recipient *rank* but
        # prevents arbitrary scale/offset from making NONE unreachable.
        centered=old_pair-valid_for_max.max(-1,keepdim=True).values
        candidate=(old_transfer[bi,source,None]+centered+
                   F.linear(combined,self.pair,self.pair_bias))
        candidate=candidate.masked_fill(~safe,-1e9)
        null=F.linear(x.fused[bi,source],self.null,self.null_bias)
        joined=torch.cat((candidate,null[:,None]),dim=-1)
        base_rejected=(old_polarity[bi,source,None,:].gather(
                       2,torch.arange(w,device=source.device)[None,None,:]
                           .expand(b,1,-1)).squeeze(1)
                       if False else None)
        # Frozen S59 source-local negation correction + old S57 pair local.
        rej=(old_rej +
             sem[bi,source]@old59.reject[:,None] if False else old_rej)
        s59_rej=(sem[bi,source]@old59.reject+
                 old59.reject_bias).unsqueeze(-1)
        polar=old_rej+s59_rej+F.linear(combined,self.polar,self.polar_bias)
        require(joined.shape==(b,w+1) and polar.shape==(b,w)
                and bool(torch.isfinite(joined).all())
                and bool(torch.isfinite(polar).all()),
                "S61 NONE-vs-recipient logits invalid")
        return joined,polar,safe

    def decode(self,x:S60Input,old57:tuple[torch.Tensor,...],
               old59:S59NativeEventSemantic):
        with torch.inference_mode():
            event=old59(x.base,old57)[0].masked_fill(
                ~x.base.event.candidate_event,float('-inf'))
            source=event.argmax(-1)
            scored,polar,safe=self.score_sources(x,old57,old59,source)
            chosen=scored.argmax(-1)
            b,w=x.base.event.candidate_event.shape
            recipient=torch.zeros((b,w,w),dtype=torch.bool,device=event.device)
            rejected=torch.zeros_like(recipient)
            no_target=0
            for row in range(b):
                dest=int(chosen[row])
                if dest==w:
                    no_target+=1;continue
                src=int(source[row])
                require(bool(safe[row,dest]),"S61 invalid directed pair escaped mask")
                recipient[row,src,dest]=True
                rejected[row,src,dest]=(float(polar[row,dest])>0)
            require(int(recipient.sum())<=b and
                    not bool((recipient&~x.base.event.valid_pair).any())
                    and not bool((rejected&~recipient).any()),
                    "S61 at most one BPE-safe link and conditional REJECTED")
            return {
                "chosen_event":source,
                "directed_recipient":recipient,
                "directed_rejected":rejected,
                "no_target_abstentions":no_target,
                "output_gate":0,
                "candidate_accepted":False,
                "activation_allowed":False,
                "streaming_causality_certified":False,
            }

    def fp16(self):
        with torch.no_grad():
            data=torch.cat([p.detach().reshape(-1)
                            for p in self.parameters()]).half().contiguous()
            require(data.shape==(PAID,) and bool(torch.isfinite(data).all()),
                    "S61 physical pair and null FP16 wrong")
            return data

    @classmethod
    def from_fp16(cls,z):
        require(z.dtype==torch.float16 and z.shape==(PAID,)
                and bool(torch.isfinite(z).all()) and not z.requires_grad,
                "S61 saved 2752FP16 parameter count mismatch")
        m=cls();start=0
        with torch.no_grad():
            for p in m.parameters():
                length=p.numel()
                p.copy_(z[start:start+length].float().reshape(p.shape))
                start+=length
        require(start==PAID,"parameter ABI cursor")
        m.eval().requires_grad_(False)
        require(torch.equal(m.fp16(),z),"S61 fp16 physical reload drift")
        return m


def fit_SPENT_216_only(groups,old57,old59):
    """Teacher force gold event ONLY in old216 TRAIN; all new S61 is excluded."""
    require(len(groups)==8 and [z[0].fused.shape[0] for z in groups]==
            [24,24,24,24,24,32,32,32],
            "S61 only eight SPENT genuine43M GT sources = 216")
    require(all(not p.requires_grad for m in (old57,old59)
                for p in m.parameters()),"old model unfrozen")
    torch.manual_seed(611118)
    model=S61NativePairNull()
    opt=torch.optim.AdamW(model.parameters(),lr=.011,weight_decay=.025)
    originals=[]
    positives=rejects=0
    for x,event,labels,rej in groups:
        b,w,_=x.fused.shape
        require(event.shape==(b,) and labels.shape==(b,)
                and rej.shape==(b,) and
                labels.dtype==event.dtype==torch.long and
                rej.dtype==torch.bool and
                bool(x.base.event.candidate_event[
                    torch.arange(b),event].all()) and
                bool(((labels>=0)&(labels<=w)).all()),
                "old216 gold event/index and NONE=W ABI")
        has=labels!=w
        positives+=int(has.sum())
        rejects+=int(rej.sum())
        require(bool((~rej|has).all()),"REJECTED cannot be NONE")
        if bool(has.any()):
            i=torch.where(has)[0]
            require(bool(x.base.event.valid_pair[
                i,event[i],labels[i]].all()),
                "gold old216 directed recipient not safe")
        with torch.no_grad():
            logits=tuple(z.detach().clone() for z in old57(x.base.event))
        originals.append(logits)
    require(positives>=130 and rejects>=65,
            "S61 old216 TRAIN recipient/rejected supervision too small")
    # No threshold/hyperparameter selection from new S61 HELD.
    for step in range(280):
        opt.zero_grad(set_to_none=True)
        total=None
        for (x,event,labels,rejected),old in zip(groups,originals):
            b,w,_=x.fused.shape
            joint,polarity,safe=model.score_sources(x,old,old59,event)
            # Full contrastive pair ranking against every eligible target,
            # INCLUDING explicit NONE on all nontransfer dative/quotes.
            ranking=F.cross_entropy(joint,labels)
            present=torch.where(labels!=w)[0]
            truth=labels[present]
            neg=F.binary_cross_entropy_with_logits(
                polarity[present,truth],rejected[present].float(),
                pos_weight=torch.tensor(1.0))
            loss=(ranking*1.8+neg*.85)*(b/216.)
            total=loss if total is None else total+loss
        total=total+.0008*sum(p.square().sum() for p in model.parameters())
        require(bool(torch.isfinite(total)),"S61 train loss nonfinite")
        total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),"S61 missing source-target gradient")
        opt.step()
    frozen=S61NativePairNull.from_fp16(model.fp16())
    return frozen,frozen.fp16(),{
        "train_groups_old":[24,24,24,24,24,32,32,32],
        "train_rows":216,
        "old_positive_recipient_edges":positives,
        "old_rejected_edges":rejects,
        "new_fp16_parameters":PAID,
        "source_target_features":PAIR_D,
        "none_features":NULL_D,
        "optimizer":"AdamW 280 steps lr=.011 wd=.025, rank1.8, polar0.85, L2 .0008",
        "train_gold_event_teacher_forcing_only_not_inference":True,
        "new_S61_HELD_raw_or_gold_never_seen":True,
        "original43M_GT_and_S26_S57_S59_frozen":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
