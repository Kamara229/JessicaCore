"""S62 joint event×recipient pair decoder, original43M+GT RESEARCH ONLY.

Unlike S61, final event selection depends on BOTH predicate and learned
recipient/explicit NONE scores. Frozen S59 proposes top8 possible source
events using ONLY original frozen 43M+GT source representations. For EACH
source in this source-only beam, frozen S61 scores the native source-target
pairs. Newly trainable source/target and polarity projections jointly
re-rank event×(all safe recipients+NONE), with explicit source-event
supervision and target-aware ranking loss on OLD SPENT248 ONLY.

BEAM8 is a declared experimental limitation: sources outside this frozen
proposal beam cannot be recovered. No oracle, new HELD gold, or protected
EVAL enters model forward. Not autoregressive causal certified. Baseline
weights (Cal43M, GT, S26, S57, S59, S61) never updated.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s20.s60_native43m_local_triclass_model import S60Input
from research.nra3_s20.s59_native320_contrastive_event_polarity import S59NativeEventSemantic
from research.nra3_s20.s61_native_pair_null_polarity_2752 import S61NativePairNull

K=8
SOURCE_D=693
PAIR_D=320+54+12+1  # frozen native elementwise interaction and source S56 pair
PAID=(SOURCE_D+1)*2+(PAIR_D+1)*2+2
assert PAIR_D==387 and PAID==2166

def require(flag,reason):
    if not bool(flag):raise ValueError("S62 43MGT JOINT SOURCE×TARGET gate0: "+reason)

@dataclass(frozen=True)
class Beam:
    sources:torch.Tensor     # B×8 source-only topK frozen S59 events
    allowed:torch.Tensor     # B×8 candidate mask
    event_base:torch.Tensor  # B×8, pretrained S59 event scores
    source_fused:torch.Tensor # B×8×693 source-only frozen feature
    pair_feature:torch.Tensor # B×8×W×387 source+target no gold
    original_target:torch.Tensor # B×8×(W+1) frozen S61 trained pair/NONE scores
    original_polarity:torch.Tensor # B×8×W frozen S61 pair negative logits
    target_safe:torch.Tensor # B×8×W frozen quote, BPE safety
    word_count:int

def source_only_frozen_beam(
    x:S60Input, old57:tuple[torch.Tensor,...],
    old59:S59NativeEventSemantic, old61:S61NativePairNull
)->Beam:
    b,w,_=x.fused.shape
    require(x.fused.shape==(b,w,SOURCE_D)
            and x.base.semantic.shape==(b,w,681)
            and all(not z.requires_grad for z in old57)
            and all(not p.requires_grad for m in (old59,old61)
                    for p in m.parameters())
            and old59.output_gate==old61.output_gate==0
            and not x.fused.requires_grad and w>=K,
            "original source-only frozen paid/nativity ABI invalid")
    with torch.no_grad():
        e=old59(x.base,old57)[0].detach()
        eligible=x.base.event.candidate_event
        masked=e.masked_fill(~eligible,float("-inf"))
        require(bool(eligible.any(-1).all()),"no candidate outer events in original CPU")
        top=masked.topk(K,dim=-1)
        allowed=torch.isfinite(top.values)
        best=top.indices[:,0:1].expand(-1,K)
        sources=torch.where(allowed,top.indices,best)
        safe=[];target=[];polarity=[];features=[]
        sem=x.base.semantic[:,:,:320]
        bi=torch.arange(b)
        tgt=sem.unsqueeze(1).expand(-1,K,-1,-1)
        for k in range(K):
            src=sources[:,k]
            joint,rejected,mask=old61.score_sources(x,old57,old59,src)
            target.append(joint.detach())
            polarity.append(rejected.detach())
            safe.append(mask.detach())
            source=sem[bi,src].unsqueeze(1).expand(-1,w,-1)
            original_pair54=x.base.event.pair[bi,src]
            scope=x.scope[bi,src].unsqueeze(1).expand(-1,w,-1)
            distance=(torch.arange(w).unsqueeze(0)-src[:,None]).float()
            distance=(distance/max(w-1,1)).unsqueeze(-1)
            pair=torch.cat((source*sem,original_pair54,scope,distance),dim=-1)
            require(pair.shape==(b,w,PAIR_D)
                    and bool(torch.isfinite(pair).all()),
                    "S62 original native pair joint residual invalid")
            features.append(pair.detach())
        original_target=torch.stack(target,dim=1)
        polarity=torch.stack(polarity,dim=1)
        safe=torch.stack(safe,dim=1)
        features=torch.stack(features,dim=1)
        sf=x.fused.gather(1,sources.unsqueeze(-1).expand(-1,-1,SOURCE_D))
        old_event=e.gather(1,sources)
        return Beam(sources.detach(),allowed.detach(),old_event.detach(),
                    sf.detach(),features.detach(),original_target.detach(),
                    polarity.detach(),safe.detach(),w)

class S62JointEventRecipient(nn.Module):
    def __init__(self):
        super().__init__()
        self.event=nn.Parameter(torch.zeros(SOURCE_D))
        self.event_bias=nn.Parameter(torch.zeros(1))
        self.none=nn.Parameter(torch.zeros(SOURCE_D))
        self.none_bias=nn.Parameter(torch.zeros(1))
        self.pair=nn.Parameter(torch.zeros(PAIR_D))
        self.pair_bias=nn.Parameter(torch.zeros(1))
        self.polar=nn.Parameter(torch.zeros(PAIR_D))
        self.polar_bias=nn.Parameter(torch.zeros(1))
        self.event_gain=nn.Parameter(torch.zeros(1))
        self.old_pair_gain=nn.Parameter(torch.zeros(1))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.streaming_causality_certified=False

    def forward(self,b:Beam):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and
                not self.streaming_causality_certified,
                "S62 gate cannot be activated")
        n,k,d=b.source_fused.shape
        require(k==K and d==SOURCE_D
                and b.pair_feature.shape==(n,K,b.word_count,PAIR_D)
                and b.original_target.shape==(n,K,b.word_count+1)
                and b.original_polarity.shape==(n,K,b.word_count)
                and b.target_safe.shape==(n,K,b.word_count)
                and b.allowed.shape==b.sources.shape==(n,K)
                and not any(z.requires_grad for z in (
                    b.source_fused,b.pair_feature,b.original_target)),
                "S62 source-only candidate beam frozen shape changed")
        event=(1.+self.event_gain)*b.event_base + (
            b.source_fused@self.event+self.event_bias)
        pair=(1.+self.old_pair_gain)*b.original_target[:,:,:b.word_count] + (
            b.pair_feature@self.pair+self.pair_bias)
        null=(1.+self.old_pair_gain)*b.original_target[:,:,-1] + (
            b.source_fused@self.none+self.none_bias)
        pair=pair.masked_fill(~b.target_safe,-1e9)
        candidate=torch.cat((pair,null.unsqueeze(-1)),dim=-1)+event.unsqueeze(-1)
        candidate=candidate.masked_fill(~b.allowed.unsqueeze(-1),-1e9)
        polarity=b.original_polarity+(
            b.pair_feature@self.polar+self.polar_bias)
        require(candidate.shape==(n,K,b.word_count+1) and
                polarity.shape==(n,K,b.word_count)
                and bool(torch.isfinite(candidate).all())
                and bool(torch.isfinite(polarity).all()),
                "S62 event-target scores invalid")
        return candidate,polarity

    def decode(self,b:Beam):
        with torch.inference_mode():
            joint,polarity=self(b)
            n,k,w1=joint.shape;w=w1-1
            ranked=joint.reshape(n,k*w1).argmax(-1)
            source_slot=torch.div(ranked,w1,rounding_mode="floor")
            target=torch.remainder(ranked,w1)
            ix=torch.arange(n)
            chosen=b.sources[ix,source_slot]
            recipient=torch.zeros((n,w,w),dtype=torch.bool,device=joint.device)
            rejected=torch.zeros_like(recipient)
            for row in range(n):
                src=int(chosen[row]);dest=int(target[row])
                if dest==w:continue
                require(bool(b.target_safe[row,source_slot[row],dest]),
                        "source pair violated original BPE or quote mask")
                recipient[row,src,dest]=True
                rejected[row,src,dest]=float(
                    polarity[row,source_slot[row],dest])>0
            require(int(recipient.sum())<=n and
                    not bool((rejected&~recipient).any()),
                    "at most one source-target edge with conditional polarity")
            return {
                "chosen_event":chosen,
                "chosen_beam_slot":source_slot,
                "directed_recipient":recipient,
                "directed_rejected":rejected,
                "no_target_abstentions":int((target==w).sum()),
                "output_gate":0,"candidate_accepted":False,
                "activation_allowed":False,
                "streaming_causality_certified":False,
            }

    def fp16(self):
        with torch.no_grad():
            p=torch.cat([x.detach().reshape(-1)
                         for x in self.parameters()]).half().contiguous()
            require(p.shape==(PAID,) and bool(torch.isfinite(p).all()),
                    "S62 physical trained2166 FP16 invalid")
            return p

    @classmethod
    def from_fp16(cls,p):
        require(p.shape==(PAID,) and p.dtype==torch.float16
                and bool(torch.isfinite(p).all()) and not p.requires_grad,
                "S62 saved readout FP16 source invalid")
        m=cls();j=0
        with torch.no_grad():
            for z in m.parameters():
                c=z.numel()
                z.copy_(p[j:j+c].float().reshape(z.shape))
                j+=c
        require(j==PAID,"physical FP16 parameter count drift")
        m.eval().requires_grad_(False)
        require(torch.equal(m.fp16(),p),"S62 FP16 reload mismatch")
        return m

def fit_SPENT248_only(groups):
    """groups = (beam, gold true source idx, gold target/NONE W, rejection).
    Manual gold is used here ONLY for loss labels; not in S62 forward/cache.
    S62 separate NEW32 must not be imported here.
    """
    require(len(groups)==9 and [z[0].sources.shape[0] for z in groups]==
            [24,24,24,24,24,32,32,32,32],
            "S62 requires exactly nine previously scored old groups=248")
    prepared=[]
    count_present=count_negative=count_eligible=0
    total_rows=0
    for b,ev,recipient,rejected in groups:
        n,k=b.sources.shape;w=b.word_count
        require(ev.shape==recipient.shape==rejected.shape==(n,)
                and ev.dtype==recipient.dtype==torch.long
                and rejected.dtype==torch.bool
                and bool(((recipient>=0)&(recipient<=w)).all())
                and bool((~rejected|(recipient!=w)).all()),
                "old248 labels incorrectly keyed to source-only beam")
        inside=(b.sources==ev[:,None])&b.allowed
        valid=inside.any(-1)
        first=inside.float().argmax(-1)
        pos=(recipient!=w)&valid
        if bool(pos.any()):
            idx=torch.where(pos)[0]
            require(bool(b.target_safe[idx,first[idx],recipient[idx]].all()),
                    "S62 old248 gold target not in safe source beam pair")
        count_present+=int((recipient!=w).sum())
        count_negative+=int(rejected.sum())
        count_eligible+=int(valid.sum())
        total_rows+=n
        prepared.append((b,ev,recipient,rejected,valid,first))
    require(total_rows==248 and count_present>=145 and count_negative>=70
            and count_eligible>=180,
            "S62 spent248 source/target supervision or beam8 coverage too low")
    torch.manual_seed(621118)
    m=S62JointEventRecipient()
    opt=torch.optim.AdamW(m.parameters(),lr=.008,weight_decay=.018)
    for step in range(220):
        opt.zero_grad(set_to_none=True)
        whole=None
        for b,ev,recipient,rejected,valid,first in prepared:
            scores,polar=m(b)
            n,k,w1=scores.shape;w=w1-1
            rows=torch.where(valid)[0]
            assert len(rows)>0
            correct=first[rows]*w1+recipient[rows]
            joint=F.cross_entropy(scores[rows].reshape(-1,k*w1),correct)
            # Additional source event supervision uses each beam's
            # source-level max over target, not manual event in forward.
            escores=scores[rows].logsumexp(-1)
            evloss=F.cross_entropy(escores,first[rows])
            r=torch.where(valid & (recipient!=w))[0]
            neg=F.binary_cross_entropy_with_logits(
                polar[r,first[r],recipient[r]],rejected[r].float())
            loss=(1.65*joint+.4*evloss+.8*neg)*(n/248.)
            whole=loss if whole is None else whole+loss
        whole=whole+.0006*sum(p.square().sum() for p in m.parameters())
        require(bool(torch.isfinite(whole)), "S62 optimizer loss not finite")
        whole.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in m.parameters()),"S62 frozen joint head missing gradient")
        opt.step()
    frozen=S62JointEventRecipient.from_fp16(m.fp16())
    return frozen,frozen.fp16(),{
        "old_spent_9_groups":[24,24,24,24,24,32,32,32,32],
        "old_spent248_source_rows":total_rows,
        "old_spent248_directed_recipient_labels":count_present,
        "old_spent248_rejected_labels":count_negative,
        "frozen_S59_beam8_contains_true_outer_event":count_eligible,
        "beam8_cannot_recover_gold_outside_beam":total_rows-count_eligible,
        "new_FP16_parameter_count":PAID,
        "S61_frozen_pair_null_used_only_precommitted_OLD248":True,
        "train_only_gold_used_in_CE_not_candidate_proposals":True,
        "new_S62_new32_RAW_and_GOLD_never_seen":True,
        "original_Calibri43M_GT_old_paid_frozen":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
