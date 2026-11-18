"""S60 trainable source-local three-way event/polarity research adapter.

Explicit factorisation:
  1) choose outer predicate with frozen S59 event + new residual;
  2) classify chosen event into NONE, ASSERTED transfer, NEGATED transfer;
  3) choose <=1 BPE/quote-safe recipient via old frozen S57 pair head.
Novelty vs S59: direct joint 3-way objective instead of two independent
binary transfer and rejected heads, with source-clause local negation,
passive-auxiliary and original real Calibri43M+GT native320 features.

Full-context read-only experiment; NOT online streaming causal, no
active Calibri/GT integration, all old paid weights physically frozen.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s20.s56_grammar_clause_scope_semantic_readout import (
    grammar_scope_metadata,VERB_BITS,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    S59NativeEventSemantic,SemanticInputs,D as SEM_D,
)
NEG=frozenset(("не","ни"))
AUX=frozenset(("был","была","было","были","будет","будут","быть"))
SCOPE_D=12
D=SEM_D+SCOPE_D
P=4*D+4 # 2,776 independent trainable FP16 parameters
assert SEM_D==681 and D==693 and P==2776

def require(flag,message):
    if not bool(flag):raise ValueError("S60 original43M GT joint3 research: "+message)

@dataclass(frozen=True)
class S60Input:
    base:SemanticInputs
    scope:torch.Tensor  # B,W,12 raw source-only, no manual labels
    fused:torch.Tensor  # B,W,693

def source_local_scope(x:SemanticInputs, texts:tuple[str,...])->S60Input:
    b,w,_=x.semantic.shape
    require(len(texts)==b and not x.semantic.requires_grad
            and x.semantic.shape[-1]==SEM_D
            and x.event.event.shape[:2]==(b,w),"frozen source-only input ABI")
    scope=torch.zeros((b,w,SCOPE_D),dtype=torch.float32)
    for row,text in enumerate(texts):
        toks,cids,clauses,neg_indices,_,_=grammar_scope_metadata(text)
        require(len(toks)<=w and len(toks)>=3,
                "original native word and raw grammar alignment")
        for i,t in enumerate(toks):
            ci=cids[i]
            negs=neg_indices[ci]
            left=[i-k for k in negs if k<i]
            right=[k-i for k in negs if k>i]
            prev=[toks[k].form.casefold() for k in range(max(0,i-3),i)]
            other=any(v for c,v in neg_indices.items() if c!=ci)
            form=t.form.casefold()
            features=(
                float(1 in left),
                float(any(1<=d<=3 for d in left)),
                float(any(1<=d<=2 for d in right)),
                float(bool(negs)),
                float(other),
                float(any(z in AUX for z in prev)),
                float(bool(negs) and any(z in AUX for z in prev)),
                float(bool(t.features[VERB_BITS[2]])),
                float(bool(t.features[VERB_BITS[0]] or
                           t.features[VERB_BITS[1]])),
                min(len(negs),4)/4.,
                min(len(clauses[ci]),24)/24.,
                i/max(1,len(toks)-1),
            )
            require(len(features)==SCOPE_D,"source-local scope ABI drift")
            scope[row,i]=torch.tensor(features)
    fused=torch.cat((x.semantic.float(),scope),dim=-1).detach()
    require(fused.shape==(b,w,D) and not fused.requires_grad
            and bool(torch.isfinite(fused).all()),
            "S60 native320 plus raw scope nonfinite")
    return S60Input(x,scope.detach(),fused)

class S60Joint3(nn.Module):
    def __init__(self):
        super().__init__()
        self.event=nn.Parameter(torch.zeros(D))
        self.event_bias=nn.Parameter(torch.zeros(1))
        self.classes=nn.Parameter(torch.zeros(3,D))
        self.class_bias=nn.Parameter(torch.zeros(3))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.streaming_causality_certified=False

    def forward(self,x:S60Input,old57:tuple[torch.Tensor,...],
                old59:S59NativeEventSemantic):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed
                and not self.streaming_causality_certified,
                "S60 research gate must remain zero")
        require(x.fused.shape[-1]==D and not x.fused.requires_grad
                and len(old57)==4 and all(not t.requires_grad for t in old57)
                and not any(t.requires_grad for t in old59.parameters())
                and old59.output_gate==0,
                "original native / frozen S57,S59 ABI")
        old_ev,old_transfer,old_rec,old_rej=old59(x.base,old57)
        b,w=x.fused.shape[:2]
        event=old_ev+x.fused@self.event+self.event_bias
        # Best eligible target is a source-only learned OLD S57 ranking.
        pair_valid=x.base.event.valid_pair
        masked=old_rec.masked_fill(~pair_valid,-1e9)
        target=masked.argmax(-1)
        row=torch.arange(b).unsqueeze(-1)
        src=torch.arange(w).unsqueeze(0)
        rr=old_rej[row,src,target]
        # S59 old reject head was also trained WITHOUT held labels.
        rejected=rr+(x.base.semantic@old59.reject)+old59.reject_bias
        zero=torch.zeros_like(old_transfer)
        old_classes=torch.stack((zero,old_transfer,
                                 old_transfer+rejected),dim=-1)
        three=old_classes+F.linear(x.fused,self.classes,self.class_bias)
        require(event.shape==(b,w) and three.shape==(b,w,3)
                and target.shape==(b,w),"S60 3class native output ABI drift")
        return event,three,target

    def decode(self,x:S60Input,old57:tuple[torch.Tensor,...],
               old59:S59NativeEventSemantic):
        with torch.inference_mode():
            e,cl,targets=self(x,old57,old59)
            eligible=x.base.event.candidate_event
            chosen=e.masked_fill(~eligible,float("-inf")).argmax(-1)
            b,w=eligible.shape
            pairs=torch.zeros((b,w,w),dtype=torch.bool,device=e.device)
            rejected=torch.zeros_like(pairs)
            action=cl.argmax(-1)
            none=without_target=0
            classes=[]
            for i in range(b):
                src=int(chosen[i]); kind=int(action[i,src])
                classes.append(kind)
                if kind==0:
                    none+=1;continue
                dst=int(targets[i,src])
                if not bool(x.base.event.valid_pair[i,src,dst]):
                    without_target+=1;continue
                pairs[i,src,dst]=True
                rejected[i,src,dst]=(kind==2)
            require(int(pairs.sum())<=b
                    and not bool((pairs&~x.base.event.valid_pair).any())
                    and not bool((rejected&~pairs).any()),
                    "S60 quote/native unsafe or more than one pair")
            return {"chosen_event":chosen,"chosen_event_class":torch.tensor(
                classes,dtype=torch.long,device=e.device),
                "directed_recipient":pairs,"directed_rejected":rejected,
                "no_transfer_abstain":none,"no_safe_target_abstain":without_target,
                "output_gate":0,"candidate_accepted":False,
                "activation_allowed":False,
                "streaming_causality_certified":False}

    def fp16(self):
        with torch.no_grad():
            z=torch.cat(tuple(q.detach().flatten() for q in self.parameters())).half()
            require(z.shape==(P,) and bool(torch.isfinite(z).all()),
                    "S60 trained 2776 FP16 source")
            return z.contiguous()

    @classmethod
    def from_fp16(cls,z):
        require(z.shape==(P,) and z.dtype==torch.float16
                and not z.requires_grad and bool(torch.isfinite(z).all()),
                "S60 2776 FP16 physical shape drift")
        obj=cls();cursor=0
        with torch.no_grad():
            for p in obj.parameters():
                count=p.numel()
                p.copy_(z[cursor:cursor+count].float().reshape(p.shape))
                cursor+=count
        obj.eval().requires_grad_(False)
        require(torch.equal(obj.fp16(),z),"S60 actual FP16 persisted drift")
        return obj

def fit_old184(groups,old57,old59):
    require(len(groups)==7
            and [int(z[0].fused.shape[0]) for z in groups]==
                 [24,24,24,24,24,32,32],
            "S60 train only S53..S59 already spent 184 cases")
    require(all(not p.requires_grad for m in (old57,old59)
                for p in m.parameters()),"S60 original paid parents unfrozen")
    torch.manual_seed(601118)
    model=S60Joint3()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.02)
    cached=[]
    count=[0,0,0]
    for x,event,classes,rec,rej in groups:
        b,w,_=x.fused.shape
        require(event.shape==classes.shape==(b,)
                and rec.shape==rej.shape==(b,w,w)
                and classes.dtype==torch.long and
                bool(((classes>=0)&(classes<3)).all())
                and bool(x.base.event.candidate_event[torch.arange(b),event].all()),
                "gold old184 event class must be valid")
        for k in range(3):count[k]+=int((classes==k).sum())
        with torch.no_grad():
            logits=tuple(z.detach().clone() for z in old57(x.base.event))
        cached.append(logits)
    require(sum(count)==184 and count[0]>=45 and count[2]>=55,
            "old184 3-way class balance unexpected")
    weights=torch.tensor([1.,1.1,1.3])
    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        total=None
        for (x,event,classes,_,_),orig in zip(groups,cached):
            b=x.fused.shape[0]; row=torch.arange(b)
            logits,tri,_=model(x,orig,old59)
            el=F.cross_entropy(logits.masked_fill(
                ~x.base.event.candidate_event,-1e9),event)
            cl=F.cross_entropy(tri[row,event],classes,weight=weights)
            loss=(.75*el+2.*cl)*(b/184.)
            total=loss if total is None else total+loss
        total=total+.0008*sum(p.square().sum() for p in model.parameters())
        require(bool(torch.isfinite(total)),"S60 3way fixed loss nonfinite")
        total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),"S60 gradient absent or invalid")
        optimizer.step()
    saved=S60Joint3.from_fp16(model.fp16())
    return saved,saved.fp16(),{
        "old_184_training_groups":[24,24,24,24,24,32,32],
        "old_supervised_event_classes_NONE_ASSERTED_REJECTED":count,
        "new_FP16_parameters":P,
        "loss":"0.75 event CE +2.0 3way CE weighted 1/1.1/1.3; 300 AdamW, lr.012 wd.02, reg.0008",
        "new_S60_held_raw_gold_never_seen":True,
        "original43M_GT_and_old_S26_S57_S59_all_frozen":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
