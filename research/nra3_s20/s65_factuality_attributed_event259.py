"""NRA3 S65: source-only attributed-vs-factual joint event/target 3-way readout.

Original Calibri43M/Green Tunnel, old S26/S57/S59/S61/S62/S63 frozen.
New 259 FP16 trained ONLY on previously scored SPENT344 (S53..S64).
Sixteen new source-only discourse, clause and local-negation features
supplement prior S63 raw quote20. S65 learns explicitly 3 factuality
classes NONE, ASSERTED, REJECTED for gold old TRAIN event and jointly
trains final global event×target/explicit NONE and directed polarity.
No oracle source or gold target at inference. Strict BPE and malformed
quote fail-close. Prefix audit remains diagnostic and conservative
source-only; NO streaming autoregressive causal certification.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    S62JointEventRecipient,K,
)
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import (
    S63Overlay,S63QuoteOuterSafe,quote_overlay,REPORT_PREFIXES,
    TRANSFER_PREFIXES,
)

FEATURE=36
PAID=(FEATURE+1)*4 + 3*(FEATURE+1)
assert PAID==259

ATTRIBUTION=frozenset(("якобы","будто","словам","сообщениям","версии",
                       "слухам","вроде","мол","по","согласно"))
CLAIM=frozenset(("говорили","утверждали","рассказывали","поговаривали",
                 "считали","написано","записано","цитировалось","версию",
                 "вспоминал","уверяла","сказал","сказала","утверждал"))
REAL=frozenset(("действительно","фактически","реальности","самом",
                "деле","действительности","настоящей"))
PIVOT=frozenset(("но","однако","зато","хотя","тем","временем",
                 "действительно","затем","всё-таки"))
SUBORD=frozenset(("когда","пока","хотя","после","прежде","если","что","будто"))
NEG=frozenset(("не","ни","ничего","никто"))
ENDINGS=("-ен","-ена","-ено","-ены","-ан","-ана","-ано",
         "-аны","-т","-та","-то","-ты")

def require(ok,why):
    if not bool(ok):raise ValueError("S65 TRUE43M GT attribution scope gate0: "+why)

@dataclass(frozen=True)
class S65Input:
    older:S63Overlay
    factual_source:torch.Tensor # B×K×36 text only
    strict_native_or_quote_unsafe:torch.Tensor # B

def discourse_features(
    base:S63Overlay,raw_texts:tuple[str,...],
)->S65Input:
    b,k=base.original.sources.shape
    require(k==K and len(raw_texts)==b and
            base.source_quote.shape==(b,K,20)
            and not base.source_quote.requires_grad,
            "S65 frozen older quote20 and raw source mismatch")
    f=torch.zeros((b,K,FEATURE),dtype=torch.float32)
    f[...,:20]=base.source_quote.detach()
    for i,text in enumerate(raw_texts):
        words=raw_surface_grammar(text)
        lo=[t.form.casefold() for t in words]
        require(len(words)<=base.original.word_count and len(words)>=4,
                "S65 original raw unicode words/native CPU alignment")
        for slot in range(K):
            ix=int(base.original.sources[i,slot])
            require(0<=ix<len(words),"frozen S59 beam event out of range")
            token=words[ix]
            prev=lo[max(0,ix-7):ix]
            preceding=lo[:ix]
            local_left=max((j for j in range(ix) if
                  any(p in text[words[j].end:token.start] for p in ".;!?")),
                  default=0)
            local=lo[local_left:ix]
            before_text=text[:token.start].casefold()
            marker_any=any(v in preceding for v in ATTRIBUTION)
            # A reported clause introduced by "будто", "якобы",
            # "по словам"/"по версии" never forces a target/predicate:
            # this is a learned feature only, not a gold label rule.
            attributed=(marker_any and
                        any(z in preceding for z in CLAIM|ATTRIBUTION))
            in_quote=bool(f[i,slot,1])
            report_only=bool(f[i,slot,4]) and bool(f[i,slot,0])
            transfer=lo[ix].startswith(TRANSFER_PREFIXES)
            passive=lo[ix].endswith(ENDINGS)
            offsets=[token.start/max(1,len(text)),
                     token.end/max(1,len(text))]
            bits=(
                float(attributed),
                float(any(z in prev for z in ATTRIBUTION)),
                float(any(z in preceding for z in CLAIM)),
                float(any(z in prev for z in REAL)),
                float(any(z in prev for z in PIVOT)),
                float(any(z in prev for z in SUBORD)),
                float(in_quote),
                float(report_only),
                float(transfer),
                float(any(z in lo[max(0,ix-2):ix] for z in NEG)),
                float(any(z in local for z in NEG)),
                float(passive),
                offsets[0],
                float(";" in before_text or "." in before_text),
                float(any(z in preceding for z in REAL)),
                float(any(z in prev for z in ("что","будто","якобы"))),
            )
            require(len(bits)==16 and all(-1e-8<=z<=1.+1e-8 for z in bits),
                    "S65 source-factuality features nonfinite/out of range")
            f[i,slot,20:]=torch.tensor(bits,dtype=torch.float32)
    require(f.shape==(b,K,FEATURE) and not f.requires_grad and
            bool(torch.isfinite(f).all()),"S65 source-only factual input not frozen")
    return S65Input(base,f.detach(),base.unsafe_row.detach().clone())

class S65FactualityJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.event=nn.Parameter(torch.zeros(FEATURE))
        self.event_bias=nn.Parameter(torch.zeros(1))
        self.none=nn.Parameter(torch.zeros(FEATURE))
        self.none_bias=nn.Parameter(torch.zeros(1))
        self.pair=nn.Parameter(torch.zeros(FEATURE))
        self.pair_bias=nn.Parameter(torch.zeros(1))
        self.polar=nn.Parameter(torch.zeros(FEATURE))
        self.polar_bias=nn.Parameter(torch.zeros(1))
        self.classifier=nn.Parameter(torch.zeros(3,FEATURE))
        self.classifier_bias=nn.Parameter(torch.zeros(3))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.streaming_causality_certified=False

    def forward(self,x:S65Input,old62:S62JointEventRecipient,
                old63:S63QuoteOuterSafe):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed
                and not self.streaming_causality_certified
                and old62.output_gate==0 and old63.output_gate==0
                and all(not p.requires_grad for m in (old62,old63)
                        for p in m.parameters())
                and not x.factual_source.requires_grad,
                "strict research-only frozen weights/gate")
        y,p=old63(x.older,old62)
        b,k,w1=y.shape;w=w1-1
        require(x.factual_source.shape==(b,K,FEATURE)
                and x.strict_native_or_quote_unsafe.shape==(b,),
                "S65 source only after original CPU native ABI")
        ev=x.factual_source@self.event+self.event_bias
        none=x.factual_source@self.none+self.none_bias
        pair=x.factual_source@self.pair+self.pair_bias
        pol=x.factual_source@self.polar+self.polar_bias
        classes=F.linear(x.factual_source,self.classifier,self.classifier_bias)
        # Logit 0=NONE, 1=ASSERTED transfer, 2=REJECTED transfer.
        # All are SOURCE-only predictions; no manual labels/teacher
        # forcing is needed anywhere in actual S65 inference.
        transfer=torch.logsumexp(classes[:,:,1:],dim=-1)
        target=(y[:,:,:w]+ev.unsqueeze(-1)+pair.unsqueeze(-1)+
                transfer.unsqueeze(-1)).masked_fill(
            ~x.older.original.target_safe,-1e9)
        null=y[:,:,-1]+ev+none+classes[:,:,0]
        out=torch.cat((target,null.unsqueeze(-1)),dim=-1)
        out=out.masked_fill(~x.older.original.allowed.unsqueeze(-1),-1e9)
        negative=p+pol.unsqueeze(-1)+(
            classes[:,:,2]-classes[:,:,1]).unsqueeze(-1)
        require(out.shape==(b,k,w1) and negative.shape==(b,k,w)
                and classes.shape==(b,k,3)
                and bool(torch.isfinite(out).all())
                and bool(torch.isfinite(negative).all()),
                "S65 original43M source×target logits finite/goldblind")
        return out,negative,classes

    def decode(self,x:S65Input,old62:S62JointEventRecipient,
               old63:S63QuoteOuterSafe):
        with torch.inference_mode():
            y,p,cls=self(x,old62,old63)
            b,k,w1=y.shape;w=w1-1
            joint=y.reshape(b,k*w1).argmax(-1)
            slot=torch.div(joint,w1,rounding_mode="floor")
            dest=torch.remainder(joint,w1)
            source=x.older.original.sources[torch.arange(b),slot]
            recipient=torch.zeros((b,w,w),dtype=torch.bool,device=y.device)
            rejected=torch.zeros_like(recipient)
            for i in range(b):
                if bool(x.strict_native_or_quote_unsafe[i]):continue
                target=int(dest[i]);src=int(source[i])
                if target==w:continue
                require(bool(x.older.original.target_safe[i,slot[i],target]),
                        "S65 non-native-safe word BPE or quote edge")
                recipient[i,src,target]=True
                rejected[i,src,target]=float(p[i,slot[i],target])>0
            require(int(recipient.sum())<=b and
                    not bool((rejected&~recipient).any()) and
                    not bool((recipient&x.strict_native_or_quote_unsafe[:,None,None]).any()),
                    "S65 FAIL closed original native invalid edge")
            return {
                "chosen_event":source,
                "directed_recipient":recipient,
                "directed_rejected":rejected,
                "strict_native_or_quote_abstentions":int(
                    x.strict_native_or_quote_unsafe.sum()),
                "output_gate":0,"candidate_accepted":False,
                "activation_allowed":False,
                "streaming_causality_certified":False,
            }

    def fp16(self):
        with torch.no_grad():
            z=torch.cat([p.detach().reshape(-1)
                         for p in self.parameters()]).half().contiguous()
            require(z.shape==(PAID,) and bool(torch.isfinite(z).all()),
                    "S65 saved FP16 model 259 parameters invalid")
            return z

    @classmethod
    def from_fp16(cls,z):
        require(z.shape==(PAID,) and z.dtype==torch.float16
                and bool(torch.isfinite(z).all()) and not z.requires_grad,
                "S65 original CPU frozen physical FP16 count/ABI")
        model=cls();i=0
        with torch.no_grad():
            for p in model.parameters():
                n=p.numel()
                p.copy_(z[i:i+n].float().reshape(p.shape))
                i+=n
        require(i==PAID and torch.equal(model.fp16(),z),
                "S65 physical FP16 read back changed")
        return model.eval().requires_grad_(False)

def fit_SPENT344_only(groups,old62,old63):
    """Ten? Actually 11 old groups: S53..S64 =24x5 +32x6 =312.

    Source-only frozen beam and older model predictions. Training labels
    enter the loss only; no S65 NEW32 source/gold is ever imported.
    """
    expected=[24]*5+[32]*7
    require(len(groups)==12 and [len(z[1]) for z in groups]==expected,
            "S65 strictly old344 true Calibri43M GT scored S53..S64")
    npos=nrej=ncovered=nunsafe=0;prepared=[]
    for x,event,rec,rejected in groups:
        n,k=x.older.original.sources.shape
        w=x.older.original.word_count
        require(event.shape==rec.shape==rejected.shape==(n,)
                and event.dtype==rec.dtype==torch.long
                and rejected.dtype==torch.bool
                and bool(((rec>=0)&(rec<=w)).all())
                and bool((~rejected|(rec!=w)).all()),
                "S65 old344 gold ABI source-only")
        slots=(x.older.original.sources==event[:,None])&x.older.original.allowed
        slot=slots.float().argmax(-1)
        possible=slots.any(-1)
        safe=possible&~x.strict_native_or_quote_unsafe
        has=(rec!=w)
        sel=torch.where(safe&has)[0]
        if len(sel):
            require(bool(x.older.original.target_safe[sel,slot[sel],rec[sel]].all()),
                    "old344 safe supervised positive transfer invalid")
        prepared.append((x,event,rec,rejected,slot,safe))
        npos+=int(has.sum());nrej+=int(rejected.sum())
        ncovered+=int(possible.sum());nunsafe+=int(x.strict_native_or_quote_unsafe.sum())
    require(npos>=170 and nrej>=75 and ncovered>=270 and nunsafe<=100,
            f"old344 source labels/beam coverage invalid positives={npos} rejected={nrej} covered={ncovered} unsafe={nunsafe}")
    torch.manual_seed(651118)
    model=S65FactualityJoint()
    old62.eval().requires_grad_(False)
    old63.eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.025)
    for step in range(260):
        opt.zero_grad(set_to_none=True)
        total=None
        for x,event,rec,rejected,slot,safe in prepared:
            rows=torch.where(safe)[0]
            if len(rows)==0:continue
            joint,polar,classes=model(x,old62,old63)
            b,k,w1=joint.shape;w=w1-1
            truth=torch.where(rec==w,torch.zeros_like(rec),
                              torch.where(rejected,torch.full_like(rec,2),
                                          torch.ones_like(rec)))
            class_loss=F.cross_entropy(classes[rows,slot[rows]],truth[rows])
            directed=slot[rows]*w1+rec[rows]
            pair_loss=F.cross_entropy(joint[rows].reshape(-1,k*w1),directed)
            event_loss=F.cross_entropy(joint[rows].logsumexp(-1),slot[rows])
            pos=torch.where(safe & (rec!=w))[0]
            pol_loss=F.binary_cross_entropy_with_logits(
                polar[pos,slot[pos],rec[pos]],rejected[pos].float()) if len(pos) else 0.
            loss=(1.65*pair_loss+.5*event_loss+.65*class_loss+
                  .95*pol_loss)*(len(rows)/344.)
            total=loss if total is None else total+loss
        require(total is not None,"S65 no genuine safe older source")
        total=total+.001*sum(p.square().sum() for p in model.parameters())
        require(bool(torch.isfinite(total)),"S65 source/model loss nonfinite")
        total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),
                "S65 multi-head event/factuality/reject must backprop")
        opt.step()
    frozen=S65FactualityJoint.from_fp16(model.fp16())
    return frozen,frozen.fp16(),{
        "old_SPENT_group_counts":expected,
        "old_312_SOURCE_rows":sum(expected),
        "old_gold_directed_recipient":npos,
        "old_gold_directed_REJECTED":nrej,
        "old_predicate_in_source_top8":ncovered,
        "causal_BPE_or_unclosed_quote_rows_EXCLUDED_from_training":nunsafe,
        "new_FP16_parameters":PAID,
        "new_S65_NEW32_raw_gold_UNSEEN":True,
        "all_parent_original43M_GT_and_readouts_frozen":True,
        "original_source_only_3way_NONE_ASSERTED_REJECTED_no_gold_in_inference":True,
        "loss_fixed_before_S65_first_NEW32":"AdamW260 steps lr.012 wd.025, joint1.65/event.5/class.65/neg.95/L2 .001",
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
