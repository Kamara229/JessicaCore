"""S63 research-only quotation/outer-event residual + strict BPE fail-closed.

Changes vs S62: a small *trained* source-only quote-scope prior on
event×recipient+NONE scores, with native S62 pair interactions unchanged.
Predicts event, target, negation jointly. Source-only raw quotes are NOT
gold and never permit oracle event selection. LATER OUTER transfer after
quoted transfer can overrule a speech-report verb; when only quoted
transfer is present, correct decision is report/no delivered recipient.

Additional strict-causal abstention on shared native-BPE words:
  unsafe any-word => NO directed edge, even if offline model would score it.
This does NOT make a full-context offline model streaming causal-safe.
S63 only fit from previously scored/spent S53..S62 true Calibri43M GT.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    Beam,S62JointEventRecipient,K,PAIR_D,
)
Q=20
JOINT_D=PAIR_D+Q
FP16=2*(Q+1)+(JOINT_D+1)+(Q+1) # event20+1, NONE20+1, pair407+1, rejection20+1
assert Q==20 and JOINT_D==407 and FP16==471
START={'«':'»','“':'”','„':'“','‘':'’','"':'"'}
END=frozenset(('»','”','“','’'))
REPORT_PREFIXES=('процит','повтор','зачит','прочит','произн',
                 'отмет','запис','озвуч','пересказ')
TRANSFER_PREFIXES=('переда','передал','передала','передали','переда́',
                   'вруч','выда','выдал','выдала','выдан','показ',
                   'отправ','направ','сообщ','предостав','достав',
                   'пересла','выслал','выслала','передан','передано')
COORD=frozenset(('и','но','затем','потом','после','однако','далее','вслед',
                 'этого','чего','также'))
NEG=frozenset(('не','ни'))

def require(ok,msg):
    if not bool(ok):raise ValueError("S63 original43M GT quote and BPE gate0: "+msg)

def raw_quote_spans(text):
    """Input text character offsets only. Explicitly track unmatched quotes.

    Paired Russian «», curly “”, „“, ‘’, ASCII ". No lexical role labels.
    """
    stack=[];spans=[];unmatched=False
    for i,ch in enumerate(text):
        if stack and ch==stack[-1][1]:
            start,_,_=stack.pop();spans.append((start,i))
        elif ch in START:
            stack.append((i,START[ch],ch))
        elif ch in END:
            unmatched=True
    return tuple(sorted(spans)),bool(stack) or unmatched

@dataclass(frozen=True)
class S63Overlay:
    original:Beam
    source_quote:torch.Tensor  # B,K,20 source-only
    pair_quote:torch.Tensor    # B,K,W,407 frozen original pair387 + quote20
    unsafe_row:torch.Tensor    # B booleans, shared native BPE/any quote mismatch

def quote_overlay(
    beam:Beam,texts:tuple[str,...],native_unsafe:torch.Tensor,
)->S63Overlay:
    b,k=beam.sources.shape;w=beam.word_count
    require(k==K and len(texts)==b and
            native_unsafe.shape==(b,w) and native_unsafe.dtype==torch.bool
            and not beam.source_fused.requires_grad and
            not beam.pair_feature.requires_grad,
            "original 43MGT beam, raw and unsafe source ABI")
    flags=torch.zeros((b,K,Q),dtype=torch.float32)
    failed=native_unsafe.any(-1).detach().clone()
    for i,text in enumerate(texts):
        words=raw_surface_grammar(text)
        require(3<=len(words)<=w,"new source raw text/native word mismatch")
        spans,unbalanced=raw_quote_spans(text)
        failed[i]=bool(failed[i]) or unbalanced
        quoted=[any(a<t.start and t.end<=z for a,z in spans) for t in words]
        last_end=max((z for a,z in spans),default=-1)
        first_start=min((a for a,z in spans),default=len(text)+1)
        def is_report(t):return t.form.casefold().startswith(REPORT_PREFIXES)
        def is_transfer(t):return t.form.casefold().startswith(TRANSFER_PREFIXES)
        report_before=any(t.start<first_start and is_report(t) for t in words)
        later_transfer=[j for j,t in enumerate(words)
                        if t.start>last_end and last_end>=0 and not quoted[j]
                        and is_transfer(t)]
        outer_transfer=[j for j,t in enumerate(words)
                        if not quoted[j] and is_transfer(t)]
        outer_verb=[j for j,t in enumerate(words)
                    if not quoted[j] and
                    bool(t.features[12] or t.features[13] or t.features[14])]
        after_has=bool(later_transfer)
        for kk in range(K):
            source=int(beam.sources[i,kk])
            require(source<len(words),"beam candidate outside original raw tokens")
            t=words[source]
            lower=t.form.casefold()
            prior=[z.form.casefold() for z in words[max(0,source-4):source]]
            after=last_end>=0 and t.start>last_end
            before=bool(spans) and t.start<first_start
            report=is_report(t);transfer=is_transfer(t)
            features=(
                float(bool(spans)),
                float(quoted[source]),
                float(before),
                float(after),
                float(report),
                float(report and before),
                float(transfer and after),
                float(after_has),
                float(before and after_has),
                float(after and report_before),
                float(bool(outer_transfer) and source==outer_transfer[-1]),
                float(bool(outer_verb) and source==outer_verb[-1]),
                float(transfer and not quoted[source]),
                float(any(z in COORD for z in prior)),
                source/max(1,len(words)-1),
                max(-1.,min(1.,(t.start-last_end)/max(1,len(text))))
                if bool(spans) else 0.,
                float(any(z in NEG for z in prior)),
                float(report and not after_has),
                min(3,len(spans))/3.,
                float(any(is_transfer(words[j]) for j,q in enumerate(quoted) if q)),
            )
            require(len(features)==Q,"S63 quote-only feature layout drift")
            flags[i,kk]=torch.tensor(features,dtype=torch.float32)
    pair=torch.cat((beam.pair_feature,flags.unsqueeze(2).expand(-1,-1,w,-1)),
                   dim=-1).detach()
    require(pair.shape==(b,K,w,JOINT_D)
            and bool(torch.isfinite(pair).all())
            and not pair.requires_grad,
            "S63 quoted event×target source-only readout invalid")
    return S63Overlay(beam,flags.detach(),pair,failed.detach())

class S63QuoteOuterSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.event=nn.Parameter(torch.zeros(Q))
        self.event_bias=nn.Parameter(torch.zeros(1))
        self.none=nn.Parameter(torch.zeros(Q))
        self.none_bias=nn.Parameter(torch.zeros(1))
        self.pair=nn.Parameter(torch.zeros(JOINT_D))
        self.pair_bias=nn.Parameter(torch.zeros(1))
        self.reject=nn.Parameter(torch.zeros(Q))
        self.reject_bias=nn.Parameter(torch.zeros(1))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.streaming_causality_certified=False

    def forward(self,x:S63Overlay,old62:S62JointEventRecipient):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed
                and not self.streaming_causality_certified
                and old62.output_gate==0
                and all(not p.requires_grad for p in old62.parameters())
                and not x.source_quote.requires_grad and
                not x.pair_quote.requires_grad,
                "S63 fail closed research overlay / old S62 immutable")
        base,old_polar=old62(x.original)
        b,k,w1=base.shape;w=w1-1
        source_delta=x.source_quote@self.event+self.event_bias
        pair_delta=x.pair_quote@self.pair+self.pair_bias
        null_delta=x.source_quote@self.none+self.none_bias
        pair=base[:,:,:w]+pair_delta+source_delta.unsqueeze(-1)
        null=base[:,:,-1]+null_delta+source_delta
        out=torch.cat((pair,null.unsqueeze(-1)),dim=-1)
        out=out.masked_fill(~x.original.target_safe.new_ones(
            (b,k,w+1),dtype=torch.bool),float("-inf")) if False else out
        out[:,:,:w]=out[:,:,:w].masked_fill(
            ~x.original.target_safe,-1e9)
        out=out.masked_fill(~x.original.allowed.unsqueeze(-1),-1e9)
        negative=old_polar+x.source_quote@self.reject[...,None] if False else old_polar+(x.source_quote@self.reject+self.reject_bias).unsqueeze(-1)
        require(out.shape==(b,k,w1) and negative.shape==(b,k,w)
                and bool(torch.isfinite(out).all())
                and bool(torch.isfinite(negative).all()),
                "S63 source-only quoted candidate finite shape")
        return out,negative

    def decode(self,x:S63Overlay,old62:S62JointEventRecipient,
               *,strict_native_safety:bool=True):
        with torch.inference_mode():
            y,n=self(x,old62)
            b,k,w1=y.shape;w=w1-1
            ix=y.reshape(b,k*w1).argmax(-1)
            kk=torch.div(ix,w1,rounding_mode="floor")
            dst=torch.remainder(ix,w1)
            src=x.original.sources[torch.arange(b),kk]
            edges=torch.zeros((b,w,w),dtype=torch.bool)
            rejected=torch.zeros_like(edges)
            for i in range(b):
                if strict_native_safety and bool(x.unsafe_row[i]):
                    continue
                z=int(dst[i]);p=int(src[i])
                if z==w:continue
                require(bool(x.original.target_safe[i,kk[i],z]),
                        "S63 original quote and native BPE unsafe target")
                edges[i,p,z]=True
                rejected[i,p,z]=float(n[i,kk[i],z])>0
            require(int(edges.sum())<=b and
                    not bool((rejected&~edges).any()),
                    "S63 one globally safe directed edge per original sentence")
            return {"chosen_event":src,
                    "directed_recipient":edges,
                    "directed_rejected":rejected,
                    "strict_native_safety":strict_native_safety,
                    "native_ambiguous_abstentions":int(
                         x.unsafe_row.sum()) if strict_native_safety else 0,
                    "no_target_abstentions":int((dst==w).sum()),
                    "output_gate":0,"candidate_accepted":False,
                    "activation_allowed":False,
                    "streaming_causality_certified":False}

    def fp16(self):
        with torch.no_grad():
            z=torch.cat([p.detach().reshape(-1)
                         for p in self.parameters()]).half().contiguous()
            require(z.shape==(FP16,) and bool(torch.isfinite(z).all()),
                    "S63 new FP16 physical readout incorrect")
            return z

    @classmethod
    def from_fp16(cls,z):
        require(z.dtype==torch.float16 and z.shape==(FP16,)
                and bool(torch.isfinite(z).all()) and not z.requires_grad,
                "S63 physical FP16 shape or safety corrupt")
        m=cls();j=0
        with torch.no_grad():
            for p in m.parameters():
                n=p.numel()
                p.copy_(z[j:j+n].float().reshape(p.shape))
                j+=n
        require(j==FP16,"physical cursor")
        m.eval().requires_grad_(False)
        require(torch.equal(m.fp16(),z),"S63 FP16 persisted drift")
        return m

def fit_PREVIOUSLY_SPENT280_only(groups,old62):
    """No new S63 HELD, only 9 S53-S62 previously scored groups=280 rows."""
    require(len(groups)==10 and [int(x[0].original.sources.shape[0])
                for x in groups]==[24,24,24,24,24,32,32,32,32,32],
            "S63 exactly ten OLD groups S53..S62 SPENT280")
    prepared=[];coverage=total=unsafe_skips=0
    for x,ev,target,rejected in groups:
        b,k=x.original.sources.shape;w=x.original.word_count
        require(ev.shape==target.shape==rejected.shape==(b,)
                and ev.dtype==target.dtype==torch.long
                and rejected.dtype==torch.bool and
                bool(((target>=0)&(target<=w)).all()),
                "S63 training only old annotated event/recipient ABI")
        slot=(x.original.sources==ev[:,None])&x.original.allowed
        any_target=slot.any(-1)
        ok=any_target&~x.unsafe_row
        if bool((ok & (target!=w)).any()):
            a=torch.where(ok & (target!=w))[0]
            sel=slot.float().argmax(-1)
            require(bool(x.original.target_safe[a,sel[a],target[a]].all()),
                    "S63 old supervised recipient pair unsafe")
        prepared.append((x,ev,target,rejected,ok,slot.float().argmax(-1)))
        coverage+=int(any_target.sum())
        unsafe_skips+=int(x.unsafe_row.sum())
        total+=b
    require(total==280 and coverage>=200 and unsafe_skips<=100,
            f"S63 old280 source-only beam/unsafe validation: coverage={coverage} unsafe={unsafe_skips}")
    torch.manual_seed(631118)
    model=S63QuoteOuterSafe()
    opt=torch.optim.AdamW(model.parameters(),lr=.016,weight_decay=.02)
    old62.eval().requires_grad_(False)
    for step in range(260):
        opt.zero_grad(set_to_none=True)
        loss=None
        for x,ev,target,rejected,ok,slot in prepared:
            rows=torch.where(ok)[0]
            if len(rows)==0: continue
            y,p=model(x,old62)
            b,k,w1=y.shape
            choice=slot[rows]*w1+target[rows]
            ce=F.cross_entropy(y[rows].reshape(-1,k*w1),choice)
            evce=F.cross_entropy(y[rows].logsumexp(-1),slot[rows])
            pos=torch.where(ok&(target!=w1-1))[0]
            pol=F.binary_cross_entropy_with_logits(
                p[pos,slot[pos],target[pos]],rejected[pos].float()) if len(pos) else 0.
            part=(1.55*ce+.45*evce+.75*pol)*(len(rows)/280.)
            loss=part if loss is None else loss+part
        require(loss is not None,"S63 old280 training source unsafe all")
        loss=loss+.0012*sum(p.square().sum() for p in model.parameters())
        require(bool(torch.isfinite(loss)),"S63 nonfinite source-only training")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),"S63 missing event/target/reject gradient")
        opt.step()
    frozen=S63QuoteOuterSafe.from_fp16(model.fp16())
    return frozen,frozen.fp16(),{
        "previously_SPENT_true_original43M_GT_groups":[24,24,24,24,24,32,32,32,32,32],
        "old_source_rows":total,
        "gold_main_event_in_frozen_S59_top8":coverage,
        "old_causal_unsafe_rows_excluded_from_new_adapter_loss":unsafe_skips,
        "new_F16_parameters":FP16,
        "S62_older_joint_base_frozen":True,
        "zero_predicate_or_recipient_gold_in_forward":True,
        "new_S63_HELD32_raw_gold_unseen":True,
        "loss":"AdamW260 lr.016 joint1.55 event.45 polarity.75 regularize.0012",
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
