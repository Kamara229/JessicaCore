"""S57: genuinely event-first staged clause/semantic Russian graph research head.

Prior S56 independent structure HELD observed 36 false RECIPIENT and
23 false REJECTED edges. S57 NEVER tunes against that already inspected
HELD as HELD again: it is now explicitly OLD TRAIN, together with
S53 S54 S55 inspected. NEW S57 HELD24 author dataset was precommitted
before this file; never imported here.

Stage 1 chooses EXACTLY ONE outer predicate per sentence, learned from
input-only raw morphology, clause scope, actual frozen native Calibri43M
post-GT 320D and old paid S26 semantic logits. Stage 2 separately
decides whether this predicate actually transfers/communicates an
object to a recipient. Stage 3 chooses AT MOST ONE eligible recipient
for that chosen event using genuine43M/S26+S56 pair54 features. Stage
4 assigns its clause-conditioned REJECTED polarity. If no event/
recipient is eligible, decoder ABSTAINS. This construction caps directed
edges per sentence to 1 (rather than emitting every positive pair).

The candidate is a small independent offline FP16 readout, never injected
into original Calibri43M/G.T. and not proven autoregressive causal:
original 43M full-prefill and old S26 contexts are bidirectional.
All model weights, thresholds, training steps are fixed BEFORE first
new S57 HELD24. Original old J7 protected separately.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    FEATURES,raw_surface_grammar,
)
from research.nra3_s19.s19f_real43m_train9_graph_targets import STATUS,ROLE
from research.nra3_s20.s56_grammar_clause_scope_semantic_readout import (
    FEATURE_NAMES as PAIR_NAMES,real43m_semantic_structure_features,
    grammar_scope_metadata,PREPOSITIONS,NEG,
)
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    SafeGraphReadOnly,
)

EVENT_META=(
 "relative_word_index",
 "relative_clause_index",
 "clause_length_norm",
 "all_clause_count_norm",
 "negative_immediately_precedes",
 "negative_within_three_to_left",
 "negative_within_two_to_right",
 "negative_any_same_clause",
 "negative_in_different_clause",
 "other_verb_candidates_same_clause",
 "quoted_content_fraction_in_clause",
 "true43M_GT_word_to_sentence_mean_cosine",
 "S26_probability_some_event",
 "S26_max_target_RECIPIENT_probability",
 "S26_max_target_REJECTED_probability",
 "S26_max_target_AGENT_probability",
 "S26_max_target_event_relation",
)
EVENT_NAMES=tuple(FEATURES)+tuple("S26_P_"+x for x in STATUS)+EVENT_META
EVENT_DIM=len(EVENT_NAMES)
PAIR_DIM=len(PAIR_NAMES)
N_PAID=EVENT_DIM*3+PAIR_DIM*2
assert len(EVENT_META)==17 and EVENT_DIM==41 and PAIR_DIM==54 and N_PAID==231

def require(ok,reason):
    if not bool(ok):raise ValueError("S57 event-first genuine43M GT research head: "+reason)

@dataclass(frozen=True)
class EventFirstInputs:
    event:torch.Tensor
    pair:torch.Tensor
    candidate_event:torch.Tensor
    valid_pair:torch.Tensor


def gold_blind_eventfirst_features(
    x339:torch.Tensor, obs:SafeGraphReadOnly, texts:tuple[str,...]
)->EventFirstInputs:
    b,w,_=x339.shape
    require(x339.shape==(b,w,339)
            and len(texts)==b and x339.dtype==torch.float32
            and not x339.requires_grad
            and obs.native_word_mask.shape==(b,w),
            "S57 actual original 43M+GT frozen native source and text mismatch")
    pair=real43m_semantic_structure_features(x339,obs,texts)
    require(pair.shape==(b,w,w,PAIR_DIM),"old S56 54D original GT ABI changed")
    old=obs.original_paid_graph_logits
    event_prob=torch.softmax(old.event_status.detach(),dim=-1)
    role_prob=torch.softmax(old.role.detach(),dim=-1)
    polar_prob=torch.softmax(old.argument_polarity.detach(),dim=-1)
    relation_prob=torch.softmax(old.event_relation.detach(),dim=-1)
    full=torch.zeros((b,w,EVENT_DIM),dtype=torch.float32)
    full[...,:18]=x339[...,320:338]
    full[...,18:18+len(STATUS)]=event_prob
    event_elig=obs.eligible_word_mask.clone()
    for bi,text in enumerate(texts):
        words,cids,groups,negs,verbs,_=grammar_scope_metadata(text)
        n=int(obs.native_word_mask[bi].sum())
        require(n==len(words) and n>=3,
                "S57 original source native token alignment/character source drift")
        avg=x339[bi,:n,:320].float().mean(dim=0)
        avg=F.normalize(avg,dim=0,eps=1e-9)
        max_clause=max(cids)
        for i,t in enumerate(words):
            # Raw negation markers and prepositions cannot serve as the
            # single outer predicate. No gold predicate lemma is used.
            if t.form.casefold() in NEG or t.form.casefold() in PREPOSITIONS:
                event_elig[bi,i]=False
            ci=cids[i]
            left=[i-k for k in negs[ci] if k<i]
            right=[k-i for k in negs[ci] if k>i]
            same=groups[ci]
            mean_sim=float(F.cosine_similarity(
                x339[bi,i,:320].float().unsqueeze(0),
                avg.unsqueeze(0),dim=-1))
            v=(
                i/max(1,n-1), ci/max(1,max_clause),
                min(len(same),24)/24.,
                min(max_clause+1,6)/6.,
                int(1 in left),
                int(any(1<=z<=3 for z in left)),
                int(any(1<=z<=2 for z in right)),
                int(bool(negs[ci])),
                int(any(z for c,z in negs.items() if c!=ci)),
                min(len(verbs[ci])-int(i in verbs[ci]),6)/6.,
                min(int(x339[bi,same,338].sum()),8)/8.,
                mean_sim,
                1-float(event_prob[bi,i,STATUS.index("NONE")]),
                float(role_prob[bi,i,:n,ROLE.index("RECIPIENT")].max()),
                float(polar_prob[bi,i,:n,2].max()),
                float(role_prob[bi,i,:n,ROLE.index("AGENT")].max()),
                float((1-relation_prob[bi,i,:n,0]).max()),
            )
            require(len(v)==len(EVENT_META) and all(
                float('-inf')<float(z)<float('inf') for z in v),
                "S57 new event source feature numerical failure")
            full[bi,i,18+len(STATUS):]=torch.tensor(v,dtype=torch.float32)
    require(bool(event_elig.any(-1).all())
            and not bool((event_elig & ~obs.native_word_mask).any())
            and not bool((event_elig & obs.quote_word_mask).any()),
            "S57 event selectors must exclude punct/quote/BPE but leave candidates")
    return EventFirstInputs(
        event=full.detach(),pair=pair.detach(),
        candidate_event=event_elig.detach(),
        valid_pair=obs.eligible_directed_pair_mask.detach())


class S57EventFirstRecipient(nn.Module):
    def __init__(self):
        super().__init__()
        self.event=nn.Parameter(torch.zeros(EVENT_DIM))
        self.transfers=nn.Parameter(torch.zeros(EVENT_DIM))
        self.recipient=nn.Parameter(torch.zeros(PAIR_DIM))
        self.rejected=nn.Parameter(torch.zeros(EVENT_DIM+PAIR_DIM))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.streaming_causality_certified=False

    def _require(self,data:EventFirstInputs):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed
                and not self.streaming_causality_certified,
                "S57 model cannot activate")
        ev,pair,mask,valid=data.event,data.pair,data.candidate_event,data.valid_pair
        require(ev.ndim==3 and ev.shape[-1]==EVENT_DIM
                and pair.shape[:2]==ev.shape[:2]
                and pair.shape[2]==ev.shape[1]
                and pair.shape[-1]==PAIR_DIM
                and mask.shape==ev.shape[:2]
                and valid.shape==pair.shape[:3]
                and mask.dtype==valid.dtype==torch.bool
                and bool(mask.any(-1).all())
                and bool(torch.isfinite(ev).all())
                and bool(torch.isfinite(pair).all()),
                "S57 actual original GT frozen event+pair ABI drift")

    def forward(self,data:EventFirstInputs):
        self._require(data)
        ev=data.event.matmul(self.event)
        transfers=data.event.matmul(self.transfers)
        recipient=data.pair.matmul(self.recipient)
        b,w,d=data.event.shape
        repeated=data.event.unsqueeze(2).expand(b,w,w,d)
        joint=torch.cat((repeated,data.pair),dim=-1)
        rejected=joint.matmul(self.rejected)
        return ev,transfers,recipient,rejected

    def decode(self,data:EventFirstInputs):
        """No oracle event, at most one directed link per source sentence."""
        with torch.inference_mode():
            e,t,r,n=self(data)
            e=e.masked_fill(~data.candidate_event,-1e9)
            chosen=e.argmax(-1)
            b,w=data.candidate_event.shape
            recipient=torch.zeros((b,w,w),dtype=torch.bool)
            reject=torch.zeros_like(recipient)
            has_transfer=torch.zeros((b,),dtype=torch.bool)
            abstained_no_target=0
            for i in range(b):
                src=int(chosen[i])
                if not bool(t[i,src]>0):
                    continue
                allowed=data.valid_pair[i,src]
                if not bool(allowed.any()):
                    abstained_no_target+=1
                    continue
                dest=int(r[i,src].masked_fill(~allowed,-1e9).argmax())
                recipient[i,src,dest]=True
                reject[i,src,dest]=bool(n[i,src,dest]>0)
                has_transfer[i]=True
            require(int(recipient.sum())<=b
                    and int(reject.sum())<=int(recipient.sum())
                    and not bool((recipient&~data.valid_pair).any()),
                    "S57 first-event decoder emitted multiple/unsafe candidate links")
            return {
                "chosen_outer_event_index":chosen,
                "predicted_has_eligible_recipient":has_transfer,
                "directed_recipient":recipient,
                "directed_rejected":reject,
                "unavailable_native_pair_abstentions":abstained_no_target,
            }

    def physical_fp16(self):
        with torch.no_grad():
            w=torch.cat((self.event.detach(),self.transfers.detach(),
                         self.recipient.detach(),self.rejected.detach()))
            require(w.numel()==N_PAID and bool(torch.isfinite(w).all()),
                    "S57 physical paid event-first 231 FP16 unfitted")
            return w.half().contiguous()

    @classmethod
    def from_physical_fp16(cls,weights:torch.Tensor):
        require(weights.dtype==torch.float16 and weights.shape==(N_PAID,)
                and not weights.requires_grad and bool(torch.isfinite(weights).all()),
                "S57 old event-first paid231 FP16 missing")
        obj=cls().eval()
        at=0
        with torch.no_grad():
            for param in (obj.event,obj.transfers,obj.recipient,obj.rejected):
                n=param.numel()
                param.copy_(weights[at:at+n].float())
                at+=n
        obj.requires_grad_(False)
        require(torch.equal(obj.physical_fp16(),weights),
                "S57 paid 231 FP16 physical weight roundtrip drift")
        return obj


def train_fixed_on_previously_inspected_96_only(
    cases:list[tuple[EventFirstInputs,torch.Tensor,
                    torch.Tensor,torch.Tensor,torch.Tensor]],
):
    """Crossentropy outer event, BCE transfer, CE recipient, BCE rejection.

    Gold indices used ONLY for supervised training loss on old/spent
    data, never model forward or new HELD evaluation.
    """
    require(len(cases)==4,"S57 only spent S53/S54/S55/S56 old96")
    pos=neg=0
    for data,event_idx,recipient_truth,rejected_truth,has_rec in cases:
        b,w=data.candidate_event.shape
        require(b==24 and event_idx.shape==(b,)
                and recipient_truth.shape==rejected_truth.shape==data.valid_pair.shape
                and has_rec.shape==(b,) and has_rec.dtype==torch.bool
                and recipient_truth.dtype==rejected_truth.dtype==torch.bool
                and bool(data.candidate_event[torch.arange(b),event_idx].all())
                and not bool((recipient_truth&~data.valid_pair).any())
                and not bool((rejected_truth&~recipient_truth).any())
                and torch.equal(recipient_truth.sum((1,2))>0,has_rec),
                "S57 train outer-event supervision not source-aligned")
        pos+=int(has_rec.sum());neg+=int(rejected_truth.sum())
    require(pos==60 and neg==36,"S57 exact previous 96 manually inspected RECIPIENT/REJECTED data must be 60/36")
    torch.manual_seed(571118)
    obj=S57EventFirstRecipient()
    optimizer=torch.optim.Adam(obj.parameters(),lr=.035)
    for step in range(450):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for data,event_idx,recipient_truth,rejected_truth,has_rec in cases:
            b,w=data.candidate_event.shape
            i=torch.arange(b)
            el,tl,rl,nl=obj(data)
            event_loss=F.cross_entropy(
                el.masked_fill(~data.candidate_event,-1e9),event_idx)
            trans_loss=F.binary_cross_entropy_with_logits(
                tl[i,event_idx],has_rec.float(),
                pos_weight=torch.tensor(.60))
            source_rl=rl[i,event_idx]
            with_rec=torch.where(has_rec)[0]
            # Select the target by explicit gold source/target ONLY in
            # OLD TRAIN supervised loss. Decoder never sees event gold.
            targets=recipient_truth[with_rec,event_idx[with_rec]].float().argmax(-1)
            pr=source_rl[with_rec]
            pm=data.valid_pair[with_rec,event_idx[with_rec]]
            require(bool(pm[torch.arange(len(with_rec)),targets].all()),
                    "S57 teacher-forced true pair masked")
            rec_loss=F.cross_entropy(pr.masked_fill(~pm,-1e9),targets)
            rejected_gold=rejected_truth[with_rec,event_idx[with_rec],targets].float()
            reject_loss=F.binary_cross_entropy_with_logits(
                nl[with_rec,event_idx[with_rec],targets],rejected_gold,
                pos_weight=torch.tensor(.65))
            losses.append(event_loss*.90+trans_loss*.95+rec_loss+reject_loss*.65)
        total=sum(losses)/len(losses)+.0007*sum(p.square().sum() for p in obj.parameters())
        require(bool(torch.isfinite(total)), "S57 fixed TRAIN loss not finite")
        total.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in obj.parameters()),"S57 fixed TRAIN gradient missing")
        optimizer.step()
    obj.eval().requires_grad_(False)
    physical=obj.physical_fp16()
    restored=S57EventFirstRecipient.from_physical_fp16(physical)
    old_predictions=[restored.decode(data) for data,_,_,_,_ in cases]
    return restored,physical,{
        "training_cases":96,
        "old_true_directed_recipient_edges":pos,
        "old_true_directed_rejected_edges":neg,
        "fixed_optimizer":"Adam lr=.035 450 steps, L2 .0007",
        "candidate_event_selector":"masked_argmax_one_nonpunct_nonquote",
        "recipient_decoder":"BCE_has_recipient_then_at_most_one_argmax",
        "new_FP16_parameters":N_PAID,
        "old_train_result_links":[int(z["directed_recipient"].sum()) for z in old_predictions],
        "old_train_result_rejected":[int(z["directed_rejected"].sum()) for z in old_predictions],
        "S57_HELD24_text_and_gold_never_loaded":True,
        "old_source_96_training_fit_NOT_independent_EVAL":True,
        "full_prefill_not_causally_safe_for_online_decoder":True,
    }
