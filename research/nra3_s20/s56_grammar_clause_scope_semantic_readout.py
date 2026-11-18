"""S56 two-stage grammar/clause-scope + semantic recipient readout (research).

TRUE frozen Calibri43M+GT layer8 vectors and old physically trained S26
five-head context are fed through an UNCHANGED frozen bridge. S56 extends
S54 pair18 with text-only structural clause, passive and negation scope,
preposition/dative, and 320D original native word cosine features.

No manually annotated predicate, recipient or human gold enters the
model forward; gold is consumed only by S53/S54/S55 already-spent
TRAIN batches in a separate script. S56 HELD24 is first-read once,
NEVER fit or select on S56 HELD labels. Neither original 43M/G.T.
nor S26 or previous S54 are updated. This is full-context offline,
NOT a streaming-causal online language model adapter.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    raw_surface_grammar,FEATURES,
)
from research.nra3_s19.s19f_real43m_train9_graph_targets import STATUS,ROLE
from research.nra3_s20.s54_trainonly_real43m_recipient_residual import (
    FEATURE_NAMES as S54_BASE_NAMES,pair_features_gold_blind,
)
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    SafeGraphReadOnly,
)

EXTRA_NAMES=(
    "same_clause","clause_distance_clip2",
    "source_first_clause","source_final_clause",
    "target_first_clause","target_final_clause",
    "negative_any_in_source_clause","negative_left_immediately",
    "negative_within_left_four_same_clause",
    "negative_other_clause_exists",
    "nearest_negative_left_within_clause_inverse",
    "nearest_negative_right_within_clause_inverse",
    "source_clause_past_byt_aux","source_passive_suffix",
    "target_preceded_preposition_other","target_preceded_ot",
    "target_preceded_preposition_any","target_preceded_k_to_dlya",
    "target_first_lexical_in_clause","source_target_clause_cross",
    "other_verb_candidates_between","prepositions_between",
    "punctuation_between","negatives_between",
    "source_clause_dative_candidate_count",
    "source_clause_verb_candidate_count",
    "old_S26_P_NONE_directed",
    "old_S26_P_relation_notNONE",
    "old_S26_P_AGENCY_reverse",
    "old_S26_P_polarity_POSITIVE",
    "native_GT_source_target_cosine",
    "native_GT_source_target_l2_normalized",
    "old_S26_P_ASSERTED_source",
    "old_S26_P_event_at_target",
    "source_clause_length_normalized",
    "quoted_words_in_source_clause_normalized",
)
FEATURE_NAMES=S54_BASE_NAMES+EXTRA_NAMES
assert len(EXTRA_NAMES)==36 and len(FEATURE_NAMES)==54
PAID_PARAM_COUNT=2*len(FEATURE_NAMES)
GATE=0
PREPOSITIONS=frozenset(("к","ко","для","по","до","на","у","в","с","из","от","об","о","при","под"))
DAT_PREPOSITIONS=frozenset(("к","ко","для"))
BOUNDARIES=frozenset((",", ";",":","—","–"))
NEG=frozenset(("не","ни"))
PUNCT_BIT=FEATURES.index("is_punctuation")
DAT_BIT=FEATURES.index("ending_dative_possible")
VERB_BITS=(FEATURES.index("past_verb_shape_possible"),
           FEATURES.index("finite_future_verb_shape_possible"),
           FEATURES.index("short_passive_participle_shape_possible"))


def require(ok,why):
    if not bool(ok):raise ValueError("S56 genuine43M+GT structure and negation scope: "+why)


def grammar_scope_metadata(text:str):
    """Language-surface-only per-word clause/negative flags, no gold."""
    words=raw_surface_grammar(text)
    n=len(words)
    require(3<=n<=40,"S56 S18 source has unsupported 3..40 word span")
    cid=[]
    current=0
    for t in words:
        cid.append(current)
        if t.form in BOUNDARIES:current+=1
    per_clause={}
    for i,c in enumerate(cid):
        per_clause.setdefault(c,[]).append(i)
    neg_idx={c:tuple(i for i in idx if words[i].form.casefold() in NEG)
             for c,idx in per_clause.items()}
    verb_idx={c:tuple(i for i in idx
                      if any(words[i].features[x] for x in VERB_BITS))
              for c,idx in per_clause.items()}
    dat_idx={c:tuple(i for i in idx
                     if words[i].features[DAT_BIT] and not words[i].features[PUNCT_BIT])
             for c,idx in per_clause.items()}
    return words,tuple(cid),per_clause,neg_idx,verb_idx,dat_idx


def real43m_semantic_structure_features(
    x339:torch.Tensor,obs:SafeGraphReadOnly,texts:tuple[str,...],
):
    """Exactly 54D: 18 old real43M/S26 plus 36 new read-only grammar features."""
    b,w,_=x339.shape
    require(x339.shape==(b,w,339)
            and b==len(texts)
            and obs.eligible_directed_pair_mask.shape==(b,w,w)
            and x339.dtype==torch.float32
            and not x339.requires_grad,"S56 needs true old original43M frozen word339")
    base=pair_features_gold_blind(x339,obs)
    require(base.shape==(b,w,w,18)
            and bool(torch.isfinite(base).all()),
            "S54 old 18D neural interface drift")
    old=obs.original_paid_graph_logits
    roles=torch.softmax(old.role.detach(),dim=-1)
    polar=torch.softmax(old.argument_polarity.detach(),dim=-1)
    relation=torch.softmax(old.event_relation.detach(),dim=-1)
    status=torch.softmax(old.event_status.detach(),dim=-1)
    true43m=x339[...,:320].detach()
    vec=F.normalize(true43m.float(),dim=-1,eps=1e-8)
    cosine=(vec@vec.transpose(1,2)).clamp(-1,1)
    diff=true43m.unsqueeze(2)-true43m.unsqueeze(1)
    L2=torch.linalg.vector_norm(diff,dim=-1).clamp(max=20.)/20.
    extra=torch.zeros((b,w,w,len(EXTRA_NAMES)),dtype=torch.float32)
    for batch,text in enumerate(texts):
        words,cids,groups,negs,verbs,datives=grammar_scope_metadata(text)
        n=int(obs.native_word_mask[batch].sum())
        require(n==len(words),"S56 original native word/Unicode raw text mismatch")
        max_clause=max(cids)
        neg_total=sum(len(z) for z in negs.values())
        for i in range(n):
            for j in range(n):
                ci,cj=cids[i],cids[j]
                same=int(ci==cj)
                lower=words[j-1].form.casefold() if j else ""
                punct_between=sum(words[k].features[PUNCT_BIT]
                                  for k in range(min(i,j)+1,max(i,j)))
                neg_between=sum(words[k].form.casefold() in NEG
                                for k in range(min(i,j)+1,max(i,j)))
                other_verbs=sum(
                    any(words[k].features[x] for x in VERB_BITS)
                    for k in range(min(i,j)+1,max(i,j)))
                preps_between=sum(words[k].form.casefold() in PREPOSITIONS
                                  for k in range(min(i,j)+1,max(i,j)))
                left=[i-k for k in negs[ci] if k<i]
                right=[k-i for k in negs[ci] if k>i]
                vals=(
                  same,min(abs(ci-cj),2)/2.,
                  int(ci==0),int(ci==max_clause),
                  int(cj==0),int(cj==max_clause),
                  int(bool(negs[ci])),
                  int(1 in left),
                  int(any(1<=d<=4 for d in left)),
                  int(neg_total>len(negs[ci])),
                  1/min(left) if left else 0.,
                  1/min(right) if right else 0.,
                  int(any(words[k].form.casefold() in ("был","была","было","были")
                          for k in groups[ci])),
                  int(bool(words[i].features[FEATURES.index(
                      "short_passive_participle_shape_possible")])),
                  int(bool(words[j].features[FEATURES.index("preposition_other")]
                           and words[j].form.casefold() in PREPOSITIONS)),
                  int(lower=="от"),
                  int(lower in PREPOSITIONS),
                  int(lower in DAT_PREPOSITIONS),
                  int(j==next((k for k in groups[cj]
                               if not words[k].features[PUNCT_BIT]),-1)),
                  int(not same),
                  min(other_verbs,3)/3.,
                  min(preps_between,3)/3.,
                  min(punct_between,3)/3.,
                  min(neg_between,3)/3.,
                  min(len(datives[ci]),5)/5.,
                  min(len(verbs[ci]),5)/5.,
                  float(roles[batch,i,j,ROLE.index("NONE")]),
                  float(1-relation[batch,i,j,0]),
                  float(roles[batch,j,i,ROLE.index("AGENT")]),
                  float(polar[batch,i,j,1]),
                  float(cosine[batch,i,j]),
                  float(L2[batch,i,j]),
                  float(status[batch,i,STATUS.index("ASSERTED")]),
                  float(1-status[batch,j,STATUS.index("NONE")]),
                  min(len(groups[ci]),24)/24.,
                  min(int(x339[batch,groups[ci],338].sum()),8)/8.,
                )
                require(len(vals)==len(EXTRA_NAMES),"S56 structural dimension changes")
                extra[batch,i,j,:]=torch.tensor(vals,dtype=torch.float32)
    features=torch.cat((base,extra),dim=-1).detach()
    require(features.shape==(b,w,w,len(FEATURE_NAMES))
            and not features.requires_grad
            and bool(torch.isfinite(features).all())
            and not bool((obs.eligible_directed_pair_mask &
                     (obs.quote_word_mask.unsqueeze(1) |
                      obs.quote_word_mask.unsqueeze(2))).any()),
            "S56 full-context structure or quote safety gate drift")
    return features


class S56HierarchicalClauseRecipient(nn.Module):
    """Research-only paired semantic heads (36+72=108 trainable FP16)."""
    def __init__(self):
        super().__init__()
        self.recipient=nn.Parameter(torch.zeros(len(FEATURE_NAMES),dtype=torch.float32))
        self.rejected=nn.Parameter(torch.zeros(len(FEATURE_NAMES),dtype=torch.float32))
        self.output_gate=GATE
        self.candidate_accepted=False
        self.activation_allowed=False
        self.online_causality_certified=False

    def forward(self,feat,eligible):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and not self.online_causality_certified,
                "S56 semantic scope head is not active or causally certified")
        require(feat.ndim==4 and feat.shape[-1]==len(FEATURE_NAMES)
                and eligible.shape==feat.shape[:3]
                and eligible.dtype==torch.bool
                and feat.dtype==torch.float32
                and bool(torch.isfinite(feat).all()),
                "S56 frozen full-context original43M neural feature ABI missing")
        return feat.matmul(self.recipient),feat.matmul(self.rejected)

    def predict(self,feat,mask):
        with torch.inference_mode():
            role,pol=self(feat,mask)
            yes=(role>0)&mask
            return yes,yes&(pol>0)

    def physical_fp16(self):
        with torch.no_grad():
            w=torch.stack((self.recipient.detach(),self.rejected.detach()))
            require(w.shape==(2,len(FEATURE_NAMES))
                    and torch.isfinite(w).all(),
                    "S56 paid residual shape not finite")
            return w.half().contiguous()

    @classmethod
    def from_physical_fp16(cls,w):
        require(w.dtype==torch.float16 and w.shape==(2,len(FEATURE_NAMES))
                and not w.requires_grad
                and bool(torch.isfinite(w).all()),"S56 old frozen readout ABI mismatch")
        obj=cls().eval()
        with torch.no_grad():
            obj.recipient.copy_(w[0].float())
            obj.rejected.copy_(w[1].float())
        obj.requires_grad_(False)
        require(torch.equal(obj.physical_fp16(),w),"S56 stored frozen FP16 roundtrip drift")
        return obj


def fixed_train_from_SPENT_S53_S54_S55_only(
    features:list[torch.Tensor],masks:list[torch.Tensor],
    truths:list[torch.Tensor],rejected:list[torch.Tensor],
):
    """Exactly three separately previously inspected 24-row TRAIN sources."""
    require(len(features)==len(masks)==len(truths)==len(rejected)==3,
            "S56 only S53 S54 S55 inspected TRAIN sources allowed")
    matrices=[]
    for x,m,r,p in zip(features,masks,truths,rejected):
        require(x.ndim==4 and x.shape[0]==24
                and x.shape[-1]==len(FEATURE_NAMES)
                and x.shape[:3]==m.shape==r.shape==p.shape
                and m.dtype==r.dtype==p.dtype==torch.bool
                and not (r&~m).any() and not (p&~r).any(),
                "S56 TRAIN model-mask/gold leak or off-mask true edge")
        matrices.append((x.detach().clone(),m.detach().clone(),
                         r.detach().clone(),p.detach().clone()))
    torch.manual_seed(561118)
    model=S56HierarchicalClauseRecipient()
    optim=torch.optim.Adam(model.parameters(),lr=.045)
    total_truth=sum(int(x[2].sum()) for x in matrices)
    total_rej=sum(int(x[3].sum()) for x in matrices)
    require(total_truth==18+18+12 and total_rej==12+12+6,
            "S56 must train on exact spent S53+S54+S55 truths 48/30")
    pos=sum(int(x[1].sum()) for x in matrices)
    require(pos>total_truth, "S56 eligible pair class balance invalid")
    weight=torch.tensor(min((pos-total_truth)/total_truth,20.),dtype=torch.float32)
    for iteration in range(350):
        optim.zero_grad(set_to_none=True)
        losses=[]
        for x,mask,truth,pol in matrices:
            role,neg=model(x,mask)
            losses.append(F.binary_cross_entropy_with_logits(
                role[mask],truth[mask].float(),pos_weight=weight))
            losses.append(F.binary_cross_entropy_with_logits(
                neg[truth],pol[truth].float(),pos_weight=torch.tensor(.55)))
        loss=sum(losses)/3+.0015*(model.recipient.square().sum()
                                  +model.rejected.square().sum())
        require(bool(torch.isfinite(loss)),"S56 TRAIN loss nonfinite")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),"S56 train gradient invalid")
        optim.step()
    model.eval().requires_grad_(False)
    physical=model.physical_fp16()
    restored=S56HierarchicalClauseRecipient.from_physical_fp16(physical)
    return restored,physical,{
        "stage":"S56_TRAIN_ON_SPENT_OLD_GOLD_ONLY",
        "TRAIN_sources":("S53_PREVIOUSLY_SCORED24",
                         "S54_PREVIOUSLY_SCORED_HELD24",
                         "S55_PREVIOUSLY_SCORED_HELD24"),
        "TRAIN_gold_RECIPIENT":total_truth,
        "TRAIN_gold_REJECTED":total_rej,
        "optimizer":"Adam fixed 350 steps lr=0.045, L2=0.0015",
        "new_fp16_parameters":PAID_PARAM_COUNT,
        "S56_HELD24_never_loaded_in_train":True,
        "no_original_43M_GT_S26_or_S54_training":True,
    }
