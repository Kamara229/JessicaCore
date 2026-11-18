"""S57 genuinely hierarchical EVENT-FIRST / TARGET-SECOND / NEG-SCOPE-THIRD.

Original frozen 12-layer Calibri43M+GreenTunnel layer8 hidden320, old S26
paid five-head semantic graph, and the previous S56 108-FP16 readout
are READ-ONLY features. Train a NEW small event-node classifier,
separately trained conditional source->recipient target ranking, and
event-local negative-polarity classifier on 96 PREVIOUSLY INSPECTED S53-
S56 research sentences ONLY. S57 fresh HELD24 never used for training.

Autoregressive online safety NOT established. Handwritten main predicate
and recipient only enter supervised TRAIN loss AFTER feature generation.
Inference NEVER receives annotated event/recipient, manually typed
transfer/none class, gold, row IDs, or category. Event/pair is picked
from native and raw grammar source under S52 quote/BPE unsafe exclusion.

ONE maximum directed transfer-recipient edge per utterance; this is an
explicit limited research policy, not a generic multi-event decoder.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    raw_surface_grammar,FEATURES,
)
from research.nra3_s19.s19f_real43m_train9_graph_targets import (
    STATUS,ROLE,
)
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    SafeGraphReadOnly,
)
from research.nra3_s20.s56_grammar_clause_scope_semantic_readout import (
    S56HierarchicalClauseRecipient,real43m_semantic_structure_features,
    FEATURE_NAMES as S56_PAIR_FEATURE_NAMES,grammar_scope_metadata,
)
from research.nra3_s20.s51_fractional_native_bpe_overlap import require

NATIVE_FEATURE_NAMES=tuple("s18_"+s for s in FEATURES)
EVENT_STATUS_NAMES=tuple("S26_status_"+s for s in STATUS)
EVENT_EXTRA_NAMES=(
    "intercept",
    "old_S56_top_link_probability",
    "old_S56_max_link_logit_clipped",
    "old_S56_number_positive_link_candidates_norm",
    "old_S56_max_rejected_logit_clipped",
    "old_S26_top_recipient_probability",
    "old_S26_top_agent_probability",
    "negative_in_own_clause",
    "negative_before_source_within_four",
    "negative_immediately_before",
    "negative_in_another_clause",
    "nearest_left_negative_reciprocal",
    "nearest_right_negative_reciprocal",
    "clause_first",
    "clause_last",
    "clause_position_normalized",
    "raw_word_position_normalized",
    "clause_dative_count_normalized",
    "clause_verb_count_normalized",
    "clause_length_normalized",
    "native_320D_L2_norm_normalized",
    "native_320D_cosine_to_valid_sentence_mean",
    "quote_flag_heuristic",
    "any_previous_verb_in_own_clause",
    "any_next_verb_in_own_clause",
    "preceded_by_preposition",
    "clause_total_negation_count_normalized",
    "other_clause_negation_count_normalized",
)
NODE_FEATURE_NAMES=NATIVE_FEATURE_NAMES+EVENT_STATUS_NAMES+EVENT_EXTRA_NAMES
NODE_DIM=len(NODE_FEATURE_NAMES)
PAIR_DIM=len(S56_PAIR_FEATURE_NAMES)
TRAINABLE_PARAMS=2*NODE_DIM+PAIR_DIM+1
RAW_PUNCT=FEATURES.index("is_punctuation")
RAW_PAST=FEATURES.index("past_verb_shape_possible")
RAW_DAT=FEATURES.index("ending_dative_possible")
RAW_PASSIVE=FEATURES.index("short_passive_participle_shape_possible")
PREPOSITIONS=frozenset(("к","ко","для","по","до","на","у","в","с","из","от","об","о","при"))
NEG=frozenset(("не","ни"))


def create_event_first_features(
    x339:torch.Tensor,obs:SafeGraphReadOnly,texts:tuple[str,...],
    s56_frozen:S56HierarchicalClauseRecipient,
):
    """Gold BLIND; old real 43M+GT and S26/S56 parameters strictly frozen."""
    b,w,_=x339.shape
    require(x339.ndim==3 and x339.shape==(b,w,339)
            and b==len(texts) and x339.dtype==torch.float32
            and not x339.requires_grad and obs.native_word_mask.shape==(b,w)
            and s56_frozen.output_gate==0
            and not s56_frozen.candidate_accepted
            and all(not p.requires_grad for p in s56_frozen.parameters()),
            "S57 original true43M input or frozen S56 paid108 invalid")
    pair_features=real43m_semantic_structure_features(x339,obs,texts)
    pair_mask=obs.eligible_directed_pair_mask
    with torch.inference_mode():
        old_link,old_neg=s56_frozen(pair_features,pair_mask)
    raw=x339[...,320:338].detach()
    status=torch.softmax(obs.original_paid_graph_logits.event_status.detach(),-1)
    role=torch.softmax(obs.original_paid_graph_logits.role.detach(),-1)
    hidden=x339[...,:320].detach()
    valid=obs.native_word_mask
    vec=F.normalize(hidden.float(),dim=-1,eps=1e-7)
    mean=(hidden.float()*valid.unsqueeze(-1)).sum(1)/valid.sum(-1,keepdim=True).clamp(min=1)
    normalized_mean=F.normalize(mean,dim=-1,eps=1e-7)
    centroid=(vec*normalized_mean.unsqueeze(1)).sum(-1).clamp(-1,1)
    node=torch.zeros((b,w,NODE_DIM),dtype=torch.float32)
    valid_pair=torch.where(pair_mask,old_link.detach(),-1e9)
    max_link=valid_pair.max(-1).values.clamp(-10,10)
    max_neg=torch.where(pair_mask,old_neg.detach(),-1e9).max(-1).values.clamp(-10,10)
    eligible_counts=pair_mask.sum(-1)
    positive_links=((old_link>0)&pair_mask).sum(-1)
    max_rec=role[...,ROLE.index("RECIPIENT")].max(-1).values
    max_agent=role[...,ROLE.index("AGENT")].max(-1).values
    for row,text in enumerate(texts):
        tokens,cids,groups,negs,verbs,datives=grammar_scope_metadata(text)
        n=int(valid[row].sum())
        require(n==len(tokens) and n<=w and n>=3,
                "S57 original true GT/Unicode word offsets diverged")
        max_clause=max(cids)
        neg_total=sum(len(v) for v in negs.values())
        for i,word in enumerate(tokens):
            clause=cids[i]
            idxs=groups[clause]
            negatives=negs[clause]
            left=[i-k for k in negatives if k<i]
            right=[k-i for k in negatives if k>i]
            previous=tokens[i-1].form.casefold() if i else ""
            previous_verb=any(t<i for t in verbs[clause])
            following_verb=any(t>i for t in verbs[clause])
            remaining=(
              1.,
              float(torch.sigmoid(max_link[row,i])) if eligible_counts[row,i] else 0.,
              float(max_link[row,i])/10. if eligible_counts[row,i] else -1.,
              min(int(positive_links[row,i]),8)/8.,
              float(max_neg[row,i])/10. if eligible_counts[row,i] else -1.,
              float(max_rec[row,i]),
              float(max_agent[row,i]),
              int(bool(negatives)),
              int(any(d<=4 for d in left)),
              int(1 in left),
              int(neg_total>len(negatives)),
              (1/min(left)) if left else 0.,
              (1/min(right)) if right else 0.,
              int(clause==0),
              int(clause==max_clause),
              clause/(max_clause+1),
              i/max(1,n-1),
              min(len(datives[clause]),6)/6.,
              min(len(verbs[clause]),6)/6.,
              min(len(idxs),28)/28.,
              float(torch.linalg.vector_norm(hidden[row,i].float()).clamp(max=50))/50.,
              float(centroid[row,i]),
              float(x339[row,i,338]),
              int(previous_verb),
              int(following_verb),
              int(previous in PREPOSITIONS),
              min(len(negatives),5)/5.,
              min(neg_total-len(negatives),5)/5.,
            )
            require(len(remaining)==len(EVENT_EXTRA_NAMES),
                    "S57 event-first node grammar dimensionality changed")
            node[row,i,:len(FEATURES)]=raw[row,i]
            node[row,i,len(FEATURES):len(FEATURES)+len(STATUS)]=status[row,i]
            node[row,i,len(FEATURES)+len(STATUS):]=torch.tensor(
                remaining,dtype=torch.float32)
    require(node.shape==(b,w,NODE_DIM) and bool(torch.isfinite(node).all())
            and bool(torch.isfinite(pair_features).all())
            and not node.requires_grad
            and not pair_features.requires_grad
            and not bool((pair_mask&~valid.unsqueeze(1)).any()),
            "S57 original 43M frozen event node/pair feature contract failed")
    # Only source-text clause ID is used to restrict pair edges, never
    # manually assigned predicate or gold recipient. This is deliberately
    # conservative: may miss valid cross-clause relations.
    same_clause=torch.zeros_like(pair_mask)
    for row,text in enumerate(texts):
        _,cids,_,_,_,_=grammar_scope_metadata(text)
        for i in range(len(cids)):
            for j in range(len(cids)):
                same_clause[row,i,j]=cids[i]==cids[j]
    limited_pair=pair_mask&same_clause
    require(not bool((limited_pair&~pair_mask).any()),
            "S57 clause gate accidentally enabled a blocked original BPE/quote pair")
    return node.detach(),pair_features.detach(),limited_pair.detach()


class S57EventFirstDecoder(nn.Module):
    """Separate node transfer, conditional recipient and node polarity heads."""
    def __init__(self):
        super().__init__()
        self.source_event=nn.Parameter(torch.zeros(NODE_DIM,dtype=torch.float32))
        self.target_recipient=nn.Parameter(torch.zeros(PAIR_DIM,dtype=torch.float32))
        self.source_negation=nn.Parameter(torch.zeros(NODE_DIM,dtype=torch.float32))
        self.abstain=nn.Parameter(torch.zeros(1,dtype=torch.float32))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.streaming_causal_certified=False

    def forward(self,node,pair,valid_words,eligible_pairs):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and not self.streaming_causal_certified,
                "S57 candidate must remain zero-gated, disconnected, research-only")
        require(node.ndim==3 and node.shape[-1]==NODE_DIM
                and pair.shape[:3]==eligible_pairs.shape
                and pair.shape[-1]==PAIR_DIM
                and node.shape[:2]==valid_words.shape
                and valid_words.dtype==eligible_pairs.dtype==torch.bool
                and node.dtype==pair.dtype==torch.float32
                and bool(torch.isfinite(node).all()) and bool(torch.isfinite(pair).all())
                and not bool((eligible_pairs&~valid_words.unsqueeze(1)).any()),
                "S57 original 43M safe event/pair input ABI malformed")
        source=node.matmul(self.source_event)
        target=pair.matmul(self.target_recipient)
        polarity=node.matmul(self.source_negation)
        none=self.abstain.expand(node.shape[0],1)
        # logits have separate native eligible word and pair masks. If
        # caller supplies unsafe source as valid, it fails in the exact
        # staging pipeline where valid is S52 eligible_word_mask.
        source=torch.cat((source.masked_fill(~valid_words,-1e7),none),dim=-1)
        target=target.masked_fill(~eligible_pairs,-1e7)
        return source,target,polarity

    def predict(self,node,pair,valid_words,eligible_pairs):
        with torch.inference_mode():
            source,target,pol=self(node,pair,valid_words,eligible_pairs)
            b,w=valid_words.shape
            selected=source.argmax(-1)
            out_pair=torch.zeros((b,w,w),dtype=torch.bool)
            out_rejected=torch.zeros_like(out_pair)
            source_selected=torch.zeros_like(valid_words)
            for row in range(b):
                i=int(selected[row])
                if i==w or not bool(valid_words[row,i]):continue
                if not bool(eligible_pairs[row,i].any()):continue
                j=int(target[row,i].argmax())
                if not bool(eligible_pairs[row,i,j]):continue
                # hard at-most-one edge per example. No gold anchors,
                # no arbitrary all-pairs argmax>0 overproduction.
                out_pair[row,i,j]=True
                source_selected[row,i]=True
                if bool(pol[row,i]>0):out_rejected[row,i,j]=True
            require(int(out_pair.sum())<=b
                    and not bool((out_pair&~eligible_pairs).any())
                    and not bool((out_rejected&~out_pair).any()),
                    "S57 event-first decoder emitted unsafe or multiple pair")
            return out_pair,out_rejected,source_selected

    def physical_fp16(self):
        with torch.no_grad():
            all_params=torch.cat((self.source_event.detach(),
                                  self.target_recipient.detach(),
                                  self.source_negation.detach(),
                                  self.abstain.detach())).cpu()
            require(all_params.ndim==1 and len(all_params)==TRAINABLE_PARAMS
                    and bool(torch.isfinite(all_params).all())
                    and self.output_gate==0 and not self.candidate_accepted,
                    "S57 original disconnected source-target-polarity FP16 missing")
            return all_params.half().contiguous()

    @classmethod
    def from_physical_fp16(cls,w):
        require(isinstance(w,torch.Tensor) and w.dtype==torch.float16
                and w.ndim==1 and len(w)==TRAINABLE_PARAMS
                and bool(torch.isfinite(w).all())
                and not w.requires_grad,
                "S57 exact paid event-first FP16 physical shard required")
        obj=cls().eval()
        with torch.no_grad():
            at=0
            for p in (obj.source_event,obj.target_recipient,obj.source_negation,obj.abstain):
                n=p.numel()
                p.copy_(w[at:at+n].float())
                at+=n
        obj.requires_grad_(False)
        require(torch.equal(obj.physical_fp16(),w),
                "S57 immutable paid FP16 physical source head reconstruction changed")
        return obj


def fixed_train_old96_only(source_node,source_pair,valid,eligible,gold_source,
                           gold_target,gold_negative):
    """96 spent old cases, 60 transfer sources, 36 NONE, 30 negative."""
    require(all(len(q)==4 for q in (
        source_node,source_pair,valid,eligible,gold_source,gold_target,gold_negative)),
        "S57 exactly four previously inspected/consumed S53-S56 groups required")
    groups=[]
    for node,pair,vm,pm,src,dst,rej in zip(
        source_node,source_pair,valid,eligible,gold_source,gold_target,gold_negative,
    ):
        require(node.ndim==3 and node.shape[0]==24
                and node.shape[-1]==NODE_DIM
                and pair.shape[:3]==pm.shape and pair.shape[-1]==PAIR_DIM
                and vm.shape==src.shape==node.shape[:2]
                and dst.shape==rej.shape==pm.shape
                and vm.dtype==pm.dtype==src.dtype==dst.dtype==rej.dtype==torch.bool
                and not bool((src&~vm).any())
                and not bool((dst&~pm).any())
                and not bool((rej&~dst).any())
                and all(int(src[i].sum())<=1 for i in range(24)),
                "S57 source gold entered original decoder or true edge excluded by mask")
        groups.append((node.detach().clone(),pair.detach().clone(),
                       vm.detach().clone(),pm.detach().clone(),
                       src.detach().clone(),dst.detach().clone(),rej.detach().clone()))
    count_source=sum(int(q[4].sum()) for q in groups)
    count_links=sum(int(q[5].sum()) for q in groups)
    count_rejected=sum(int(q[6].sum()) for q in groups)
    require(count_source==count_links==60 and count_rejected==36,
            "S57 old96 manually scored gold should be 60 links / 36 REJECTED")
    # Gold REJECTED = S53 12 + S54 12 + S55 6 + S56 6 = 36
    torch.manual_seed(571118)
    model=S57EventFirstDecoder()
    opt=torch.optim.Adam(model.parameters(),lr=.04,weight_decay=0.)
    for iteration in range(330):
        opt.zero_grad(set_to_none=True)
        loss_event=0.
        loss_target=0.
        loss_scope=0.
        for node,pair,vm,pm,src,dst,rej in groups:
            sl,tl,nl=model(node,pair,vm,pm)
            w=vm.shape[1]
            target_idx=torch.full((24,),w,dtype=torch.long)
            for row in range(24):
                if bool(src[row].any()):
                    target_idx[row]=int(torch.where(src[row])[0][0])
            loss_event=loss_event+F.cross_entropy(sl,target_idx)
            goldrows=torch.where(src.any(-1))[0]
            for row in goldrows:
                i=int(torch.where(src[row])[0][0])
                j=int(torch.where(dst[row,i])[0][0])
                loss_target=loss_target+F.cross_entropy(
                    tl[row,i].unsqueeze(0),torch.tensor([j]))
                loss_scope=loss_scope+F.binary_cross_entropy_with_logits(
                    nl[row,i].unsqueeze(0),rej[row,i,j].float().reshape(1))
        loss=(loss_event/4+loss_target/60+loss_scope/60+
              .001*(model.source_event.square().sum()
                   +model.target_recipient.square().sum()
                   +model.source_negation.square().sum()
                   +model.abstain.square().sum()))
        require(bool(torch.isfinite(loss)),"S57 old TRAIN96 loss became nonfinite")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),
                "S57 old TRAIN96 event-first backward missing or unstable")
        opt.step()
    model.eval().requires_grad_(False)
    paid=model.physical_fp16()
    frozen=S57EventFirstDecoder.from_physical_fp16(paid)
    with torch.inference_mode():
        outputs=[frozen.predict(*q[:4]) for q in groups]
    return frozen,paid,{
        "old_spent96_gold_source_events":count_source,
        "old_spent96_gold_recipient_edges":count_links,
        "old_spent96_gold_REJECTED_edges":count_rejected,
        "old_spent96_nontransfer_NONE_events":96-count_source,
        "event_head_node_dim":NODE_DIM,
        "recipient_head_pair_dim":PAIR_DIM,
        "new_FP16_trainable_parameters":TRAINABLE_PARAMS,
        "optimizer":"Adam 330 steps lr=0.04 L2=0.001",
        "old_TRAIN96_reestimated_RECIPIENT":sum(int(t[0].sum()) for t in outputs),
        "old_TRAIN96_reestimated_REJECTED":sum(int(t[1].sum()) for t in outputs),
        "old_TRAIN96_resubstitution_NOT_independent":True,
        "S57_fresh_HELD24_not_loaded_in_gradient":True,
        "original_Calibri43M_GT_S26_S54_S56_frozen":True,
    }
