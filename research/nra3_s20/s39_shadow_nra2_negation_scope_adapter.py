"""S39 isolated NRA2-compatible shadow parser: TRAIN-only gold-blind forward.

An experimental small 3-head grammar adapter, NOT the original NRA2 or
Calibri43M+GT architecture: head1 predicate candidate scores, head2
quoted-mention token scores, head3 directed rejected-recipient scores.
All forward features derive exclusively from *plain raw Russian text*.
Role/event/scope human annotations are passed ONLY to training loss.
There is no 43M hidden320 embedding or trained Calibri adapter integration.
No S38 challenge data may enter training or hyperparameter selection.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import torch
from torch import nn
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    FEATURES,raw_surface_grammar,
)

S39_DIM=32
PRED_DIM=16
QUOTE_DIM=5
PAID_S39_GATE=0
NAMES=("predicate_event","quote_mention","directed_rejected_recipient")

def require(ok,message):
    if not bool(ok):raise ValueError("S39 TRAIN-only shadow NRA2 graph: "+message)

def _inside_quotes(tokens):
    # Stateful plain-text Unicode punctuation quotation boundaries.
    inside=[]
    depth=0
    for t in tokens:
        if t.form in ("«","“","„","\"","'"):
            if t.form in ("\"","'"):
                depth=1-depth
            else:
                depth+=1
            inside.append(0.)
        elif t.form in ("»","”"):
            inside.append(0.)
            depth=max(0,depth-1)
        else:
            inside.append(float(depth>0))
    return inside

def _infinitive_shape(token):
    s=token.form.casefold()
    return int(s.endswith(("ть","ти","чь")) and len(s)>3)

def _nonlocal_negative_pronoun(token):
    # Visible orthography, NOT gold morphological annotations.
    s=token.form.casefold()
    return int(s.startswith(("ник","нич","никт","никому","никак")))

def _neg_before(tokens,j,window):
    b=max(0,j-window)
    return int(any(x.form.casefold() in ("не","ни") for x in tokens[b:j]))

def raw_token_pair_features(text):
    """Return (W,16) predicate, (W,5) quotes, (W,W,32) pair, FROM TEXT ONLY."""
    tok=raw_surface_grammar(text)
    n=len(tok)
    require(n>=3 and n<=40,"unsupported raw word count")
    quotes=_inside_quotes(tok)
    pred=[]
    quote=[]
    pair=[]
    offset={k:FEATURES.index(k) for k in (
      "past_verb_shape_possible","finite_future_verb_shape_possible",
      "short_passive_participle_shape_possible","negative_particle",
      "previous_word_negative_particle","ending_dative_possible",
      "initial_uppercase","is_punctuation")}
    for i,t in enumerate(tok):
        f=t.features
        before=tok[i-1].form.casefold() if i else ""
        item=[
           float(f[offset["past_verb_shape_possible"]]),
           float(f[offset["finite_future_verb_shape_possible"]]),
           float(f[offset["short_passive_participle_shape_possible"]]),
           float(_infinitive_shape(t)),
           float(f[offset["initial_uppercase"]]),
           float(f[offset["is_punctuation"]]),
           float(f[offset["ending_dative_possible"]]),
           float(f[offset["previous_word_negative_particle"]]),
           float(f[offset["negative_particle"]]),
           float(quotes[i]),
           float(before in ("не","ни")),
           float(_neg_before(tok,i,3)),
           float(i/max(1,n-1)),
           float(n<=8),
           float(t.form[0].isupper()),
           1.,
        ]
        require(len(item)==PRED_DIM,"predicate feature ABI changed")
        pred.append(item)
        quote.append([
            float(quotes[i]),
            float(f[offset["previous_word_negative_particle"]]),
            float(f[offset["ending_dative_possible"]]),
            float(f[offset["is_punctuation"]]),
            1.,
        ])
    for i in range(n):
        line=[]
        for j in range(n):
            a,b=tok[i],tok[j]
            fa,fb=a.features,b.features
            between=tok[min(i,j)+1:max(i,j)]
            explicit_not_target=fb[offset["previous_word_negative_particle"]]
            negative_target=_nonlocal_negative_pronoun(b)
            neg_before_source=_neg_before(tok,i,2)
            signal=[
               float(fa[offset["past_verb_shape_possible"]]),
               float(fa[offset["finite_future_verb_shape_possible"]]),
               float(fa[offset["short_passive_participle_shape_possible"]]),
               float(_infinitive_shape(a)),
               float(fa[offset["is_punctuation"]]),
               float(fb[offset["ending_dative_possible"]]),
               float(fb[offset["previous_word_negative_particle"]]),
               float(fb[offset["is_punctuation"]]),
               float(fb[offset["negative_particle"]]),
               float(negative_target),
               float(quotes[i]),float(quotes[j]),
               float(i<j),float(i>j),
               float(abs(i-j)/max(1,n-1)),
               float(_neg_before(tok,j,3)),
               float(_neg_before(tok,j,5)),
               float(neg_before_source),
               float(any(t.form.casefold()=="не" for t in between)),
               float(any(t.form.casefold()=="ни" for t in between)),
               float(any(t.form.casefold()=="но" for t in between)),
               float(any(t.form in ("«","»","“","”") for t in between)),
               float(any(t.form.casefold()=="вовсе" for t in between)),
               float(any(t.form.casefold()=="совсем" for t in between)),
               float(any(t.form.casefold()=="никому" for t in tok)),
               float(any(t.form.casefold()=="не" for t in tok)),
               float(fa[offset["initial_uppercase"]]),
               float(fb[offset["initial_uppercase"]]),
               float(i==j),
               float(i==0),
               float(j==0),
               1.,
            ]
            require(len(signal)==S39_DIM,"pair feature ABI changed")
            line.append(signal)
        pair.append(line)
    return (torch.tensor(pred,dtype=torch.float32),
            torch.tensor(quote,dtype=torch.float32),
            torch.tensor(pair,dtype=torch.float32))

@dataclass
class GraphPrediction:
    event_logits:torch.Tensor
    quote_logits:torch.Tensor
    rejected_logits:torch.Tensor

class ShadowNRA2S39(nn.Module):
    """Small TRAIN-only trained linguistic-graph shadow (NOT active NRA2)."""
    def __init__(self):
        super().__init__()
        self.event=nn.Sequential(nn.Linear(PRED_DIM,12),nn.Tanh(),
                                 nn.Linear(12,1))
        self.quote=nn.Linear(QUOTE_DIM,1)
        self.scope=nn.Sequential(nn.Linear(S39_DIM+2,32),nn.Tanh(),
                                 nn.Linear(32,1))
        self.output_gate=PAID_S39_GATE
        self.candidate_accepted=False
        self.activation_allowed=False

    def forward(self,pred:torch.Tensor,quote:torch.Tensor,pair:torch.Tensor):
        """Plain raw-only inference, one sentence or padded text batch."""
        single=(pred.ndim==2)
        if single:
            pred=pred.unsqueeze(0)
            quote=quote.unsqueeze(0)
            pair=pair.unsqueeze(0)
        require(pred.ndim==3 and quote.ndim==3 and pair.ndim==4
                and pred.shape[:2]==pair.shape[:2]==pair.shape[0:1]+pair.shape[2:3]
                and quote.shape[:2]==pred.shape[:2]
                and pred.shape[-1]==PRED_DIM and quote.shape[-1]==QUOTE_DIM
                and pair.shape[-1]==S39_DIM,"unannotated raw text batch geometry required")
        b,w=pred.shape[:2]
        ev=self.event(pred).squeeze(-1)
        qc=self.quote(quote).squeeze(-1)
        # Learned event and quote predictions, never reference labels.
        features=torch.cat((
            pair,
            ev.unsqueeze(2).unsqueeze(-1).expand(b,w,w,1),
            qc.unsqueeze(1).unsqueeze(-1).expand(b,w,w,1),
        ),dim=-1)
        rejection=self.scope(features).squeeze(-1)
        if single:
            return GraphPrediction(ev[0],qc[0],rejection[0])
        return GraphPrediction(ev,qc,rejection)

def labels_train_only(row):
    """Only supervised TRAIN targets; never pass to model.forward."""
    n=len(raw_surface_grammar(row.text))
    ev=torch.zeros(n,dtype=torch.float32)
    quotes=torch.tensor(_inside_quotes(raw_surface_grammar(row.text)),
                        dtype=torch.float32)
    pair=torch.zeros(n,n,dtype=torch.float32)
    ev[row.source_predicate_token_index0]=1.
    if row.rejected_recipient:
        pair[row.source_predicate_token_index0,
             row.potential_recipient_token_index0]=1.
    require(int(pair.sum())==int(row.rejected_recipient),
            "gold TRAIN-only graph supervision corrupted")
    return ev,quotes,pair

def model_weights_digest(model):
    tensor_hash=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        tensor_hash.update(key.encode("utf-8"))
        tensor_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    return tensor_hash.hexdigest()
