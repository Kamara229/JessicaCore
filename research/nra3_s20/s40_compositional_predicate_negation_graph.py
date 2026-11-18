"""S40: explicit compositional predicate/negation/quotation graph (TRAIN-only).

A research-only NRA2-adjacent shadow grammar graph: deterministic candidate
scoping from plain Russian Unicode text, with a learned residual jointly
trained with S39's 3-head event/quote/rejection adapter. Source morphology,
quotation boundaries and negation scope are separate interpretable gates.
NO original NRA2, true43M+Green Tunnel hidden states or original 616J7
are loaded or modified. No S38 challenge texts are used.

This is a hand-specified inductive bias with strong prior knowledge of
synthetic S39 constructions. Improved held-construction TRAIN-development
scores would not constitute independent language generalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    FEATURES,raw_surface_grammar,
)
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import (
    ShadowNRA2S39,GraphPrediction,_inside_quotes,_infinitive_shape,
)

STRUCTURE_DIM=12
PREDECLARED_BIAS_ALLOWED=4.0
PREDECLARED_BIAS_BLOCKED=-12.0
CLAUSE_PUNCT=frozenset((".",",",";",":","!","?"))
NEG=frozenset(("не","ни"))
CONNECTORS=frozenset(("но","а","зато","однако"))
NEG_PRONOUNS=frozenset(("никому","ничему","никого","ничего"))
PRED_FORMS=tuple(FEATURES.index(x) for x in (
    "past_verb_shape_possible",
    "finite_future_verb_shape_possible",
    "short_passive_participle_shape_possible",
))
DAT=FEATURES.index("ending_dative_possible")

def plain_scope_features(text):
    """Raw text only; returns [words,words,12] compositional edge cues.

    Do not pass source gold, construction names, TRAIN labels or S38 rows.
    A quoted mention is never interpreted as the object recipient. This
    is a constrained synthetic grammar, not a general linguistic parser.
    """
    tok=raw_surface_grammar(text)
    n=len(tok)
    q=_inside_quotes(tok)
    lower=[t.form.casefold() for t in tok]
    punct=[t.form in CLAUSE_PUNCT for t in tok]
    verb=[bool(any(t.features[k] for k in PRED_FORMS) or _infinitive_shape(t))
          and not t.features[FEATURES.index("is_punctuation")] for t in tok]
    infinitive=[bool(_infinitive_shape(t)) for t in tok]
    result=torch.zeros((n,n,STRUCTURE_DIM),dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            before=lower[j-1] if j else ""
            # Candidate-local negation and discontinuous negative pronoun:
            # not a blanket 'НЕ anywhere in sentence' heuristic.
            local=before in NEG
            pronoun=lower[j] in NEG_PRONOUNS
            between=lower[min(i,j)+1:max(i,j)]
            quote_target=q[j]>0.
            quote_source=q[i]>0.
            # Coordinate 'но' may introduce an explicitly rejected
            # alternate recipient in the SAME directed event scope.
            # Other punctuation separates clauses (heuristic only).
            clause_break=any(
                punct[k] and not (
                    tok[k].form=="," and k+1<max(i,j)
                    and lower[k+1] in CONNECTORS)
                for k in range(min(i,j)+1,max(i,j))
            )
            # Scope may contain an adverb, but neither cross another
            # predicate nor turn "НЕ [predicate]" into rejected-recipient.
            near_ne=False
            if not local and not pronoun:
                lo=max(0,j-4)
                indices=[k for k in range(lo,j) if lower[k] in NEG]
                for k in indices:
                    if not any(verb[x] for x in range(k+1,j)) and not any(
                        punct[x] for x in range(k+1,j)
                    ):
                        near_ne=True
            target_case=bool(tok[j].features[DAT]) or pronoun
            source_is_verb=verb[i]
            # A finite control verb before an infinitive points to a
            # clausal decision, not the inner recipient's source.
            intermediate_inf=any(infinitive[k] for k in range(i+1,j)) if i<j else False
            negative_scope=bool(local or pronoun or near_ne)
            eligible=bool(source_is_verb and target_case and negative_scope
                          and not quote_target and not quote_source and
                          not clause_break and not intermediate_inf)
            feat=[
                float(source_is_verb),float(infinitive[i]),
                float(any(tok[i].features[k] for k in PRED_FORMS)),
                float(target_case),float(local),float(pronoun),
                float(near_ne),float(quote_target),float(quote_source),
                float(clause_break or intermediate_inf),
                float(i<j),float(eligible),
            ]
            result[i,j]=torch.tensor(feat)
    return result

class CompositionalNRA2S40(nn.Module):
    """Learned correction over explicit directed grammar eligibility."""
    def __init__(self):
        super().__init__()
        self.base=ShadowNRA2S39()
        self.structural=nn.Sequential(nn.Linear(STRUCTURE_DIM,16),
                                       nn.Tanh(),nn.Linear(16,1))
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False

    def forward(self,pred,quote,pair,structure):
        single=(pred.ndim==2)
        if single: structure=structure.unsqueeze(0)
        output=self.base(pred,quote,pair)
        require_shape=(
            structure.ndim==4 and structure.shape[-1]==STRUCTURE_DIM and
            structure.shape[1]==structure.shape[2] and
            structure.shape[0]==(1 if single else pred.shape[0]))
        if not require_shape:raise ValueError("S40 expected raw-text-only 12D pair grammar")
        correction=self.structural(structure).squeeze(-1)
        structural_eligible=structure[...,11]
        prior=structural_eligible*PREDECLARED_BIAS_ALLOWED+(
            1.-structural_eligible)*PREDECLARED_BIAS_BLOCKED
        corrected=output.rejected_logits+(
            (correction+prior)[0] if single else correction+prior)
        return GraphPrediction(output.event_logits,output.quote_logits,corrected)
