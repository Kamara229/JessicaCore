"""NRA2 S82: predicate-centered source-only situation fields and 8 experts.

S81 one-time novel-syntax first-read: frozen S78 same-template diagnostic
94.74% -> 28.83% on unseen syntactic structures (teacher anchored).
Never use S79 NOVEL48 as S82 training, hyperparameter fitting,
threshold tuning or model selection. It is SPENT first-read research.

S78 global-mean + isolated event H8 is too coarse. S82 represents
a source-only event-relative FIELD: first-class clause/quote boundaries,
left/right 320D H8 word neighborhoods and 32 deterministic surface
indicators of negation, report, conditional, passive, temporal,
competing clauses and referential cues. Every event receives an
independent nonexclusive set of eight situation logits. No GOLD
is accepted by forward. All text is original input, not event labels.

CRITICAL: these are FALLIBLE shallow lexical indicators, NOT Russian
syntactic/UD dependency parsing, not semantic correctness certificates,
not arbitrary event detection, and NOT streaming causal safety. This
is architecture only: not trained, no true original43M CPU fit, no
independent evaluation, no real-world accuracy claims, output_gate=0.
Original S65 ASSERTED quote exclusion is never lifted.
"""
from __future__ import annotations

import re
from typing import Sequence

import torch
from torch import nn

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar

STAGE="NRA2_S82_SOURCE_ONLY_PREDICATE_CENTERED_CLAUSE_QUOTE_FIELD_8SPECIALISTS"
AREAS=(
    "quoted_or_reported",
    "negation_scope",
    "passive_role_assignment",
    "temporal_revision",
    "hypothetical_or_dream",
    "reference_resolution",
    "competing_event_clauses",
    "uncertainty_abstain",
)
FIELDS=(
    "inside_literal_quote",
    "quote_mark_within_2",
    "left_immediate_negator",
    "left_near_negator",
    "right_near_negator",
    "negator_away_outside_clause",
    "conditional_trigger_same_clause",
    "conditional_trigger_previous_clause",
    "dream_counterfactual_same_clause",
    "reported_claim_same_clause",
    "report_previous_clause",
    "passive_ending_event_word",
    "past_copula_preceding_event",
    "pronoun_left_near",
    "pronoun_right_near",
    "pronoun_earlier_in_clause",
    "two_possible_verbs_same_clause",
    "contrast_word_left_near",
    "contrast_word_right_near",
    "temporal_word_left_near",
    "temporal_word_right_near",
    "temporal_word_previous_clause",
    "uncertainty_local",
    "uncertainty_prev_clause",
    "future_event_morphology",
    "instrumental_agent_left_or_right",
    "multiple_independent_clauses",
    "event_after_clause_boundary",
    "event_before_clause_boundary",
    "left_clause_context_available",
    "right_clause_context_available",
    "quoted_or_nonquoted_other_clause",
)
WIDTH=len(FIELDS)
assert WIDTH==32 and len(AREAS)==8
NEG=frozenset(("не","ни","нет","никогда","вовсе","отнюдь","ничуть"))
COND=frozenset(("если","когда","при","условии","допустим","предположим","пусть","пускай"))
DREAM=frozenset(("сон","сне","снилось","приснилось","вообразил","представил","будто","якобы","мечтал","грезил","предположим","допустим","бы"))
REPORT=frozenset(("сказал","сказала","утверждал","утверждала","сообщили","сообщил",
                  "рассказал","пересказал","написал","говорят","говорил",
                  "цитировал","процитировал","цитата","слух","слова","якобы",
                  "заявил","заявила","текст","стенограмме","сведениях",
                  "письме","записи","протоколе","сообщении","утверждение"))
PRONOUN=frozenset(("он","она","оно","они","ему","ей","им","их","ее","её","его","их",
                   "тот","та","те","это","этот","эта","свой","свои","свою","своим",
                   "нему","ней","нее","неё","сам","сама","того","той","тому"))
CONTRAST=frozenset(("но","хотя","однако","зато","впрочем","вместо","тогда","теперь","всё-таки","все-таки","а"))
TEMPORAL=frozenset(("сначала","сперва","потом","затем","позже","ранее","вчера",
                    "сегодня","завтра","после","перед","сейчас","впоследствии",
                    "вначале","наконец","вновь","снова","теперь","уже","ещё","еще"))
UNCERTAIN=frozenset(("возможно","вероятно","предположительно","слух","слухи",
                     "якобы","будто","наверное","неизвестно","неясно","сомнительно",
                     "кажется","вряд","ли","может","мог","могла","могли","пока"))
PASSIVE_SUFFIX=("ен","ён","ан","ян","ено","ена","аны","ены","ён","ена","ено",
                "енa","ёнa","анa","т","та","то","ты","н","на","но","ны")
COPULA=frozenset(("был","была","было","были","будет","будут"))
INSTRUMENT_SUFFIX=("ом","ем","ым","им","ой","ей","ою","ею","ами","ями")
BOUNDARY=frozenset((".","!","?",";",",",":","…"))
CLOSEQUOTE=frozenset(("»","”","\""))
OPENQUOTE=frozenset(("«","„","“","\""))


def require(ok,what):
    if not bool(ok):
        raise ValueError("S82 source-only predicate field: "+what)


def _raw_only_clauses_and_quotes(words):
    """Input-only lexical window; never reference an annotation or label."""
    clause=[]
    scope=[]
    cid=0
    inside=False
    for z in words:
        w=z.form
        if w in OPENQUOTE:
            inside=not inside if w=="\"" else True
        clause.append(cid)
        scope.append(inside)
        if w in CLOSEQUOTE:
            inside=not inside if w=="\"" else False
        if w in BOUNDARY:
            cid+=1
    return clause,scope


def source_only_event_fields(texts:Sequence[str],word_valid:torch.Tensor,
                             native_unsafe:torch.Tensor,
                             candidate_word_indices:torch.Tensor,
                             candidate_allowed:torch.Tensor):
    """Surface 32D cues. No gold argument or target-label import possible.

    Rule cues examine ONLY the clause containing the proposed event and
    the immediately previous clause for report/conditional/temporal
    context. An unrelated appended sentence cannot turn an earlier
    event into an assertion or negate it.
    """
    require(word_valid.ndim==2 and word_valid.dtype==torch.bool
            and native_unsafe.shape==word_valid.shape
            and native_unsafe.dtype==torch.bool
            and candidate_word_indices.ndim==2
            and candidate_allowed.shape==candidate_word_indices.shape
            and candidate_allowed.dtype==torch.bool
            and candidate_word_indices.dtype==torch.long
            and len(texts)==word_valid.shape[0],
            "source word mask and candidate only")
    b,w=word_valid.shape;k=candidate_word_indices.shape[1]
    require(not bool((native_unsafe&~word_valid).any())
            and bool(((candidate_word_indices>=0)&
                      (candidate_word_indices<w)).all()),
            "valid candidate indices/unsafe original BPE rows")
    out=torch.zeros((b,k,WIDTH),dtype=torch.float32)
    for i,text in enumerate(texts):
        tokens=raw_surface_grammar(text)
        n=int(word_valid[i].sum().item())
        require(len(tokens)==n and n<=w
                and bool(word_valid[i,:n].all())
                and not bool(word_valid[i,n:].any()),
                "exact unchanged source/tokenizer original Russian word order")
        clause,quote=_raw_only_clauses_and_quotes(tokens)
        raw=[z.form.casefold() for z in tokens]
        # Only clauses up to the proposed event may affect the
        # structural field; a later *unrelated* sentence must not
        # change the earlier event's factors.
        for j in range(k):
            if not bool(candidate_allowed[i,j]):continue
            p=int(candidate_word_indices[i,j])
            require(p<n and not bool(native_unsafe[i,p]),
                    "S82 never train/use unsafe original BPE candidate")
            c=clause[p]
            left=[t for t in range(p) if clause[t]==c and tokens[t].form not in OPENQUOTE]
            right=[t for t in range(p+1,n) if clause[t]==c and tokens[t].form not in CLOSEQUOTE]
            earlier=[t for t in range(p) if clause[t]==c-1]
            elsewhere=[t for t in range(p) if clause[t]!=c]
            prev=left[-5:];nxt=right[:5]
            w0=raw[p]
            # Fully source-only, fallible morphological cues.
            source_past=bool(tokens[p].features[12])
            source_passive=bool(tokens[p].features[13]) and (
                w0.endswith(PASSIVE_SUFFIX)
                and not (source_past and w0.endswith(("л","ла","ли","ло"))))
            possible_verbs=sum(
                int(bool(tokens[z].features[12]) or
                    bool(tokens[z].features[13]) or
                    bool(tokens[z].features[14]))
                for z in left+right)
            values=(
                quote[p],
                any(tokens[t].form in OPENQUOTE|CLOSEQUOTE
                    for t in range(max(0,p-2),min(n,p+3))),
                bool(prev and raw[prev[-1]] in NEG),
                any(raw[t] in NEG for t in prev[:-1]),
                any(raw[t] in NEG for t in nxt[:3]),
                any(raw[t] in NEG for t in elsewhere),
                any(raw[t] in COND for t in left+right),
                any(raw[t] in COND for t in earlier),
                any(raw[t] in DREAM for t in left+right),
                any(raw[t] in REPORT for t in left+right),
                any(raw[t] in REPORT for t in earlier),
                source_passive,
                any(raw[t] in COPULA for t in prev),
                any(raw[t] in PRONOUN for t in prev),
                any(raw[t] in PRONOUN for t in nxt),
                any(raw[t] in PRONOUN for t in left[:-5]),
                possible_verbs>=1,
                any(raw[t] in CONTRAST for t in prev),
                any(raw[t] in CONTRAST for t in nxt),
                any(raw[t] in TEMPORAL for t in prev),
                any(raw[t] in TEMPORAL for t in nxt),
                any(raw[t] in TEMPORAL for t in earlier),
                any(raw[t] in UNCERTAIN for t in left+right),
                any(raw[t] in UNCERTAIN for t in earlier),
                bool(tokens[p].features[14]),
                any(raw[t].endswith(INSTRUMENT_SUFFIX)
                    for t in prev+nxt if len(raw[t])>3),
                c>0,
                p>0 and clause[p-1]!=c,
                p+1<n and clause[p+1]!=c,
                bool(left),
                bool(right),
                bool(elsewhere and any(quote[t]!=quote[p] for t in elsewhere)),
            )
            require(len(values)==WIDTH,"fixed 32D original source field")
            out[i,j]=torch.tensor(values,dtype=torch.float32)
    require(torch.isfinite(out).all()
            and bool(((out==0)|(out==1)).all()),
            "S82 strictly binary source-only field with no inferred target")
    return out


class _AreaHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,6),nn.GELU(),nn.Linear(6,1))

    def forward(self,x):
        return self.net(x).squeeze(-1)


class S82PredicateCenteredMiniAreas(nn.Module):
    """Neural H8 lexical context + frozen raw clause geometry; SHADOW only."""
    def __init__(self):
        super().__init__()
        self.event_proj=nn.Linear(320,6,bias=False)
        self.left_proj=nn.Linear(320,6,bias=False)
        self.right_proj=nn.Linear(320,6,bias=False)
        self.areas=nn.ModuleList([_AreaHead(50) for _ in AREAS])
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.is_trained=False
        self.online_causal_certified=False

    def forward(self,h8_original320,word_valid,original_bpe_unsafe,
                source_only_candidate_indices,source_candidate_allowed,
                raw_source_texts):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed
                and not self.online_causal_certified,
                "cannot activate research candidate or infer online")
        h=h8_original320
        require(h.ndim==3 and h.shape[-1]==320 and h.dtype==torch.float32
                and not h.requires_grad and bool(torch.isfinite(h).all())
                and word_valid.shape==original_bpe_unsafe.shape==h.shape[:2]
                and source_candidate_allowed.shape==source_only_candidate_indices.shape
                and len(raw_source_texts)==h.shape[0],
                "source-only original Calibri43M 320D and native masks")
        b,w,_=h.shape;k=source_only_candidate_indices.shape[1]
        fields=source_only_event_fields(
            raw_source_texts,word_valid,original_bpe_unsafe,
            source_only_candidate_indices,source_candidate_allowed).to(h.device)
        safe=word_valid & ~original_bpe_unsafe
        indices=source_only_candidate_indices
        clauses=[]
        for i,text in enumerate(raw_source_texts):
            raw=raw_surface_grammar(text)
            cids,_=_raw_only_clauses_and_quotes(raw)
            require(len(cids)<=w,"original source word count <= padded width")
            clauses.append(cids+[-100]*(w-len(cids)))
        clause_tensor=torch.tensor(clauses,dtype=torch.long,device=h.device)
        event_clause=clause_tensor.gather(1,indices)
        # Clause-local H8 context, not global future-sentence mean.
        ev=h.gather(1,indices[:,:,None].expand(b,k,320))
        left=torch.zeros_like(ev)
        right=torch.zeros_like(ev)
        for d in (1,2,3,4):
            lid=(indices-d).clamp(min=0)
            rid=(indices+d).clamp(max=w-1)
            left_valid=((indices>=d)&safe.gather(1,lid)
                        &(clause_tensor.gather(1,lid)==event_clause))
            right_valid=((indices+d<w)&safe.gather(1,rid)
                         &(clause_tensor.gather(1,rid)==event_clause))
            left+=h.gather(1,lid[:,:,None].expand(b,k,320))*left_valid[:,:,None]
            right+=h.gather(1,rid[:,:,None].expand(b,k,320))*right_valid[:,:,None]
        left=left/4.
        right=right/4.
        merged=torch.cat(
            (self.event_proj(ev),self.left_proj(left),
             self.right_proj(right),fields),dim=-1)
        require(merged.shape==(b,k,50),"S82 3x6 H8 fields +32 literal geometry")
        logits=torch.stack([a(merged) for a in self.areas],dim=-1)
        logits=logits.masked_fill(~source_candidate_allowed[:,:,None],-1.e5)
        require(logits.shape==(b,k,len(AREAS))
                and bool(torch.isfinite(logits).all()),
                "S82 independent eight experts output no pretrained parent changes")
        return logits

    def budget(self):
        total=sum(p.numel() for p in self.parameters())
        return {
            "new_neural_parameters":total,
            "within_10000_new_params_budget":total<=10000,
            "standalone_FP16_bytes":total*2,
            "n_event_relative_structural_fields":WIDTH,
            "eight_independent_expert_modules":len(self.areas),
            "original43m_GT_S65_weights_changed":False,
            "train_or_J7_proof":False,
            "bpp_and_cpu_NOT_CERTIFIED":True,
            "output_gate":0,
            "candidate_accepted":False,
            "activation_allowed":False,
        }
