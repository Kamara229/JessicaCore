"""NRA3 S4: causal complete-prefix word/token alignment and native in-area loss.

Not a second adapter. Reuses NRA3's own grammar, semantic and dependency
heads, with NO additional learned weights. At the end of the current
available prefix, the final token's hidden state has access to all earlier
tokens and can condition earlier word representations for dependency
heads that occur to their right in Russian ("Анна ... отправила").

This *does not* leak future generation tokens. Invocation is exclusively at
a complete, currently-visible prefix. No gold UD word structure is ever
passed into the original frozen Calibri model; training annotations are
used downstream only after its tokenizer/CPU hidden320 has been captured.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import torch
from torch.nn import functional as F

from research.nra3_s0.neural_area import UPOS,MORPH,SEMANTIC_ROLES,NRA3NeuralArea
from research.nra3_s4.russian_ud_train_only import (
    RussianTrainSentence,TRAIN_SPLIT,
)


@dataclass(frozen=True)
class AlignedWord:
    word_id:int
    token_positions:tuple[int,...]
    last_token:int


@dataclass(frozen=True)
class CausalAlignment:
    original_prompt_sha256:str
    tokenizer_token_count:int
    prefill_token_ids:tuple[int,...]
    word_alignment:tuple[AlignedWord,...]
    complete_prefix_position:int
    independent_eval_labels_loaded:bool=False


def align_complete_chat_prefix(tokenizer,row:RussianTrainSentence,
                               *,expected_input_ids=None)->CausalAlignment:
    """Fast-tokenizer offset ABI, nonoverlapping completed word spans.

    The full chat template includes user text ONCE; model tokens and offsets
    must describe exactly the same prefix. No token may overlap two words.
    Each character of a word must be covered by aligned token offsets.
    """
    if row.split!=TRAIN_SPLIT:
        raise ValueError("S4 only accepts explicit Russian UD TRAIN")
    if tokenizer is None:
        raise ValueError("S4 tokenizer is required")
    try:
        full=tokenizer.apply_chat_template(
            [{"role":"user","content":row.text}],
            tokenize=False,add_generation_prompt=True)
    except Exception as exc:
        raise ValueError("S4 tokenizer chat-template error") from exc
    if not isinstance(full,str) or full.count(row.text)!=1:
        raise ValueError("S4 chat-template must contain exact user prompt once")
    start_user=full.index(row.text)
    try:
        ids=tuple(tokenizer.encode(full,add_special_tokens=False))
        enc=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
        returned_ids=tuple(enc["input_ids"])
        offsets=tuple((int(a),int(b)) for a,b in enc["offset_mapping"])
    except Exception as exc:
        raise ValueError("S4 tokenizer must expose exact fast offsets") from exc
    if (not ids or ids!=returned_ids or len(ids)!=len(offsets)
            or (expected_input_ids is not None and
                ids!=tuple(expected_input_ids))):
        raise ValueError("S4 actual Calibri prefill token IDs changed")
    if any(a<0 or b<a or b>len(full) for a,b in offsets):
        raise ValueError("S4 invalid tokenizer codepoint offsets")
    words=[]
    used={}
    for w in row.words:
        a,b=start_user+w.start,start_user+w.end
        positions=tuple(i for i,(l,r) in enumerate(offsets)
                        if l<b and r>a)
        if not positions or any(offsets[i][0]==offsets[i][1] for i in positions):
            raise ValueError("S4 missing or special-token-only word span")
        for pos in positions:
            if pos in used and used[pos]!=w.word_id:
                raise ValueError("S4 tokenizer merged distinct UD words")
            used[pos]=w.word_id
        for char in range(a,b):
            if not any(offsets[p][0]<=char<offsets[p][1]
                       for p in positions):
                raise ValueError("S4 tokenizer offsets incompletely cover word")
        words.append(AlignedWord(w.word_id,positions,max(positions)))
    if (len(words)!=len(row.words)
            or len({w.last_token for w in words})!=len(words)
            or not all(words[i].last_token<words[i+1].last_token
                       for i in range(len(words)-1))
            or max(w.last_token for w in words)>=len(ids)):
        raise ValueError("S4 UD word-to-token ordering invalid")
    return CausalAlignment(
        original_prompt_sha256=row.sha256(),tokenizer_token_count=len(ids),
        prefill_token_ids=ids,word_alignment=tuple(words),
        complete_prefix_position=len(ids)-1,
    )


@dataclass
class CompletePrefixScores:
    upos_logits:torch.Tensor    # [W, UPOS]
    morph_logits:torch.Tensor   # [W, MORPH]
    dep_head_logits:torch.Tensor # [W, W+1] right heads allowed, + root
    semantic_role_logits:torch.Tensor # [W, roles]
    relation_summary:torch.Tensor # [relation_width]


def complete_prefix_readout(area, features,
                            aligned:CausalAlignment)->CompletePrefixScores:
    """Pure re-readout from *internal* existing NRA3 heads, no new adapter.

    Reconstruct each earlier word's context by adding the current final
    token state. This cannot operate on unseen future tokens and does NOT
    make the earlier prefix-local logits magically bidirectional.
    """
    packed=(type(area).__name__=="NRA3NativePackedCPU"
            and getattr(area,"NRA3_NATIVE_CPU_PACKED",False) is True
            and getattr(area,"native_cpu_weights_dense_resident",True) is False)
    if (not isinstance(area,NRA3NeuralArea) and not packed):
        raise ValueError("S4 only accepts the unified NRA3 dense TRAIN or native packed CPU area")
    if (not hasattr(features,"grammar")
            or features.grammar.ndim!=3 or features.grammar.shape[0]!=1
            or features.grammar.shape[1]!=aligned.tokenizer_token_count
            or features.semantics.shape[:2]!=features.grammar.shape[:2]
            or features.relations.shape[:2]!=features.grammar.shape[:2]):
        raise ValueError("S4 complete-prefix NRA3 actual feature ABI mismatch")
    if aligned.independent_eval_labels_loaded:
        raise ValueError("S4 prohibited EVAL input")
    if (not aligned.word_alignment or
            any(type(w.last_token) is not int or
                w.last_token<0 or w.last_token>=aligned.tokenizer_token_count
                for w in aligned.word_alignment)):
        raise ValueError("NRA3 S4 requires one integer completed-prefix token per UD word")
    ix=torch.tensor([w.last_token for w in aligned.word_alignment],
                    dtype=torch.long,device=features.grammar.device)
    end=aligned.complete_prefix_position
    if end!=features.grammar.shape[1]-1 or (ix>end).any():
        raise ValueError("S4 incomplete or future-dependent prefix")
    grammar_end=features.grammar[:,end,:]
    semantic_end=features.semantics[:,end,:]
    # Every decoded word can now attend to the summary of CURRENT visible
    # text. Per-word features and final prefix both originate from causal
    # NRA3 internal layers, no explicit train label is injected here.
    gram=features.grammar[0,ix,:]+grammar_end
    sem=features.semantics[0,ix,:]+semantic_end
    if packed:
        # Same existing physically packed 5-bit heads; no extra decoder
        # weights and no hidden dense matrix reconstruction.
        upos=area.linear("upos_head",gram)
        morph=area.linear("morph_head",gram)
        roles=area.linear("role_head",sem)
        q=area.linear("dep_q",gram)
        k=area.linear("dep_k",gram)
        root=area.linear("dep_k",grammar_end)
    else:
        upos=area.upos_head(gram)
        morph=area.morph_head(gram)
        roles=area.role_head(sem)
        q=area.dep_q(gram)
        k=area.dep_k(gram)
        root=area.dep_k(grammar_end)
    normal=(q@k.T)/math.sqrt(area.config.hidden)
    root_logit=(q@root.T)/math.sqrt(area.config.hidden)
    full=torch.cat((normal,root_logit),dim=-1)
    return CompletePrefixScores(upos,morph,full,roles,
                                features.relations[0,end,:])


@dataclass
class GrammarSemanticLoss:
    total:torch.Tensor
    upos:torch.Tensor
    morphology:torch.Tensor
    syntax:torch.Tensor
    semantic_roles:torch.Tensor


def train_only_multitask_loss(area:NRA3NeuralArea, hidden320:torch.Tensor,
                              row:RussianTrainSentence,
                              alignment:CausalAlignment,
                              *, feature_override=None)->GrammarSemanticLoss:
    """Differentiable pure TRAIN objective; no optimizer, no weight export.

    The caller must prove a separate frozen Calibri source+tokenizer SHA
    before any *real* optimization in a subsequent stage. Tests exercise
    the numerical contract with dummy hidden320 only.
    """
    if row.split!=TRAIN_SPLIT or alignment.original_prompt_sha256!=row.sha256():
        raise ValueError("S4 only admits locked Russian TRAIN source")
    if (hidden320.ndim!=3 or hidden320.shape!=(1,alignment.tokenizer_token_count,320)
            or hidden320.dtype!=torch.float32 or not torch.isfinite(hidden320).all()):
        raise ValueError("S4 missing genuine finite hidden320 prefix")
    # S7 may reuse the same causal features for BOTH grammar and relation
    # supervision; no second full 6M forward or future-only tokens.
    features=area(hidden320) if feature_override is None else feature_override
    predictions=complete_prefix_readout(area,features,alignment)
    ud=torch.tensor([UPOS.index(w.upos) for w in row.words],
                    device=hidden320.device)
    morph=torch.zeros((len(row.words),len(MORPH)),device=hidden320.device)
    for i,w in enumerate(row.words):
        for feature in w.feats:
            if feature not in MORPH:
                raise ValueError("S4 unsupported morphological training label")
            morph[i,MORPH.index(feature)]=1.
    heads=torch.tensor(
        [w.head-1 if w.head!=0 else len(row.words) for w in row.words],
        dtype=torch.long,device=hidden320.device)
    roles=torch.tensor([SEMANTIC_ROLES.index(v) for v in row.roles],
                       dtype=torch.long,device=hidden320.device)
    upos_loss=F.cross_entropy(predictions.upos_logits,ud)
    morph_loss=F.binary_cross_entropy_with_logits(predictions.morph_logits,morph)
    head_loss=F.cross_entropy(predictions.dep_head_logits,heads)
    sem_loss=F.cross_entropy(predictions.semantic_role_logits,roles)
    total=(upos_loss+morph_loss+head_loss+sem_loss)/4.
    if not bool(torch.isfinite(total)):
        raise AssertionError("S4 nonfinite Russian TRAIN objective")
    return GrammarSemanticLoss(total,upos_loss,morph_loss,head_loss,sem_loss)
