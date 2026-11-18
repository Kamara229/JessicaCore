"""S49: strictly zero-gated quote-sidecar for PHYSICAL frozen Calibri43M+GT layer8 word338.

The original real 12-layer Calibri43M + Green Tunnel was physically
run on CPU in S22b, after protected 616 J7 output reproduction. S22c
persisted word vectors: first 320 are REAL CPU post-GT layer8 native
hidden; final 18 are text-only Russian surface cues. S49 consumes ONLY
the original SHA-checked S22c raw338 artifact; it does NOT load a model
checkpoint, rerun 616 J7, connect a runtime hook, or change model weights.

This is a REAL-native-feature *offline integration interface* with
original typed S48 Unicode quote parser as disconnected read-only
sidecar. It is NOT yet live 12-layer integration or Calibri43M NLP gain.
The old S48 4-parameter trained head depends on different S45 shadow
logits and must NOT be presented as calibrated for original 43M features.

At gate=0 the original tensor object is returned by reference,
bitwise unchanged. Any gate!=0 raises. No gradient, optimizer, gold,
protected S38, examined S47, separate EVAL or deployment.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

import torch
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s48_matched_unicode_quote_graph import (
    matched_quote_graph_from_raw,
)

REAL_NATIVE_FEATURES=338
REAL_LAYER8_HIDDEN=320
RAW18=18
ZERO_GATE=0


def require(condition,why):
    if not bool(condition):
        raise ValueError("S49 frozen TRUE43M GT zero-gate quote bridge: "+why)


@dataclass(frozen=True)
class ReadOnlyQuoteNativeFeatures:
    """Gold-free, non-injecting native 43M post-GT layer8 sidecar."""
    detached_original_hidden320:torch.Tensor
    text_only_quote_mask:torch.Tensor
    supported_mask:torch.Tensor
    per_row_tokens:tuple[int,...]
    per_row_quote_tokens:tuple[int,...]
    per_row_matched_pairs:tuple[int,...]
    per_row_unmatched_openers:tuple[int,...]
    unsupported_row_ids:tuple[int,...]
    # No model scores or positive gold accuracy fields.


class Frozen43MGreenTunnelQuotePort:
    """Original [256,W,338] remains exact; any candidate activation denied."""

    def __init__(self):
        self.output_gate=ZERO_GATE
        self.candidate_accepted=False
        self.activation_allowed=False
        self.training_allowed=False
        self.read_only=True
        self.native_source="S22b_TRUE43M_12_LAYER_POST_GT_LAYER8_REAL_CPU"
        self.original_layer_zero_based=7

    def _validate(self,raw338,valid,texts):
        require(self.output_gate==0 and not self.candidate_accepted
                and not self.activation_allowed and not self.training_allowed,
                "original 43M model may not be activated or trained")
        require(isinstance(raw338,torch.Tensor)
                and raw338.ndim==3 and raw338.shape[-1]==REAL_NATIVE_FEATURES
                and raw338.dtype==torch.float32 and raw338.device.type=="cpu"
                and not raw338.requires_grad and bool(torch.isfinite(raw338).all()),
                "requires physical detached REAL43M CPU word338")
        require(isinstance(valid,torch.Tensor)
                and valid.dtype==torch.bool and valid.device.type=="cpu"
                and valid.shape==raw338.shape[:2],
                "original native valid word mask mismatched")
        b,w=valid.shape
        require(isinstance(texts,tuple) and len(texts)==b
                and all(isinstance(s,str) and s for s in texts),
                "original sealed raw text tuple absent")
        require(bool(valid[:,0].all())
                and bool((valid.long().cummin(1).values==valid.long()).all())
                and bool((raw338[~valid]==0).all())
                and bool(((raw338[...,REAL_LAYER8_HIDDEN:]==0)
                           |(raw338[...,REAL_LAYER8_HIDDEN:]==1)).all()),
                "native word338 source or padding ABI drift")
        return b,w

    def observe(self,raw338,valid,texts):
        """Only compute text-derived quotes, never inject into original tensor."""
        batch,width=self._validate(raw338,valid,texts)
        phrase_mask=torch.zeros((batch,width),dtype=torch.float32)
        cover=torch.zeros((batch,width),dtype=torch.bool)
        counts=[]
        quote_counts=[]
        pairs=[]
        dangling=[]
        unsupported=[]
        for row,text in enumerate(texts):
            tokens=raw_surface_grammar(text)
            n=len(tokens)
            require(n==int(valid[row].sum().item()),
                    "native original word alignment to S48 Unicode lexer drift")
            # Verify original trailing raw18 exact rather than trusting features.
            grammar=torch.tensor([t.features for t in tokens],
                                 dtype=torch.float32)
            require(torch.equal(grammar,raw338[row,:n,320:]),
                    "real frozen original43M native word338 raw18 lex alignment drift")
            counts.append(n)
            if not 3<=n<=40:
                unsupported.append(row)
                quote_counts.append(0)
                pairs.append(0)
                dangling.append(0)
                # Fail closed: old original features are still identical.
                continue
            res=matched_quote_graph_from_raw(text)
            require(res.content.numel()==n,
                    "S48 typed quote structure word count drift")
            phrase_mask[row,:n]=res.content
            cover[row,:n]=True
            quote_counts.append(int(res.content.sum()))
            pairs.append(len(res.valid_outer_pairs))
            dangling.append(res.dangling_openers)
        require(bool((phrase_mask[~valid]==0).all())
                and bool((phrase_mask[~cover]==0).all()),
                "S49 sidecar quote features escaped original valid support")
        return ReadOnlyQuoteNativeFeatures(
            raw338[...,:REAL_LAYER8_HIDDEN].detach(),
            phrase_mask,
            cover,
            tuple(counts),tuple(quote_counts),tuple(pairs),
            tuple(dangling),tuple(unsupported))

    def forward_original_unchanged(self,original_word338,valid,texts):
        """One-sided gate contract: exact ORIGINAL Tensor identity, not clone."""
        self._validate(original_word338,valid,texts)
        # No add/mul/clamp/residual/write. Original input object is returned.
        require(self.output_gate==0,"nonzero quote sidecar gate forbidden")
        return original_word338

    def shadow_augmented_features_only(self,original_word338,valid,texts):
        """Disconnected research-only [real hidden320, raw18, S48 quote_bit].

        339-D is a new adapter ABI and MUST NEVER be passed as replacement
        for original 338-D core or paid graph logits without a separate
        model compatibility and quality/retention protocol.
        """
        self._validate(original_word338,valid,texts)
        aux=self.observe(original_word338,valid,texts)
        detached=original_word338.detach()
        extended=torch.cat([detached,aux.text_only_quote_mask.unsqueeze(-1)],dim=-1)
        require(extended.shape==(*original_word338.shape[:2],339)
                and torch.equal(extended[...,:338],original_word338)
                and extended.data_ptr()!=original_word338.data_ptr(),
                "S49 append-only disconnected adapter 339-D ABI drift")
        return extended,aux


def frozen_native_manifest_digest(raw338,valid):
    require(raw338.device.type=="cpu" and valid.device.type=="cpu",
            "S49 digest only CPU frozen physical tensors")
    h=hashlib.sha256()
    for tensor in (raw338,valid):
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
