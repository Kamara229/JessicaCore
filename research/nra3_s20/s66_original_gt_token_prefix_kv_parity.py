"""S66 gold-blind ORIGINAL Calibri43M GT native prefix + incremental KV parity.

This is a *falsification probe*, NOT a streaming safety certificate:
fixed full-text S93 profile/prompt-only bridge conditioning may itself
depend on future text, and same-prefix consistency of 2 hidden states
does not prove all 12 causal layers/token boundaries under arbitrary
generation. Measures actual original 12-layer model post-layer8 [320D]
on IDENTICAL native full-prompt token ID prefix and then 2 tokens via
KV cache. Uses frozen original GT in EXACT SAME S93 CPU phase.
No text rewriting, retokenization, model weights or gold labels.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import torch
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar

SUBSET=(0,4,8,12,16,20,24,28)  # two per preregistered S66 family; no gold
MAX_ROWS=8

def require(ok,why):
    if not bool(ok):raise ValueError("S66 original43M+GT read-only prefix/KV parity: "+why)

def stable_native_prefix_cut(words,token_ids):
    """Only source tokenizer word/native offsets; NO annotated predicate."""
    n=len(token_ids)
    require(n>9 and len(words)>=4,"original Calibri GT full input too short")
    options=[]
    for i,w in enumerate(words):
        ids=tuple(int(z) for z in w.native_token_positions)
        if not ids or not bool(w.causally_safe) or bool(w.shared_native_tokens):
            continue
        # Exact native token boundary at the end of the strictly safe
        # original token word; require >=2 subsequent native tokens.
        cut=max(ids)+1
        if 4<=cut<=n-3:
            options.append((abs(cut-n*.5),cut,i))
    require(options,"no S66 BPE-safe native mid-sentence cut; fail-closed")
    _,cut,index=min(options)
    return {"cut":cut,"word_index":index,"original_native_token_count":n,
            "prefix_selection_from_source_only":True}

def compare_real_native(full_hidden,prefix_hidden,kv_new_hidden,cut):
    require(full_hidden.ndim==2 and full_hidden.shape[1]==320
            and prefix_hidden.shape==(cut,320)
            and kv_new_hidden.shape==(2,320)
            and cut+2<=full_hidden.shape[0],
            "original physical post-layer8 GT hidden320 ABI")
    def vec(a,b):
        diff=(a.float()-b.float()).abs()
        return {"max_abs":float(diff.max().item()),
                "mean_abs":float(diff.mean().item()),
                "rms":float(diff.square().mean().sqrt().item()),
                "exact_fp16_all_320":bool(torch.equal(a,b)),
                "all_finite":bool(torch.isfinite(diff).all())}
    pre=vec(full_hidden[cut-1],prefix_hidden[cut-1])
    step=vec(full_hidden[cut:cut+2],kv_new_hidden)
    return {"prefill_hidden320_at_strict_word_end":pre,
            "KV_cache_incremental_hidden320_next2":step,
            "original_GT_full_context_future_source_not_masked_by_prompt_features":True,
            "same_bpe_token_ID_prefix_not_natural_retokenized_sentence":True,
            "cannot_certify_autoregressive_streaming_from_this_probe":True,
            "fail_closed_online_activation":True}

def forensic_source_row(row_id,text,cut_info,metrics):
    require(len(row_id)>0 and text and cut_info["prefix_selection_from_source_only"],
            "source-only audit record")
    return {
        "row_id":row_id,
        "raw_source_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest(),
        **cut_info,**metrics,
        "independent_human_gold_NEVER_LOADED_during_probe":True,
        "original_model_and_GT_weights_mutation":False,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
