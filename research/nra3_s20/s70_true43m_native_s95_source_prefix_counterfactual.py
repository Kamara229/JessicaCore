"""S70: true original Calibri43M CPU *S95 full vs source-prefix* conditioning A/B.

The S67/S69 physical forward kept original FULL-text S95 prompt features
fixed even when native input tokens were truncated to an immutable BPE-safe
prefix. S68 exact S95 source algorithm surrogate demonstrated feature future
sensitivity but used *non-native surrogate tokens* and no physical H8.
S70 uses ORIGINAL S160/S95 REAL native BPE tokenizer for BOTH source
feature vectors and the exact SAME original model native prefix token IDs
in two physically frozen read-only model forwards.

No inference uses the S66 previously spent held GOLD; no training, adapter
forward, checkpoint selection, or acceptance occurs. A difference in H8
shows original bridge can encode a future-dependent S95 vector at the same
prefix, but is NOT an online leakage incidence estimate, proof that
ordinary autoregressive generation is broken, or a safety certification.
"""
from __future__ import annotations

import hashlib
import torch
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear
from research.nra3_s20.s67_original_cache_phase_isolation import (
    delta, require,
)

def source_prefix_from_safe_cut(original_source,raw_words,word_index,token_ids,cut):
    """Text-only safe source prefix tied to already verified native BPE cut."""
    require(isinstance(original_source,str) and original_source
            and type(word_index) is int and 0<=word_index<len(raw_words)
            and isinstance(cut,int) and 3<cut<len(token_ids)-2,
            "S70 BPE-safe source-only cut")
    word=raw_words[word_index]
    require(bool(getattr(word,"causally_safe",False))
            and not getattr(word,"shared_native_tokens",())
            and max(int(x) for x in word.native_token_positions)+1==cut,
            "S70 original BPE-safe word boundary consistency")
    end=int(word.end)
    require(4<=end<len(original_source)-2 and original_source[:end].strip(),
            "S70 original raw UTF8 prefix outside source")
    return original_source[:end]


def physical_unchanged_prefix_forward(model,sources,bridge,expansion,tunnel,
                                      native_prefix_ids,feature_vector):
    """One frozen original true 12-layer model forward, post-GT physical H8."""
    require(not model.training and len(model.model.layers)==12
            and next(model.parameters()).device.type=="cpu"
            and all(type(i) is int and i>=0 for i in native_prefix_ids)
            and len(native_prefix_ids)>3
            and isinstance(feature_vector,torch.Tensor)
            and feature_vector.shape==(512,)
            and bool(torch.isfinite(feature_vector).all())
            and not feature_vector.requires_grad
            and not expansion._active and not tunnel._active,
            "S70 genuine real original 43M frozen CPU native prefix ABI")
    observed=[]
    def hook(_layer,_args,output):
        h=output[0] if isinstance(output,tuple) else output
        require(isinstance(h,torch.Tensor)
                and h.shape==(1,len(native_prefix_ids),320)
                and bool(torch.isfinite(h).all()),"S70 post-layer8 hidden320")
        observed.append(h.detach()[0,-1].to(torch.float16).contiguous().clone())
        return None
    handle=model.model.layers[7].register_forward_hook(hook)
    try:
        with (torch.inference_mode(),
              expansion.prefill_only(model),
              s160.prefill_only_tunnel(model,tunnel)):
            with s160.frozen_cpu_phase_scope(
                    model,bridge,"S93",feature_vector,phase="prefill_only"):
                clear(sources)
                out=model(input_ids=torch.tensor([native_prefix_ids],
                              dtype=torch.long),past_key_values=None,
                          use_cache=True,output_hidden_states=True)
                logits=out.logits[0,-1].detach().clone()
                require(out.past_key_values is not None
                        and logits.numel()>256,"S70 frozen logits/KV physical")
                del out
                clear(sources)
    finally:
        handle.remove()
    require(len(observed)==1 and bool(torch.isfinite(logits).all()),
            "S70 exactly one native layer8 hook")
    return observed[0],logits


def audited_source_row(model,sources,bridge,expansion,tunnel,tok,
                       source,source_words,word_index,native_ids,cut,
                       full_source_vector):
    """Read identical BPE prefix with three different original S95 source vectors."""
    prefix=source_prefix_from_safe_cut(source,source_words,word_index,
                                      native_ids,cut)
    original=s160.prompt_only_vector(tok,source)
    source_only=s160.prompt_only_vector(tok,prefix)
    counterfactual=s160.prompt_only_vector(
        tok,prefix+" Позже другая комиссия нашла совершенно новый документ.")
    vs=[
        torch.as_tensor(x,dtype=torch.float32).clone().detach()
        for x in (original,source_only,counterfactual)
    ]
    require(all(z.shape==(512,) and bool(torch.isfinite(z).all())
                for z in vs)
            and torch.equal(vs[0],full_source_vector),
            "S70 frozen original S95 whole-source vector unexpectedly changed")
    # Same EXACT native prefix token IDs each forward, only original
    # prompt-side S95 features differ: original/full, source prefix, other
    # unlabeled future suffix. No candidate selection on results.
    reads=[
        physical_unchanged_prefix_forward(
            model,sources,bridge,expansion,tunnel,list(native_ids[:cut]),vec)
        for vec in vs
    ]
    return {
        "native_token_cut":cut,"native_token_length":len(native_ids),
        "source_sha256":hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "original_native_ids_prefix_reused_bit_identical":True,
        "true_original_native_BPE_not_byte_surrogate":True,
        "original_S95_source_prefix_sha256":
            hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "features_original_full_vs_prefix":delta(vs[0],vs[1]),
        "features_prefix_vs_new_suffix":delta(vs[1],vs[2]),
        "same_native_prefix_H8_full_vs_prefix":delta(reads[0][0],reads[1][0]),
        "same_native_prefix_H8_prefix_vs_suffix":delta(reads[1][0],reads[2][0]),
        "same_native_prefix_logits_full_vs_prefix":delta(reads[0][1],reads[1][1]),
        "same_native_prefix_logits_prefix_vs_suffix":delta(reads[1][1],reads[2][1]),
        "S95_full_source_conditioning_may_expose_future_input":True,
        "normal_generation_safety_or_root_cause_proven":False,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }


def audit_rows(rows):
    require(len(rows)==8,"S70 exactly eight immutable S66 prior-spent native source rows")
    require(all(r["original_native_ids_prefix_reused_bit_identical"]
                and r["true_original_native_BPE_not_byte_surrogate"]
                and not r["normal_generation_safety_or_root_cause_proven"]
                and r["output_gate"]==0
                and r["candidate_accepted"] is False
                and r["activation_allowed"] is False for r in rows),
            "S70 no candidate/online certification")
    def count(key):
        return sum(x[key]["max_abs"]>1.e-6 for x in rows)
    return {
        "stage":"NRA3_S70_ORIGINAL43M_REAL_NATIVE_S95_FULLVS_PREFIX_AND_SUFFIX_COUNTERFACTUAL",
        "original_616_J7_BASELINE_replayed_before_capture":True,
        "frozen_43M_S95_GT_S132_weights_unchanged":True,
        "original_native_BPE_same_prefix_inputs_in_all_three_forwards":True,
        "S66_previously_spent_raw_only_no_GOLD":True,
        "real_H8_source_feature_conditioning_diagnostic_NOT_online_causality":True,
        "rows":rows,
        "source_feature_full_vs_prefix_changed":count("features_original_full_vs_prefix"),
        "source_feature_prefix_vs_suffix_changed":count("features_prefix_vs_new_suffix"),
        "same_native_H8_full_vs_prefix_changed":count("same_native_prefix_H8_full_vs_prefix"),
        "same_native_H8_prefix_vs_suffix_changed":count("same_native_prefix_H8_prefix_vs_suffix"),
        "same_native_logits_full_vs_prefix_changed":count("same_native_prefix_logits_full_vs_prefix"),
        "same_native_logits_prefix_vs_suffix_changed":count("same_native_prefix_logits_prefix_vs_suffix"),
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
        "online_causal_safety_certified":False,
    }
