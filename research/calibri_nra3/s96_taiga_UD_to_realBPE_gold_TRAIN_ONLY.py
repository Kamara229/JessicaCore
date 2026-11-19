"""NRA3 S96: GOLD-ONLY pinned Taiga UD TRAIN -> real Calibri native BPE.

The ORIGINAL Calibri/GT 12-layer source-only S95 read and protected
616 J7 must have succeeded BEFORE this separate teacher-only module
is opened. Native source BPE positions come only from S95 frozen
source offsets, never guessed or retokenized. S79/S86 tests SPENT.
This is grammatical supervision, not event factuality or a live
knowledge base. CC BY-SA license review before commercial use.
"""
from __future__ import annotations

import hashlib
import json
import torch
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    N_ROLES,N_STATES,N_MICRO,STATE_NAMES,
)
from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    NRA3TrainOnlyTargets,IGNORE,UNKNOWN,
)

STAGE="CALIBRI_NRA3_S96_PINNED_TAIGA_UD_ONLY_TRAIN_NATIVE_BPE_GRAPH_GOLD"
S94_SOURCE_SHA="257fcc7518debb81e779c1419c34fdd9e565ad7e5ed9b04d08e55484fddba917"
S94_GOLD_SHA="f6717e81c4620e1da0744b6aaaf3a83647ad76c7ebeeb42722195b432b4f1880"
S95_PHYSICAL_SHA="S95_REAL12_PHYSICAL_SOURCE_SHA_TO_PIN_AFTER_SUCCESS"
N_SOURCE=256


def require(ok,why):
    if not bool(ok):
        raise ValueError("NRA3 S96 Taiga original Calibri BPE TRAIN gold: "+why)


def serialized_sha(value):
    return hashlib.sha256(json.dumps(
        value,ensure_ascii=False,sort_keys=True,
        separators=(",",":")).encode("utf8")).hexdigest()


def build_native_training_targets(source_rows, offline_gold, original):
    """Teacher enters ONLY this post-source step, never model.forward."""
    require(len(source_rows)==len(offline_gold)==N_SOURCE
            and serialized_sha(source_rows)==S94_SOURCE_SHA
            and serialized_sha(offline_gold)==S94_GOLD_SHA
            and all(set(x)=={"row_id","text"} for x in source_rows)
            and [x["row_id"] for x in source_rows]==[x["row_id"] for x in offline_gold],
            "SHA-pinned S94 TRAIN ONLY source and physically separate UD gold")
    names={
        "original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY",
        "original_native_BPE_token_ids",
        "native_BPE_source_valid_mask",
        "source_row_indices_in_PINNED_TAIGA_TRAIN256",
        "source_raw_word_char_start","source_raw_word_char_end",
        "source_raw_word_last_native_BPE",
        "source_raw_word_native_BPE_causally_safe",
    }
    require(set(original)==names,
            "only exact S95 source-only genuine original BPE/offset tensors")
    h=original["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    valid=original["native_BPE_source_valid_mask"]
    original_idx=original["source_row_indices_in_PINNED_TAIGA_TRAIN256"]
    starts=original["source_raw_word_char_start"]
    ends=original["source_raw_word_char_end"]
    native=original["source_raw_word_last_native_BPE"]
    safe=original["source_raw_word_native_BPE_causally_safe"]
    n,t=valid.shape
    require(h.shape==(n,12,t,320)
            and h.dtype==torch.float16
            and valid.dtype==torch.bool
            and starts.shape==ends.shape==native.shape==safe.shape
            and starts.shape[0]==n and safe.dtype==torch.bool
            and original_idx.shape==(n,)
            and original_idx.tolist()==sorted(set(original_idx.tolist()))
            and all(0<=x<N_SOURCE for x in original_idx.tolist())
            and bool(torch.isfinite(h).all()),
            "same genuine 12layer Calibri original native BPE shape and source offsets")
    mention=torch.full((n,t),IGNORE,dtype=torch.long)
    role=torch.full((n,N_ROLES,t),IGNORE,dtype=torch.long)
    state=torch.full((n,t,N_STATES),float(UNKNOWN),dtype=torch.float32)
    areas=torch.full((n,t,N_MICRO),float(UNKNOWN),dtype=torch.float32)
    usable=torch.zeros(n,dtype=torch.bool)
    counts={"unmatched_UD_raw_Unicode_char_span":0,
            "unsafe_shared_or_missing_native_BPE_piece":0,
            "no_valid_role_edges":0,"annotated_source_rows":0}
    role_links=0
    for row,original_row in enumerate(original_idx.tolist()):
        annotation=offline_gold[original_row]
        sentence=source_rows[original_row]["text"]
        require(annotation["row_id"]==source_rows[original_row]["row_id"]
                and annotation["not_yet_original_native_BPE_coordinates"]
                and annotation["reported_speech_source_and_coreference_GOLD_UNKNOWN"],
                "S94 offline word labels cannot be source model inputs")
        spans=annotation["word_surface_start_end_char_offsets_TRAIN_ONLY"]
        event_index=annotation["predicate_UD_word_zero_index_TRAIN_ONLY"]
        roles=annotation["role_argument_UD_word_zero_indices_TRAIN_ONLY"]
        require(len(roles)==N_ROLES
                and len(spans)>=5 and 0<=event_index<len(spans)
                and all(x==-1 or 0<=x<len(spans) for x in roles)
                and all(0<=a<b<=len(sentence) for a,b in spans),
                "original UD word index / character span errors")
        offsets={(a,b):j for j,(a,b) in
                 enumerate(zip(starts[row].tolist(),ends[row].tolist()))
                 if 0<=a<b}
        def native_for(ud_index):
            raw_word=offsets.get(tuple(spans[ud_index]))
            if raw_word is None:
                return None,"unmatched"
            if not bool(safe[row,raw_word]):
                return None,"unsafe"
            j=int(native[row,raw_word])
            if not (0<=j<t and bool(valid[row,j])):
                return None,"unsafe"
            return j,"ok"
        ev,reason=native_for(event_index)
        if ev is None:
            counts["unmatched_UD_raw_Unicode_char_span" if reason=="unmatched"
                   else "unsafe_shared_or_missing_native_BPE_piece"]+=1
            continue
        edges={}
        rejected=False
        for rid,udword in enumerate(roles):
            if udword<0:
                continue
            child,reason=native_for(udword)
            if child is None:
                rejected=True
                counts["unmatched_UD_raw_Unicode_char_span" if reason=="unmatched"
                       else "unsafe_shared_or_missing_native_BPE_piece"]+=1
                break
            require(child!=ev,"predicate cannot be its own UD argument")
            edges[rid]=child
        if rejected:
            continue
        if not edges:
            counts["no_valid_role_edges"]+=1
            continue
        usable[row]=True
        mention[row,ev]=1
        for rid,child in edges.items():
            role[row,rid,ev]=child
        # Negative means absence of THIS attached UD marker, not
        # absence of a real-world negated/passive event elsewhere.
        neg=int(annotation["UD_negation_particle_link_present_TRAIN_ONLY"])
        passive=int(annotation["UD_voice_passive_TRAIN_ONLY"])
        state[row,ev,STATE_NAMES.index("negated")]=float(neg)
        state[row,ev,STATE_NAMES.index("passive")]=float(passive)
        areas[row,ev,1]=float(neg)  # NRA2 negation mini-head
        areas[row,ev,2]=float(passive)  # NRA2 passive mini-head
        counts["annotated_source_rows"]+=1
        role_links+=len(edges)
    require(int(usable.sum())>=32 and role_links>=40,
            "fail closed: too few real original BPE-aligned UD TRAIN edges")
    # Stable internal split within SAME Taiga TRAIN-a only, not a
    # clean independent first-read. Never tune to S79/S86 spent sets.
    train=usable & ((original_idx.long()%5)!=0)
    diag=usable & ((original_idx.long()%5)==0)
    require(int(train.sum())>=24 and int(diag.sum())>=4
            and not bool((train&diag).any()),
            "internal source-only 80:20 split too small")
    targets=NRA3TrainOnlyTargets(
        mention=mention,role_argument_token=role,state=state,
        mini_area=areas,original_native_valid=valid.detach().clone(),
        evidence="offline_taiga_CC_BY_SA_train_a_UD_not_independent_event_truth",
    )
    return targets,train,diag,{
        "stage":STAGE,
        "pinned_Taiga_S94_source_sha256":S94_SOURCE_SHA,
        "pinned_Taiga_separate_UD_gold_sha256":S94_GOLD_SHA,
        "n_source_only_12layer_genuine_BPE_rows":n,
        "n_safe_UD_predicate_with_role_training_rows":int(usable.sum()),
        "n_native_BPE_training_role_edges":role_links,
        "n_internal_UD_training_rows":int(train.sum()),
        "n_same_Taiga_TRAIN_a_INTERNAL_diagnostic_rows":int(diag.sum()),
        "fail_closed_alignment_stats":counts,
        "unannotated_event_or_factuality_is_UNKNOWN_not_negative":True,
        "only_UD_attached_negation_and_passive_grammatical_targets":True,
        "held_firstread_S79_and_S86_never_imported_or_scored":True,
        "new_independently_annotated_firstread_NOT_run":True,
        "third_party_license":"CC BY-SA 4.0",
        "commercial_or_production_use_BLOCKED":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
