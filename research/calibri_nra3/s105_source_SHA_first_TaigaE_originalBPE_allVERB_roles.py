"""NRA3 S105: STRICT original Calibri BPE source-first Taiga TRAIN-e all-VERB UD target mapping.

S104 must physically replay original 616 J7 and SHA-lock genuine
Calibri CB59+GT ALL12 native BPE source first. ONLY THEN open the
separate S100 original Taiga TRAIN-e UD multi-event/multi-argument
gold and morphology. No gold passed to actual frozen original model
or in-Calibri NRA3 forward.

S100 selected 256 *NEW* CC BY-SA 4.0 UD-Taiga TRAIN-e sources;
TRAIN-a internal S96, TRAIN-b internal S99, S79/S86 firstreads and
entire TRAIN-c reserved prospective corpus may NEVER enter here.
Fixed source index 0..47/64 TRAIN, 48..63/64 internal developer.
No developing on these held samples in the same stage.

Targets use exact char-offset→last *causally safe* original BPE
piece, fail-closed for every source row if any annotated VERB or
any annotated predicate/argument edge is unsafe or ambiguous.
No fake event negatives or zero links for unannotated slots;
multiple positive arguments in one (predicate,role) remain distinct.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.calibri_nra3.s103_new_Taiga_TRAINe_allVERB_source_gold_wall import (
    SOURCE_FILE,GOLD_FILE,MANIFEST_FILE,
)
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    sha256_serialized,
)
from research.calibri_nra3.s91_neural_multilayer_event_graph import N_ROLES,ROLE_NAMES
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    UD_UPOS,UD_CASE,UD_NUMBER,UD_VOICE,
)
from research.calibri_nra3.s100_offline_allverb_multiargument_nativeBPE_train_loss import (
    OriginalBPEAllVerbsEdges,
)

STAGE="CALIBRI_NRA3_S105_SOURCE_SHA_FIRST_ALL_VERB_TAIGAE_UD_NATIVE_BPE_TRAIN_ONLY"
SOURCE_SHA="5fa1c234cf2bd53aee62eb8c4faba9a8ec116c0e70f7b8c5e2b13a11f4d574fc"
GOLD_SHA="6f0c05aa988c500693d088a4147159ad106b10601e388ee317690fab7570a2bb"
UPSTREAM_BLOB="8df5a8b13b934576aa50b7fb55f8a16cb0151003"
S104_SOURCE="NRA3_S104_TAIGA_TRAINE256_ORIGINAL12_H320_NO_UD_GOLD.safetensors"
S104_REPORT="NRA3_S104_REAL43MGT_TAIGA_TRAINE256_SOURCE_ONLY_PROVENANCE.json"
UNKNOWN=-100
MORPH={
    "upos":("offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY",UD_UPOS),
    "case":("offline_Case_ALL_UD_WORDS_TRAIN_ONLY",UD_CASE),
    "number":("offline_Number_ALL_UD_WORDS_TRAIN_ONLY",UD_NUMBER),
    "voice":("offline_Voice_ALL_UD_WORDS_TRAIN_ONLY",UD_VOICE),
}


def require(ok,reason):
    if not bool(ok):
        raise ValueError("NRA3 S105 genuine original-Calibri BPE source-FIRST gold wall: "+reason)


@dataclass(frozen=True)
class S105TrueNativeBPEAllVerbs:
    original_source_row_indices:torch.Tensor
    morphology:dict[str,torch.Tensor]
    role:OriginalBPEAllVerbsEdges
    train:torch.Tensor
    internal_dev:torch.Tensor
    eligible:torch.Tensor
    excluded:tuple[dict,...]
    counts:dict[str,int]


def map_UD_after_original_source_sha(*,source:list,gold:list,
      valid:torch.Tensor,original_indices:torch.Tensor,
      raw_start:torch.Tensor,raw_end:torch.Tensor,
      raw_final_BPE:torch.Tensor,raw_safe:torch.Tensor)->S105TrueNativeBPEAllVerbs:
    """Pure targets AFTER original model source; no source forward here."""
    require(len(source)==len(gold)==256
            and sha256_serialized(source)==SOURCE_SHA
            and sha256_serialized(gold)==GOLD_SHA
            and all(set(z)=={"row_id","text"} for z in source),
            "S100 SHA-frozen NEW TRAIN-e source and separate all-verb teacher gold")
    require(valid.dtype==torch.bool and valid.ndim==2,
            "exact source tokenizer native BPE bool valid")
    n,t=valid.shape
    require(original_indices.shape==(n,)
            and original_indices.tolist()==sorted(set(original_indices.tolist()))
            and all(0<=int(i)<256 for i in original_indices)
            and raw_start.shape==raw_end.shape==raw_final_BPE.shape==raw_safe.shape
            and raw_start.shape[0]==n and raw_safe.dtype==torch.bool,
            "S104 original real12 source geometry and ordered source rows")
    positive=torch.zeros((n,N_ROLES,t,t),dtype=torch.bool)
    mention=torch.zeros((n,t),dtype=torch.bool)
    morph={k:torch.full((n,t),UNKNOWN,dtype=torch.long) for k in MORPH}
    eligible=torch.zeros(n,dtype=torch.bool)
    train=torch.zeros(n,dtype=torch.bool)
    dev=torch.zeros(n,dtype=torch.bool)
    excluded=[]
    for batch,x in enumerate(original_indices.tolist()):
        src=source[x];g=gold[x]
        try:
            require(g["row_id"]==src["row_id"]
                    and g["no_negative_event_labels_invented"]
                    and g["multiple_distinct_arguments_PER_role_NOT_collapsed"]
                    and g["original_Calibri_BPE_alignment_NOT_YET_known"]
                    and g["reported_speech_factuality_coreference_not_adjudicated_UNKNOWN"],
                    "S100 source→gold row ID/known-unknown audit")
            spans=[tuple(z) for z in
                   g["complete_original_source_UD_word_start_end_chars_TRAIN_ONLY"]]
            require(len(spans)>=4 and
                    all(len(z)==2 and 0<=z[0]<z[1]<=len(src["text"]) for z in spans)
                    and len(set(spans))==len(spans),
                    "strict original Unicode UD word spans and unique original text")
            by_span={}
            for word_id in range(raw_start.shape[1]):
                start=int(raw_start[batch,word_id])
                stop=int(raw_end[batch,word_id])
                if start<0 or stop<0:
                    continue
                span=(start,stop)
                require(span not in by_span,"duplicate raw source Unicode word span")
                j=int(raw_final_BPE[batch,word_id])
                if bool(raw_safe[batch,word_id]) and 0<=j<t and bool(valid[batch,j]):
                    by_span[span]=j
            # For all annotated role predicate/argument links every
            # position MUST align unambiguously, else skip WHOLE row.
            def position(ud_word_idx):
                require(type(ud_word_idx) is int and 0<=ud_word_idx<len(spans),
                        "original UD word index out of complete source bounds")
                span=spans[ud_word_idx]
                require(span in by_span,
                        "event or argument UD word not exact causally-safe original BPE")
                return by_span[span]
            events=g["all_UD_VERB_predicates_and_all_unambiguous_multi_role_edges_TRAIN_ONLY"]
            require(isinstance(events,list) and len(events)>0,
                    "all original UD verb events and role list required")
            row_edges=[]
            row_mentions=[]
            for event in events:
                pred=position(event["predicate_UD_word_zero_index_TRAIN_ONLY"])
                row_mentions.append(pred)
                arcs=event["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"]
                require(arcs,"every selected original verb needs annotated UD args")
                for arc in arcs:
                    rname=arc["role"]
                    require(rname in ROLE_NAMES,"known eight original NRA2 event roles")
                    arg=position(arc["arg_ud_word"])
                    require(pred!=arg,"self-dependency is invalid original UD tree")
                    row_edges.append((ROLE_NAMES.index(rname),pred,arg))
            require(len(row_edges)>0 and len(set(row_edges))==len(row_edges)
                    and len(set(row_mentions))==len(row_mentions),
                    "complete unique original UD verb/argument links")
            # Other words may be unsafe; skip their unknown morphology
            # (not an invented label or a row rejection).
            row_morph={}
            for name,(field,classes) in MORPH.items():
                tags=g[field]
                require(isinstance(tags,list) and len(tags)==len(spans),
                        "complete S100 original word morphology list")
                vals=[]
                for word,tag in enumerate(tags):
                    if tag is None:
                        continue
                    require(tag in classes,"unsupported complex UD tag must be UNKNOWN")
                    if spans[word] in by_span:
                        vals.append((by_span[spans[word]],classes.index(tag)))
                require(len(set(p for p,_ in vals))==len(vals),
                        "raw words cannot share SAME original safe BPE label index")
                row_morph[name]=vals
            # WRITE only after FULL row validated: fail closed.
            for pred in row_mentions:mention[batch,pred]=True
            for role,pred,arg in row_edges:positive[batch,role,pred,arg]=True
            for name,labels in row_morph.items():
                for word,label in labels:morph[name][batch,word]=label
            eligible[batch]=True
            train[batch]=(x%64)<48
            dev[batch]=(x%64)>=48
        except (KeyError,TypeError,IndexError,ValueError) as e:
            excluded.append({"original_source_idx":int(x),"reason":str(e)[:200]})
    supervised=positive.any(dim=-1)
    require(torch.equal(eligible,train|dev)
            and not bool((train&dev).any())
            and int(eligible.sum())+len(excluded)==n
            and not bool((positive&~(
                valid[:,None,:,None]&valid[:,None,None,:])).any())
            and torch.equal(supervised,positive.any(-1))
            and not bool((supervised&~mention[:,None,:]).any()),
            "all gold-positive original native BPE links and no faux negatives")
    counts={
        "source_accepted":n,
        "full_UD_BPE_rows":int(eligible.sum()),
        "TRAIN_only_source":int(train.sum()),
        "same_corpus_INTERNAL_dev":int(dev.sum()),
        "known_verb_predicate_BPE_positions":int(mention.sum()),
        "known_supervised_verb_role_slots":int(supervised.sum()),
        "positive_event_argument_edges":int(positive.sum()),
    }
    counts.update({f"known_{k}_native_BPE":int((v!=UNKNOWN).sum())
                   for k,v in morph.items()})
    return S105TrueNativeBPEAllVerbs(
        original_indices,morph,
        OriginalBPEAllVerbsEdges(valid,positive,supervised,mention),
        train,dev,eligible,tuple(excluded),counts)


def load_source_VERIFY_sha_THEN_OPEN_gold(
        true12_source:Path,source_report:Path,
        separate_S100_gold_folder:Path):
    """Order enforced: protected true12 source SHA BEFORE gold open."""
    require(true12_source.name==S104_SOURCE and
            source_report.name==S104_REPORT,
            "only sealed original genuine12 source physical artifact accepted")
    report=json.loads(source_report.read_text())
    require(report["stage"]==
            "CALIBRI_NRA3_S104_REAL43MGT_ALL12_TAIGA_TRAINE256_STRUCTURAL_SOURCE_ONLY_AFTER616"
            and report["S103_TAIGA_TRAIN_E_PINNED_SOURCE256_SHA256"]==SOURCE_SHA
            and report["S103_TAIGA_TRAIN_E_original_git_blob"]==UPSTREAM_BLOB
            and report["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and bool(report["all_original_43M_GT_CB59_params_sha_pre_and_post_identical"])
            and report["source_ONLY_no_S103_TRAINe_UD_multiverb_GOLD_loaded_in_original_CPU"]
            and report["NRA3_morph_neural_37087_weights_REGISTERED_INSIDE_actual_Calibri_model"]
            and report["S79_S86_S96_S99_S102_spent_diagnostics_NEVER_reused"]
            and report["output_gate"]==0
            and report["new_NRA3_train_steps"]==0
            and not report["candidate_accepted"],
            "S104 original616 protected full12 frozen SHA/goldblind verified")
    require(hashlib.sha256(true12_source.read_bytes()).hexdigest()==
            report["ALL12_CPU_SAFETENSORS_SHA256"],
            "true original12 CB59+GT native source SHA MUST match BEFORE UD gold read")
    source_tensor=load_file(str(true12_source),device="cpu")
    # No gold was touched above; only AFTER this point open separate
    # licensed S100 TRAIN-e UD grammar/morphology teacher.
    source=json.loads((separate_S100_gold_folder/SOURCE_FILE).read_text())
    labels=json.loads((separate_S100_gold_folder/GOLD_FILE).read_text())
    upstream=json.loads((separate_S100_gold_folder/MANIFEST_FILE).read_text())
    require(upstream["stage"]==
                "CALIBRI_NRA3_S103_ALL_UD_VERBS_MULTI_EDGE_TAIGA_TRAIN_E_PRECOMMITTED"
            and upstream["source_only_SHA256"]==SOURCE_SHA
            and upstream["offline_all_verbs_morph_multiedge_GOLD_SHA256"]==GOLD_SHA
            and upstream["immutable_real_Taiga_TRAIND_git_blob"]==UPSTREAM_BLOB
            and upstream["ONLY_upstream_train_e_USED_no_train_a_b_c_d_dev_test"]
            and upstream["Taiga_TRAIN_c_REMAINS_RESERVED_unopened_for_future_firstread"]
            and upstream["n_offline_UD_all_verb_events"]==361
            and upstream["n_offline_UD_multi_argument_role_edges"]==600
            and upstream["license"]=="CC BY-SA 4.0",
            "pinned licensed S100 gold SHA and reserved prospective TRAIN-c")
    aligned=map_UD_after_original_source_sha(
        source=source,gold=labels,
        valid=source_tensor["native_BPE_source_valid_mask"],
        original_indices=source_tensor["source_row_indices_in_PINNED_TAIGA_TRAIN256"].long(),
        raw_start=source_tensor["source_raw_word_char_start"],
        raw_end=source_tensor["source_raw_word_char_end"],
        raw_final_BPE=source_tensor["source_raw_word_last_native_BPE"],
        raw_safe=source_tensor["source_raw_word_native_BPE_causally_safe"],
    )
    # Preserve at least half of precommitted real source. Report all
    # exclusions and TRAIN/DEV per-stratum counts, no label peeking
    # to reselect source rows or adjust thresholds.
    require(aligned.counts["full_UD_BPE_rows"]>=128
            and aligned.counts["TRAIN_only_source"]>=96
            and aligned.counts["same_corpus_INTERNAL_dev"]>=16
            and aligned.counts["positive_event_argument_edges"]>=200
            and all(int(((aligned.original_source_row_indices>=64*i)&
                         (aligned.original_source_row_indices<64*(i+1))&
                         aligned.train).sum())>=16 for i in range(4)),
            "minimal strict S104 genuine native BPE TRAIN-e ALLVERB coverage")
    audit={
        "stage":STAGE,
        "physical_original12_S104_SHA":report["ALL12_CPU_SAFETENSORS_SHA256"],
        "original_S103_source_sha":SOURCE_SHA,
        "separate_S103_all_verbs_GOLD_sha":GOLD_SHA,
        "new_Taiga_TRAIN_e_original_git_blob":UPSTREAM_BLOB,
        "source_first_no_teacher_in_actual_calibri":True,
        "original616_protected":True,
        "new_NRA3_morph_pair_IN_CALIBRI_child_37087":True,
        "eight_NRA2_mini_areas_preserved":True,
        "all12_layer_original_Calibri_GT_hidden_frozen":True,
        "per_original_source_stratum":{
            str(i):{
                "TRAIN":int(((aligned.original_source_row_indices>=64*i)&
                             (aligned.original_source_row_indices<64*(i+1))&
                             aligned.train).sum()),
                "INTERNAL_DEV":int(((aligned.original_source_row_indices>=64*i)&
                                    (aligned.original_source_row_indices<64*(i+1))&
                                    aligned.internal_dev).sum())
            } for i in range(4)},
        "counts":aligned.counts,
        "row_exclusions":aligned.excluded,
        "TRAIN_c_reserved_NOT_opened":True,
        "S96_S99_developer_and_S79_S86_firstreads_spent_not_loaded":True,
        "third_party_CC_BY_SA_license_requires_legal_review":True,
        "not_independent_grammar_event_or_factuality_evidence":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return source_tensor,aligned,audit
