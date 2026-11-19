"""NRA3 S96: separate OFFLINE TRAIN UD word offsets -> ORIGINAL Calibri BPE.

S95 already physically ran protected original 616 J7 and captured all
12 genuine Calibri+GT layers for source-only licensed UD-Taiga TRAIN-a.
Only AFTER immutable S95 source tensor and checksum verification do
we open original S94 separate UD annotations here. No labels enter
the original Calibri forward or the embedded NRA3 model.forward.

UD word (character start/end) indices are mapped ONLY to exactly
matching RAW S95 original-language character spans, then to LAST
causally safe ORIGINAL Calibri BPE index. Ambiguous, unaligned, shared
BPE or multi-argument repeated same role => row excluded with reason,
never silently reindexed. There are no pseudo-negative EVENT or
unlabeled roles. Only true UD supervised actor/patient/recipient/
negator role slots are targets. Voice=Pass and attached negator mark
the two relevant old NRA2 mini-areas, other six are UNKNOWN.

Fixed in advance: for each original Taiga stratum of 64 consecutive
source rows, indices 0..47 TRAIN; 48..63 same-corpus internal
diagnostic ONLY, never use for hyperparams/checkpoint selection.
It is NOT an independent human-reviewed factuality or new syntax test.

No inference-time UD parsing, no S79 or S86 spent first-read reuse,
no base Calibri/GreenTunnel/S59/S65 changes, gate0, no commercial
use under CC BY-SA 4.0 pending full license review.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    NRA3TrainOnlyTargets,IGNORE,UNKNOWN,
)
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    ROLE_NAMES,STATE_NAMES,N_ROLES,N_STATES,N_MICRO,
)
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    UD_UPOS,UD_CASE,UD_NUMBER,UD_VOICE,
)
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    sha256_serialized,
)

STAGE="CALIBRI_NRA3_S99_STRICT_REAL12_TAIGA_TRAIN_B_ALLWORD_MORPH_AND_UD_ROLE_MAPPING"
S94_SOURCE_SHA="437b6d62564e4cda6c58975910ccc98221c3e81938bda22429418f16700b1dad"
S95_SOURCE_TENSOR="NRA3_S98_TAIGA_TRAINB256_REAL12_ORIGINAL_H320_NO_UD_GOLD.safetensors"
S95_REPORT="NRA3_S98_REAL43MGT_TAIGA_TRAINB256_ALL12_SOURCEONLY_PROVENANCE.json"
S94_GOLD="NRA3_S97_TAIGA_TRAINB256_OFFLINE_UD_ALLWORD_MORPH_GOLD.json"
S94_SOURCE="NRA3_S97_TAIGA_TRAINB256_SOURCE_ONLY_NO_GOLD.json"
S94_MANIFEST="NRA3_S97_TAIGA_TRAINB256_SOURCE_GOLD_LICENSE_PROVENANCE.json"
LABELS_NAME="NRA3_S99_TAIGA_TRAIN_B_REAL12_UD_AND_MORPH_TRAIN_ONLY_LABELS.safetensors"
REPORT_NAME="NRA3_S99_TAIGA_TRAIN_B_NATIVE_BPE_MORPH_LABEL_MAPPING_REPORT.json"


def require(condition,reason):
    if not bool(condition):
        raise ValueError("NRA3 S99 real12 original morphology offline UD mapping: "+reason)


@dataclass(frozen=True)
class AlignedS99:
    original_source_indices:torch.Tensor
    native_H8_original_word_last_indices:torch.Tensor
    targets:NRA3TrainOnlyTargets
    eligible:torch.Tensor
    train:torch.Tensor
    same_corpus_development:torch.Tensor
    row_exclusions:tuple[dict,...]
    n_mapped_roles:int
    upos:torch.Tensor
    case:torch.Tensor
    number:torch.Tensor
    voice:torch.Tensor
    n_morph_supervised:dict[str,int]


def map_UD_word_labels_ONLY_AFTER_source_features(
        *, source_rows:list,ud_labels:list, source_valid:torch.Tensor,
        source_row_ids:torch.Tensor,
        raw_char_starts:torch.Tensor,raw_char_ends:torch.Tensor,
        raw_last_original_BPE:torch.Tensor,raw_causal_safe:torch.Tensor,
        ud_gold_sha_expected:str) -> AlignedS99:
    """No source model invocation. Pure row-index and character mapping."""
    require(len(source_rows)==len(ud_labels)==256
            and sha256_serialized(source_rows)==S94_SOURCE_SHA
            and sha256_serialized(ud_labels)==ud_gold_sha_expected
            and all(set(x)=={"row_id","text"} for x in source_rows),
            "S97 NEW Taiga TRAIN-b source SHA and separate UD morphology labels only")
    n,t=source_valid.shape
    require(n>=1 and t>=2 and source_valid.dtype==torch.bool
            and source_row_ids.shape==(n,)
            and raw_char_starts.shape==raw_char_ends.shape==
                raw_last_original_BPE.shape==raw_causal_safe.shape
            and raw_char_starts.shape[0]==n
            and raw_causal_safe.dtype==torch.bool
            and source_row_ids.tolist()==sorted(set(source_row_ids.tolist()))
            and all(0<=int(x)<256 for x in source_row_ids),
            "real S95 original BPE source index order and raw offsets")
    mention=torch.full((n,t),IGNORE,dtype=torch.long)
    role=torch.full((n,N_ROLES,t),IGNORE,dtype=torch.long)
    state=torch.full((n,t,N_STATES),float(UNKNOWN),dtype=torch.float32)
    areas=torch.full((n,t,N_MICRO),float(UNKNOWN),dtype=torch.float32)
    eligible=torch.zeros(n,dtype=torch.bool)
    tr=torch.zeros(n,dtype=torch.bool)
    development=torch.zeros(n,dtype=torch.bool)
    skipped=[]
    roles_count=0
    morph={
        "upos":torch.full((n,t),IGNORE,dtype=torch.long),
        "case":torch.full((n,t),IGNORE,dtype=torch.long),
        "number":torch.full((n,t),IGNORE,dtype=torch.long),
        "voice":torch.full((n,t),IGNORE,dtype=torch.long),
    }
    classes={"upos":UD_UPOS,"case":UD_CASE,"number":UD_NUMBER,"voice":UD_VOICE}
    offline_field={
        "upos":"offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY",
        "case":"offline_Case_ALL_UD_WORDS_TRAIN_ONLY",
        "number":"offline_Number_ALL_UD_WORDS_TRAIN_ONLY",
        "voice":"offline_Voice_ALL_UD_WORDS_TRAIN_ONLY",
    }
    morph_counts={key:0 for key in morph}

    for batch,id_tensor in enumerate(source_row_ids):
        source_idx=int(id_tensor)
        src=source_rows[source_idx]
        gold=ud_labels[source_idx]
        reason=None
        try:
            require(gold["row_id"]==src["row_id"]
                    and gold["not_yet_original_native_BPE_coordinates"]
                    and gold["reported_speech_source_and_coreference_GOLD_UNKNOWN"]
                    and gold["not_independently_adjudicated_for_event_F1"],
                    "S94 gold exact source ID, not native/QA gold")
            spans=gold["word_surface_start_end_char_offsets_TRAIN_ONLY"]
            event_word=gold["predicate_UD_word_zero_index_TRAIN_ONLY"]
            arguments=gold["role_argument_UD_word_zero_indices_TRAIN_ONLY"]
            require(len(arguments)==N_ROLES and len(spans)>=3
                    and 0<=event_word<len(spans)
                    and all(len(z)==2 and 0<=z[0]<z[1]<=len(src["text"])
                            for z in spans),
                    "strict S94 UD word count and actual source Unicode spans")
            native_by_span={}
            for word in range(raw_char_starts.shape[1]):
                beg=int(raw_char_starts[batch,word])
                end=int(raw_char_ends[batch,word])
                if beg<0 or end<0:
                    continue
                slot=(beg,end)
                require(slot not in native_by_span,
                        "nonunique original raw source span")
                piece=int(raw_last_original_BPE[batch,word])
                if (bool(raw_causal_safe[batch,word])
                    and 0<=piece<t and bool(source_valid[batch,piece])):
                    native_by_span[slot]=piece
            def locate(ud_word:int):
                require(0<=ud_word<len(spans),"invalid UD word index")
                span=tuple(spans[ud_word])
                require(src["text"][span[0]:span[1]]!="",
                        "empty UD source span")
                require(span in native_by_span,
                        "UD word cannot map to one exact safe ORIGINAL BPE raw span")
                return native_by_span[span]
            ev=locate(event_word)
            edges=[]
            for role_id,index in enumerate(arguments):
                require(type(index) is int and index>=-1,
                        "role argument is original word zero-index or -1")
                if index>=0:
                    arg=locate(index)
                    require(arg!=ev,
                            "self-referential label requires separate reviewed evidence")
                    edges.append((role_id,arg))
            require(len(edges)>=1,
                    "at least one fully aligned labelled UD argument edge needed")
            require(len({(role_id,arg) for role_id,arg in edges})==len(edges),
                    "duplicate fixed UD role label not allowed")
            # Build morphology labels on FULL exact native-BPE-aligned
            # UD words (not just event and arguments). Missing/unknown
            # is -100 IGNORE, NOT the negative class or a fake grammar.
            row_morph={key:[] for key in morph}
            for label in morph:
                values=gold[offline_field[label]]
                require(len(values)==len(spans),
                        "S97 full morphology word-label count must match original source UD spans")
                for word_idx,value in enumerate(values):
                    if value is None:
                        continue
                    require(value in classes[label],
                            "unsupported UD feature must be explicit unknown, never guessed")
                    span=tuple(spans[word_idx])
                    if span not in native_by_span:
                        continue
                    pos=native_by_span[span]
                    row_morph[label].append((pos,classes[label].index(value)))
            # Labels are assigned ONLY after every validation of THIS row,
            # so failing rows do not contaminate outputs.
            for label,pairs in row_morph.items():
                for pos,value in pairs:
                    require(morph[label][batch,pos]==IGNORE,
                            "two offline UD word labels must not share one ORIGINAL BPE token")
                    morph[label][batch,pos]=value
                    morph_counts[label]+=1
            mention[batch,ev]=1
            for role_id,arg in edges:
                role[batch,role_id,ev]=arg
            if gold["UD_voice_passive_TRAIN_ONLY"]:
                state[batch,ev,STATE_NAMES.index("passive")]=1.
                areas[batch,ev,2]=1.  # NRA2 passive independent mini-area
            if gold["UD_negation_particle_link_present_TRAIN_ONLY"]:
                state[batch,ev,STATE_NAMES.index("negated")]=1.
                areas[batch,ev,1]=1.  # NRA2 negation mini-area
            eligible[batch]=True
            tr[batch]=source_idx%64<48
            development[batch]=source_idx%64>=48
            roles_count+=len(edges)
        except (KeyError,IndexError,TypeError,ValueError) as ex:
            reason=str(ex)[:200]
        if reason is not None:
            skipped.append({"original_source_index":source_idx,
                            "reason":reason})
    require(not bool((tr&development).any())
            and torch.equal(eligible,tr|development)
            and int(eligible.sum())+len(skipped)==n,
            "no training/development overlap and explicit alignment omissions")
    return AlignedS99(
        original_source_indices=source_row_ids,
        native_H8_original_word_last_indices=raw_last_original_BPE,
        targets=NRA3TrainOnlyTargets(
            mention=mention,role_argument_token=role,
            state=state,mini_area=areas,
            original_native_valid=source_valid.clone(),
            evidence="licensed_UD_Taiga_TRAIN_only_offline_teacher_not_independent_human_QA",
        ),eligible=eligible,train=tr,
        same_corpus_development=development,
        row_exclusions=tuple(skipped),n_mapped_roles=roles_count,
        upos=morph["upos"],case=morph["case"],
        number=morph["number"],voice=morph["voice"],
        n_morph_supervised=morph_counts,
    )


def physical_source_then_separate_gold(path_s95:Path,path_report_s95:Path,
                                       s94_dir:Path) -> tuple[dict,AlignedS96,dict]:
    import hashlib
    require(path_s95.name==S95_SOURCE_TENSOR and
            path_report_s95.name==S95_REPORT,
            "physical genuine frozen S95 source artifact paths only")
    r=json.loads(path_report_s95.read_text())
    require(r["stage"]=="CALIBRI_NRA3_S98_REAL43MGT_ALL12_TAIGA_TRAINB256_MORPH_SOURCE_ONLY_AFTER616"
            and r["S97_TAIGA_TRAIN_B_PINNED_SOURCE256_SHA256"]==S94_SOURCE_SHA
            and r["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and r["source_only_no_Taiga_UD_gold_loaded_during_original_CPU"]
            and r["NRA3_morph_neural_28887_weights_REGISTERED_INSIDE_actual_Calibri_model"]
            and r["all_original_43M_GT_CB59_params_sha_pre_and_post_identical"]
            and r["new_NRA3_train_steps"]==0
            and r["old_S79_S86_and_S96_internal_dev_NEVER_reused"]
            and r["no_commercial_or_production_use_before_legal_review"]
            and not r["candidate_accepted"],
            "genuine original 12-layer native source must be finalized BEFORE gold")
    require(hashlib.sha256(path_s95.read_bytes()).hexdigest()==
            r["ALL12_CPU_SAFETENSORS_SHA256"],
            "true S95 12layer physical SHA pinned before separate annotation read")
    old=load_file(str(path_s95),device="cpu")
    # Protected gold files accessed only AFTER successful physically
    # verified frozen source and all-original weight checks above.
    source=json.loads((s94_dir/S94_SOURCE).read_text())
    gold=json.loads((s94_dir/S94_GOLD).read_text())
    manifest=json.loads((s94_dir/S94_MANIFEST).read_text())
    require(manifest["n_training_rows"]==256
            and manifest["license"]=="CC BY-SA 4.0"
            and manifest["strict_SOURCE_only_sha256"]==S94_SOURCE_SHA
            and manifest["TRAIN_B_genuine_pinned_git_blob_sha"]==
                 "3b0f6947bd0d6ac022d91d5fb17269450a876bb3"
            and manifest["ONLY_original_Taiga_TRAIN_b_no_train_a_dev_test"]
            and manifest["commercial_production_use_forbidden_until_legal_review"],
            "pinned licensed training-only annotation provenance")
    alignment=map_UD_word_labels_ONLY_AFTER_source_features(
        source_rows=source,ud_labels=gold,
        source_valid=old["native_BPE_source_valid_mask"],
        source_row_ids=old["source_row_indices_in_PINNED_TAIGA_TRAIN256"].long(),
        raw_char_starts=old["source_raw_word_char_start"],
        raw_char_ends=old["source_raw_word_char_end"],
        raw_last_original_BPE=old["source_raw_word_last_native_BPE"],
        raw_causal_safe=old["source_raw_word_native_BPE_causally_safe"],
        ud_gold_sha_expected=manifest["strict_separate_offline_MORPH_and_role_GOLD_sha256"],
    )
    counts={}
    for group in range(4):
        sr=alignment.original_source_indices.long()
        sel=(sr>=64*group)&(sr<64*(group+1))
        counts[str(group)]={
            "eligible_train":int((sel&alignment.train).sum()),
            "eligible_internal_dev":int((sel&alignment.same_corpus_development).sum()),
        }
    require(len(alignment.row_exclusions)<=64
            and all(z["eligible_train"]>=24
                    and z["eligible_internal_dev"]>=7
                    for z in counts.values())
            and alignment.n_mapped_roles>=140
            and alignment.n_morph_supervised["upos"]>=1000
            and alignment.n_morph_supervised["case"]>=200
            and alignment.n_morph_supervised["number"]>=250
            and alignment.n_morph_supervised["voice"]>=30,
            "insufficient exact licensed UD→original BPE TRAIN label coverage")
    audit={
        "stage":STAGE,
        "S94_SOURCE_SHA256":S94_SOURCE_SHA,
        "S95_physical_12layer_original_BPE_SHA256":r["ALL12_CPU_SAFETENSORS_SHA256"],
        "S94_separate_gold_SHA256":manifest["strict_separate_offline_MORPH_and_role_GOLD_sha256"],
        "licensed_CC_BY_SA_4_with_production_use_BLOCKED":True,
        "n_original_Taiga_TRAIN256":256,
        "n_physical_S95_original_source_rows":int(alignment.eligible.numel()),
        "n_native_BPE_mapping_eligible":int(alignment.eligible.sum()),
        "n_native_BPE_mapping_excluded":len(alignment.row_exclusions),
        "per_stratum_train_and_same_corpus_diag_coverage":counts,
        "n_annotated_event_argument_relations":alignment.n_mapped_roles,
        "n_annotated_morphology_originalBPE_by_head":alignment.n_morph_supervised,
        "original_train_strata_indices_0_47_dev_48_63_predeclared":True,
        "source_model_forwards_all_gold_blind_before_label_open":True,
        "unannotated_tokens_roles_states_and_6_miniareas_UNKNOWN":True,
        "source_S79_S86_spent_TEST48_not_loaded":True,
        "S96_old_Taiga_TRAIN_a_internal_development_NOT_used":True,
        "original_Calibri_616_J7_still_protected":True,
        "new_28_887_morph_neural_weights_IN_model_not_standalone":True,
        "not_independent_human_factuality_benchmark":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return old,alignment,audit
