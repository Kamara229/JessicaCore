"""S16 fixed directed valence source: only existing S9 Russian TRAIN72.

Each edge carries direction PREDICATE -> ARGUMENT and semantic role.
A surface object may precede its verb or vice versa: the graph edge is
created as soon as both original wordpieces exist, never earlier.
Agent, patient, recipient and source are distinguished. Source is not
automatically an AGENT; passive nsubj:pass is PATIENT.

No EVAL data is imported/compared, no model/hidden readout or training
in this source-only prereg. An S16b optimizer may later use this schema
on the original 72 TRAIN but not any S12c/S15c consumed EVAL.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from research.nra3_s9.new_ru_train72 import (
    build_s9_train72,S9_TRAIN72_CANONICAL_SHA256,S9_TRAIN_SPLIT,
)
from research.nra3_s16.paid_directed_valence import ROLES

S16_TRAIN_SOURCE_SHA="9843c35ee9aa75b8f5c9c963e441589dd2f0292e41ec7933971d5aa3dcd365d6" # pin after first SOURCE-only GitHub Actions
SPLIT="NRA3_S16_DIRECTED_PREDICATE_ARGUMENT_EXISTING_S9_TRAIN72_ONLY"


@dataclass(frozen=True)
class DirectedGoldLink:
    predicate_word:int   # 1-based, true VERB during TRAIN annotation only
    argument_word:int    # 1-based
    role:str             # AGENT/PATIENT/RECIPIENT/SOURCE
    current_word:int     # max(predicate_word,argument_word)
    prior_word:int       # min(predicate_word,argument_word)
    orientation:str      # predicate_now or argument_now


@dataclass(frozen=True)
class DirectedTrainRow:
    text:str
    text_sha256:str
    construction:str
    lexical_family:int
    links:tuple[DirectedGoldLink,...]
    word_count:int
    source_split:str=SPLIT


def build_directed_train72()->tuple[DirectedTrainRow,...]:
    original=build_s9_train72()
    if len(original)!=72:
        raise ValueError("S16 only immutable original 72 training sentences")
    registered=[]
    for row in original:
        if row.split!=S9_TRAIN_SPLIT:
            raise ValueError("S16 cannot consume heldout EVAL or other TRAIN split")
        verbs={w.word_id for w in row.words if w.upos=="VERB"}
        edges=[]
        for w,role in zip(row.words,row.roles):
            if role not in ROLES[1:] or w.word_id in verbs:
                continue
            if w.head not in verbs:
                # The core training objective is only direct governing verb.
                # Never invent role links from indirect/unobserved heads.
                continue
            pred=w.head;arg=w.word_id
            current=max(pred,arg);prior=min(pred,arg)
            edges.append(DirectedGoldLink(
                pred,arg,role,current,prior,
                "predicate_now" if pred>arg else "argument_now"))
        registered.append(DirectedTrainRow(
            row.text,row.sha256(),row.construction,row.lexical_family,
            tuple(sorted(edges,key=lambda x:(x.current_word,x.prior_word,x.role))),
            len(row.words)))
    result=tuple(registered)
    validate_train72(result)
    return result


def validate_train72(rows:tuple[DirectedTrainRow,...])->None:
    if type(rows) is not tuple or len(rows)!=72:
        raise ValueError("S16 source must contain exactly 72 existing TRAIN")
    if len({r.text_sha256 for r in rows})!=72:
        raise ValueError("S16 duplicate TRAIN source")
    if {r.lexical_family for r in rows}!=set(range(12)):
        raise ValueError("S16 cannot lose lexical triplets")
    if {r.construction for r in rows}!={x.construction for x in build_s9_train72()}:
        raise ValueError("S16 construction strata changed")
    original={r.sha256():r for r in build_s9_train72()}
    roles=set();orient=set();counts=0
    for r in rows:
        src=original.get(r.text_sha256)
        if (src is None or src.text!=r.text or src.construction!=r.construction
            or src.lexical_family!=r.lexical_family or r.source_split!=SPLIT
            or len(src.words)!=r.word_count or
            len(r.links)!=len(set(r.links))):
            raise ValueError("S16 source identity/UD graph changed")
        expected=[]
        verbs={w.word_id for w in src.words if w.upos=="VERB"}
        for w,role in zip(src.words,src.roles):
            if role in ROLES[1:] and w.word_id not in verbs and w.head in verbs:
                pred=w.head;arg=w.word_id
                expected.append(DirectedGoldLink(
                    pred,arg,role,max(pred,arg),min(pred,arg),
                    "predicate_now" if pred>arg else "argument_now"))
        if set(expected)!=set(r.links):
            raise ValueError("S16 directed predicate↔argument TRAIN labels drift")
        for x in r.links:
            if (x.role not in ROLES[1:] or
                x.predicate_word==x.argument_word or
                x.current_word!=max(x.predicate_word,x.argument_word)
                or x.prior_word!=min(x.predicate_word,x.argument_word)
                or x.prior_word>=x.current_word or
                x.orientation!=("predicate_now" if
                    x.predicate_word>x.argument_word else "argument_now")):
                raise ValueError("S16 future role or wrong predicate/argument direction")
            orient.add(x.orientation);roles.add(x.role);counts+=1
    if (counts<180 or set(ROLES[1:])!=roles or
            orient!={"predicate_now","argument_now"}):
        raise ValueError("S16 directed agent/patient/recipient/source training coverage absent")


def canonical_directed_train72(rows:tuple[DirectedTrainRow,...])->bytes:
    validate_train72(rows)
    source=[{
        "text_sha256":r.text_sha256,"text":r.text,
        "construction":r.construction,"lexical_family":r.lexical_family,
        "source_split":r.source_split,"word_count":r.word_count,
        "links":[{
            "predicate_word":x.predicate_word,
            "argument_word":x.argument_word,
            "role":x.role,
            "current_word":x.current_word,
            "prior_word":x.prior_word,
            "orientation":x.orientation
        } for x in r.links]
    } for r in rows]
    return json.dumps(source,sort_keys=True,
                      ensure_ascii=False,separators=(",",":")).encode("utf-8")


def sealed_directed_train72()->dict:
    rows=build_directed_train72()
    physical=canonical_directed_train72(rows)
    sha=hashlib.sha256(physical).hexdigest()
    if sha!=S16_TRAIN_SOURCE_SHA:
        raise ValueError("S16 TRAIN directed label source mutated after seal")
    roles={name:sum(link.role==name for r in rows for link in r.links)
           for name in ROLES[1:]}
    orientation={x:sum(link.orientation==x for r in rows for link in r.links)
                 for x in ("predicate_now","argument_now")}
    return {
        "stage":"NRA3_S16_SOURCE_ONLY_DIRECTED_AGENT_ACTION_OBJECT_S9_TRAIN72",
        "original_S9_canonical_72_TRAIN_sha256":S9_TRAIN72_CANONICAL_SHA256,
        "S16_complete_predicate_to_argument_gold_source_sha256":sha,
        "S16_source_rows":72,
        "S16_source_lexical_families":12,
        "S16_source_constructions":6,
        "S16_gold_directed_links":sum(len(r.links) for r in rows),
        "S16_gold_directed_role_counts":roles,
        "S16_gold_directed_orientation_counts":orientation,
        "S16_original_Calibri43M_GT_NRA_C_data_loaded":False,
        "S16_existing_train72_only":True,
        "S16_future_argument_unavailable_until_current_word":True,
        "S16_predicate_to_argument_direction_not_surface_word_order":True,
        "S16_passive_subject_PATIENT_not_AGENT":True,
        "S16_new_EVAL_scored":False,
        "S16_used_consumed_S15c_EVAL24":False,
        "S16_used_consumed_S12c_EVAL16":False,
        "S16_used_old_five_EVAL128_or_reserved_sixth_S3CI":False,
        "S16_optimizer_steps":0,
        "S16_no_LIVE_Jessica_chat_Render_Supabase_APK":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
