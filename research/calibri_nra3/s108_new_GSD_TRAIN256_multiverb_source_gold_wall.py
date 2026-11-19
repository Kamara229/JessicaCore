"""NRA3 S108: precommit NEW Russian GSD TRAIN as RAW source and separate all-VERB UD GOLD.

S99 physically trained the IN-Calibri 28,887 parameter morphology +
NRA2 8 mini-area neural child on real original Calibri+GT all12.
MORPH transfer was measurable (UD UPOS 56% / Number 83% on same
corpus developer) but EVENT-ARGUMENT arcs overfit (86% TRAIN vs
16.7% same-corpus developer). S99 baseline is rejected gate0.

S108: DO NOT tune on S105/S102/S99/S96 already spent developer,
S79/S86 original firstreads. A new original TRAIN-only upstream file
UD_Russian-GSD official TRAIN (NOT Taiga train-a/b/c/d/e or GSD dev/test) supplies a wider
teacher: ALL VERB predicates and ALL unambiguous UD predicate→
argument arcs in each eligible sentence, not just ONE selected event.
Multiple argument words of the same role are retained separately.
Original word UPOS, Case, Number and Voice also remain available as
explicit OFFLINE teacher annotations. No invented negative labels.

train-c remains completely RESERVED as an optional prospective
cross-partition check, distinct from both training sources. Never
open train-c gold for training or tune on prospective first-read.

Corpora are third-party CC BY-SA 4.0. Attribution/share-alike/legal
clearance required before commercial or production use. The files
below are research outputs only. Original Calibri/GT/616 J7 untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    LICENSE,STRATA,
    git_blob_sha,parse_conllu,align_ud_surfaces_exact,
    select_locked_UD_train256,sha256_serialized,
)
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    UD_UPOS,UD_CASE,UD_NUMBER,UD_VOICE,
)
from research.calibri_nra3.s97_pin_Taiga_TRAINb_morph_source_vs_labels import one_feat
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES

STAGE="CALIBRI_NRA3_S108_NEW_GSD_TRAIN256_MULTIVERB_SOURCE_GOLD_PRECOMMITTED"
UPSTREAM_REPO="UniversalDependencies/UD_Russian-GSD"
UPSTREAM_COMMIT="71b92ce4c8da2b779450b0297abada721796425f"
TRAIN_FILE="ru_gsd-ud-train.conllu"
TRAIN_GIT_BLOB="de392eb8d1a92fe1d12bc2e49c21cb717ae89859"
SOURCE_FILE="NRA3_S108_GSD_TRAIN256_SOURCE_ONLY_NO_GOLD.json"
GOLD_FILE="NRA3_S108_GSD_TRAIN256_ALL_VERBS_MULTIEDGE_OFFLINE_UD_GOLD.json"
MANIFEST_FILE="NRA3_S108_GSD_TRAIN256_MULTIEVENT_SOURCE_GOLD_PROVENANCE.json"

MORPH_KEYS={
    "UPOS":UD_UPOS,"Case":UD_CASE,
    "Number":UD_NUMBER,"Voice":UD_VOICE,
}
NEGATOR_DEPRELS=frozenset(("advmod","advmod:neg","mark","discourse"))


def require(ok,reason):
    if not bool(ok):
        raise ValueError("NRA3 S108 Russian GSD train all-verbs UD offline source/GOLD wall: "+reason)


def all_ud_verbs_roles_with_multiedges(nodes):
    """OFFLINE UD training-only graph; NOT syntax at neural inference.

    Return list of all event words with ALL valid local dependency
    links, never selecting a single training-gold event among verbs.
    Multiple distinct arguments per role remain as multiple edges.
    """
    by_head={}
    for w in nodes:
        by_head.setdefault(w.head,[]).append(w)
    events=[]
    for w in nodes:
        if w.upos!="VERB":
            continue
        kids=by_head.get(w.index,())
        passive=("Voice=Pass" in w.feats
                 or any(z.deprel.startswith(("nsubj:pass","csubj:pass"))
                        for z in kids))
        edges=set()
        for z in kids:
            role=None
            if z.deprel in ("nsubj:pass","csubj:pass"):
                role="patient"
            elif z.deprel=="nsubj":
                role="patient" if passive else "actor"
            elif z.deprel=="obj":
                role="patient"
            elif z.deprel=="iobj":
                role="recipient"
            elif z.deprel in ("obl:agent","obl:pass"):
                role="actor"
            elif z.lemma.casefold()=="не" and z.deprel in NEGATOR_DEPRELS:
                role="negation_scope"
            elif z.deprel in ("ccomp","xcomp") and z.upos=="VERB":
                role="event_relation"
            if role is not None and z.index!=w.index:
                require(0<=z.index-1<len(nodes) and role in ROLE_NAMES,
                        "UD arg token cannot leave complete original word spans")
                edges.add((role,z.index-1))
        if edges:
            events.append({
                "predicate_UD_word_zero_index_TRAIN_ONLY":w.index-1,
                "UD_voice_passive_TRAIN_ONLY":bool(passive),
                "UD_negation_particle_link_present_TRAIN_ONLY":
                    any(z[0]=="negation_scope" for z in edges),
                "multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY":[
                    {"role":name,"arg_ud_word":i}
                    for name,i in sorted(edges,
                        key=lambda z:(ROLE_NAMES.index(z[0]),z[1]))
                ],
            })
    return events


def offline_allword_morphology(nodes):
    """Missing UD feature is NULL/UNKNOWN, never a negative class."""
    return {
        "offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY":[
            w.upos if w.upos in UD_UPOS else None for w in nodes],
        "offline_Case_ALL_UD_WORDS_TRAIN_ONLY":[
            one_feat(w.feats,"Case",UD_CASE) for w in nodes],
        "offline_Number_ALL_UD_WORDS_TRAIN_ONLY":[
            one_feat(w.feats,"Number",UD_NUMBER) for w in nodes],
        "offline_Voice_ALL_UD_WORDS_TRAIN_ONLY":[
            one_feat(w.feats,"Voice",UD_VOICE) for w in nodes],
    }


def precommit_real_gsd_train256(data:bytes):
    require(git_blob_sha(data)==TRAIN_GIT_BLOB,
            "immutable official Russian GSD TRAIN Git blob SHA mismatch")
    source0,chosen_first,source_manifest=select_locked_UD_train256(
        data,require_pinned_blob=False)
    # The source selection is predetermined by new GSD source+UD TRAIN-only
    # grammatical strata, NOT by any S99 diagnostic performance.
    source_to_node={}
    for sent_id,text,nodes in parse_conllu(data.decode("utf8")):
        if (sent_id,text) in source_to_node:
            continue
        spans=align_ud_surfaces_exact(text,nodes)
        if spans is not None:
            source_to_node[(sent_id,text)]=(nodes,spans)
    source=[]
    gold=[]
    counts=Counter()
    tag_counts=Counter()
    total_edges=0
    total_events=0
    for idx,(old,selected) in enumerate(zip(source0,chosen_first)):
        key=(selected["ud_sentence_ID_offline_only"],old["text"])
        require(key in source_to_node,
                "re-find exactly the same original UD GSD TRAIN sentence")
        nodes,spans=source_to_node[key]
        require([list(z) for z in spans]==
                [list(z) for z in selected["word_surface_start_end_char_offsets_TRAIN_ONLY"]],
                "rechecked original Unicode source byte-for-byte UD word offsets")
        events=all_ud_verbs_roles_with_multiedges(nodes)
        require(events and any(len(z["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"])
                               for z in events),
                "each selected GSD TRAIN sentence has at least one UD verb-role edge")
        rid=f"NRA3_S108_GSD_{idx+1:03d}"
        source.append({"row_id":rid,"text":old["text"]})
        morphology=offline_allword_morphology(nodes)
        for head,values in morphology.items():
            tag_counts[head]+=sum(v is not None for v in values)
        all_edges=sum(len(event["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"])
                      for event in events)
        total_events+=len(events)
        total_edges+=all_edges
        counts[selected["grammar_supervision_bucket_TRAIN_ONLY"]]+=1
        gold.append({
            "row_id":rid,
            "upstream_UD_sentence_ID_OFFLINE_ONLY":selected["ud_sentence_ID_offline_only"],
            "complete_original_source_UD_word_start_end_chars_TRAIN_ONLY":
                [list(z) for z in spans],
            "all_UD_VERB_predicates_and_all_unambiguous_multi_role_edges_TRAIN_ONLY":
                events,
            "reported_speech_factuality_coreference_not_adjudicated_UNKNOWN":True,
            "no_negative_event_labels_invented":True,
            "multiple_distinct_arguments_PER_role_NOT_collapsed":True,
            "original_Calibri_BPE_alignment_NOT_YET_known":True,
            "UD_GSD_not_real_human_QA_gold":True,
            **morphology,
        })
    require(len(source)==len(gold)==256
            and len({z["text"] for z in source})==256
            and all(set(x)=={"row_id","text"} for x in source)
            and all(a["row_id"]==b["row_id"] for a,b in zip(source,gold))
            and dict(counts)=={k:64 for k in STRATA}
            and total_events>=256 and total_edges>=300
            and all(tag_counts[k]>=30 for k in morphology),
            "strict balanced 256 real NEW GSD multi-event source+separate GOLD")
    manifest={
        "stage":STAGE,
        "license":LICENSE,
        "source_upstream":UPSTREAM_REPO,
        "immutable_upstream_commit":UPSTREAM_COMMIT,
        "immutable_real_Russian_GSD_TRAIN_git_blob":TRAIN_GIT_BLOB,
        "ONLY_upstream_GSD_train_USED_no_GSD_dev_test_or_Taiga_data":True,
        "S96_S99_S102_S105_developer_results_SPENT_not_used_for_training_selection":True,
        "old_S79_S86_firstread48_SPENT_not_used":True,
        "Taiga_TRAIN_c_REMAINS_RESERVED_unopened_for_future_firstread":True,
        "source_only_SHA256":sha256_serialized(source),
        "offline_all_verbs_morph_multiedge_GOLD_SHA256":sha256_serialized(gold),
        "n_new_GSD_train_source_rows":256,
        "fixed_stratification_counts":dict(counts),
        "n_offline_UD_all_verb_events":total_events,
        "n_offline_UD_multi_argument_role_edges":total_edges,
        "n_offline_UD_word_morph_tags":dict(tag_counts),
        "all_8_NRA2_neural_mini_areas_remain_in_Calibri":True,
        "same_original12_layer_route_and_neural_relational_memory":True,
        "source_ONLY_does_not_contain_any_UD_labels":True,
        "offline_ud_labels_NEVER_allowed_at_runtime":True,
        "gold_only_after_source_original_Calibri_BPE_SHA_pinned":True,
        "not_independent_naturalistic_factuality_QUALITY":True,
        "attribution_and_sharealike_license_review_before_any_commercial_use":True,
        "new_inmodel_NRA3_training_steps":0,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
    return source,gold,manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gsd-train",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    require(args.gsd_train.is_file() and not args.out.exists(),
            "must use freshly downloaded pinned licensed ORIGINAL Russian GSD TRAIN")
    src,gold,manifest=precommit_real_gsd_train256(args.gsd_train.read_bytes())
    args.out.mkdir(parents=True,exist_ok=False)
    for name,obj in ((SOURCE_FILE,src),(GOLD_FILE,gold),
                     (MANIFEST_FILE,manifest)):
        (args.out/name).write_text(
            json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
            encoding="utf8")
    print("NRA3_S108_REAL_RUSSIAN_GSD_TRAIN_MULTIVERB_UD_SOURCE_GOLD_PRECOMMIT",
          json.dumps({"source_SHA":manifest["source_only_SHA256"],
                      "gold_SHA":manifest["offline_all_verbs_morph_multiedge_GOLD_SHA256"],
                      "events":manifest["n_offline_UD_all_verb_events"],
                      "edges":manifest["n_offline_UD_multi_argument_role_edges"],
                      "morph":manifest["n_offline_UD_word_morph_tags"],
                      "license":"CC BY-SA 4.0","output_gate":0},
                     ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
