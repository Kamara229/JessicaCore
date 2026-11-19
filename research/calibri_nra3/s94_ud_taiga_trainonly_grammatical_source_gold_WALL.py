"""NRA3 S94: offline-only CC-BY-SA Russian UD treebank TRAIN acquisition.

Use true annotated Russian dependency relations to supervise neural
event->actor/patient/recipient/negation arcs INSIDE CALIBRI. The UD
treebank is a THIRD-PARTY OFFLINE TRAIN SOURCE, never a runtime
parser or extra neural model. Its TRAIN text and human-reviewed/
partially-auto tree dependencies are NOT a ground-truth event/recipient
QA benchmark; quoted speech, coreference, factuality, uncertainty and
long-term relation memory must remain UNKNOWN until independently
annotated.

PINNED upstream:
  UniversalDependencies/UD_Russian-Taiga
  commit 445ea57e47a3adae3146d0a42d2a8876214f2d20
  train-a Git blob c63aa8de90f2260e9681fde49d56d2c933366ef0
  LICENSE CC BY-SA 4.0, attribution + share-alike requirements.
  NO dev/test split, NO commercial/production use until license review.

Only 256 eligible Russian TRAIN sentences, with 4x64
morphosyntactic strata (active transitive, passive, negated,
other intransitive), deterministic by sentence order and strict
UTF-8 surface offsets; no label, POS, lemma, role, stratification
or correct answer enters source-only output or Calibri forward.
Constructed SOURCE and offline word-offset GOLD are separate.
Any original model training requires a FUTURE truly genuine 12-layer
native BPE offset conversion that fails closed on ambiguous positions.
Prior S79/S86 first-read and S83 synthetic DIAG are not reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

STAGE="CALIBRI_NRA3_S94_TAIGA_LICENSED_UD_TRAIN_ONLY_OFFLINE_SOURCE_GOLD_WALL"
UPSTREAM_REPO="UniversalDependencies/UD_Russian-Taiga"
UPSTREAM_COMMIT="445ea57e47a3adae3146d0a42d2a8876214f2d20"
TRAIN_FILE="ru_taiga-ud-train-a.conllu"
TRAIN_BLOB_SHA="c63aa8de90f2260e9681fde49d56d2c933366ef0"
LICENSE="CC BY-SA 4.0"
LIMIT_PER_STRATUM=64
STRATA=("active_transitive","passive","negated","other_intransitive")
AREA_COUNT=8
ROLE_ORDER=("actor","patient","recipient","negation_scope",
            "speech_source","time_anchor","coreference","event_relation")


def require(ok,why):
    if not bool(ok):
        raise ValueError("NRA3 S94 UD-Taiga TRAIN-only: "+why)


def sha256_serialized(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,
        sort_keys=True,separators=(",",":")).encode("utf8")).hexdigest()


def git_blob_sha(content:bytes):
    return hashlib.sha1(
        f"blob {len(content)}\0".encode("ascii")+content).hexdigest()


@dataclass(frozen=True)
class UDNode:
    index:int
    form:str
    lemma:str
    upos:str
    feats:str
    head:int
    deprel:str
    misc:str


def parse_conllu(input_string:str)->Iterator[tuple[str,str,tuple[UDNode,...]]]:
    """Strict CoNLL-U input, no test/dev splits, no prose guessing."""
    blocks=input_string.split("\n\n")
    for block in blocks:
        doc_id=""
        sent_id=""
        sentence=""
        words=[]
        for line in block.splitlines():
            if line.startswith("# sent_id = "):
                sent_id=line.split(" = ",1)[1].strip()
            if line.startswith("# text = "):
                sentence=line.split(" = ",1)[1]
            if not line or line.startswith("#"):
                continue
            fields=line.split("\t")
            if len(fields)!=10:
                words=[];break
            field_id=fields[0]
            if "-" in field_id or "." in field_id:
                # Multiword tokens/empty nodes: do not invent an offset.
                continue
            if not field_id.isdecimal() or not fields[6].isdecimal():
                words=[];break
            words.append(UDNode(int(field_id),fields[1],fields[2],
                                fields[3],fields[5],int(fields[6]),
                                fields[7],fields[9]))
        if not (sent_id and sentence and 5<=len(words)<=38
                and len({w.index for w in words})==len(words)
                and all(w.index==i+1 for i,w in enumerate(words))):
            continue
        yield sent_id,sentence,tuple(words)


def align_ud_surfaces_exact(text:str,nodes:tuple[UDNode,...]):
    """Fail closed on punctuation/spacing mismatch, no hidden gold in source."""
    cursor=0
    spans=[]
    for w in nodes:
        i=text.find(w.form,cursor)
        if i<0 or any(not x.isspace() for x in text[cursor:i]):
            return None
        spans.append((i,i+len(w.form)))
        cursor=i+len(w.form)
    if text[cursor:].strip():
        return None
    if any(not a<b<=len(text) for a,b in spans):
        return None
    return spans


def supervised_verb_candidate(nodes):
    """UD morphology/dependencies -- no asserted factuality claims."""
    by_parent={}
    for w in nodes:
        by_parent.setdefault(w.head,[]).append(w)
    for event in nodes:
        if event.upos!="VERB":
            continue
        kids=by_parent.get(event.index,[])
        by_role={}
        passive=("Voice=Pass" in event.feats or
                 any(w.deprel.startswith("nsubj:pass") for w in kids))
        for child in kids:
            dep=child.deprel
            if dep=="nsubj" and not passive:
                by_role.setdefault("actor",child.index-1)
            elif dep in ("nsubj:pass","csubj:pass"):
                by_role.setdefault("patient",child.index-1)
            elif dep=="nsubj" and passive:
                by_role.setdefault("patient",child.index-1)
            elif dep=="obj":
                by_role.setdefault("patient",child.index-1)
            elif dep=="iobj":
                by_role.setdefault("recipient",child.index-1)
            elif dep in ("obl:agent","obl:pass"):
                by_role.setdefault("actor",child.index-1)
            if child.lemma.casefold()=="не" and dep in (
                    "advmod","advmod:neg","discourse","mark"):
                by_role.setdefault("negation_scope",child.index-1)
        return event,by_role,passive
    return None


def class_of(event,roles,passive):
    if passive:
        return "passive"
    if "negation_scope" in roles:
        return "negated"
    if "actor" in roles and "patient" in roles:
        return "active_transitive"
    return "other_intransitive"


def select_locked_UD_train256(input_bytes:bytes,*,require_pinned_blob:bool=True):
    require(isinstance(input_bytes,bytes),"CoNLL-U must arrive as pinned UTF8 bytes")
    if require_pinned_blob:
        require(git_blob_sha(input_bytes)==TRAIN_BLOB_SHA,
                "UPSTREAM Taiga train-a original pinned Git blob SHA mismatch")
    text=input_bytes.decode("utf8")
    chosen={k:[] for k in STRATA}
    seen_texts=set()
    for sent_id,sentence,nodes in parse_conllu(text):
        if len(sentence)>500 or sentence in seen_texts:
            continue
        spans=align_ud_surfaces_exact(sentence,nodes)
        if spans is None:
            continue
        found=supervised_verb_candidate(nodes)
        if found is None:
            continue
        event,roles,passive=found
        bucket=class_of(event,roles,passive)
        if len(chosen[bucket])>=LIMIT_PER_STRATUM:
            continue
        if not any(k in roles for k in ("actor","patient","recipient")):
            continue
        if not (all(0<=v<len(nodes) for v in roles.values())
                and event.index-1<len(nodes)):
            continue
        seen_texts.add(sentence)
        chosen[bucket].append({
            "sentence":sentence,
            "ud_sent_id":sent_id,
            "ud_token_offsets":spans,
            "ud_event_token_index":event.index-1,
            "ud_roles":roles,
            "ud_passive":passive,
            "ud_bucket":bucket,
        })
        if all(len(chosen[k])==LIMIT_PER_STRATUM for k in STRATA):
            break
    require(all(len(chosen[k])==LIMIT_PER_STRATUM for k in STRATA),
            "not enough strictly aligned ORIGINAL Taiga TRAIN examples in every stratum")
    source=[];gold=[]
    for family in STRATA:
        for value in chosen[family]:
            rid=f"S94_TAIGA_TRAIN_{len(source)+1:03d}"
            source.append({"row_id":rid,"text":value["sentence"]})
            # Labels stay in a physically separate TRAIN-ONLY file;
            # all indexes are UD word indexes, NOT yet native BPE IDs.
            role_UD=[value["ud_roles"].get(k,-1) for k in ROLE_ORDER]
            gold.append({
                "row_id":rid,
                "ud_sentence_ID_offline_only":value["ud_sent_id"],
                "predicate_UD_word_zero_index_TRAIN_ONLY":
                    value["ud_event_token_index"],
                "role_argument_UD_word_zero_indices_TRAIN_ONLY":role_UD,
                "word_surface_start_end_char_offsets_TRAIN_ONLY":
                    value["ud_token_offsets"],
                "UD_voice_passive_TRAIN_ONLY":value["ud_passive"],
                "UD_negation_particle_link_present_TRAIN_ONLY":
                    "negation_scope" in value["ud_roles"],
                "grammar_supervision_bucket_TRAIN_ONLY":family,
                "reported_speech_source_and_coreference_GOLD_UNKNOWN":True,
                "not_independently_adjudicated_for_event_F1":True,
                "not_yet_original_native_BPE_coordinates":True,
            })
    require(len(source)==len(gold)==4*LIMIT_PER_STRATUM
            and all(set(r)=={"row_id","text"} for r in source)
            and len({z["text"] for z in source})==len(source)
            and [z["row_id"] for z in source]==[z["row_id"] for z in gold],
            "separate source-and-offline-UD grammar label hard wall")
    manifest={
        "stage":STAGE,
        "third_party_corpus":UPSTREAM_REPO,
        "upstream_immutable_git_commit":UPSTREAM_COMMIT,
        "training_only_original_git_blob":TRAIN_BLOB_SHA,
        "license":LICENSE,
        "license_notice_and_attribution_required":True,
        "share_alike_requirements_review_before_downstream_use":True,
        "no_production_or_commercial_use_until_license_review":True,
        "only_upstream_TRAIN_a_used_NO_upstream_dev_or_test":True,
        "n_training_rows":len(source),
        "stratification_counts":{k:len(chosen[k]) for k in STRATA},
        "source_only_sha256":sha256_serialized(source),
        "separately_supervised_only_gold_sha256":sha256_serialized(gold),
        "source_rows_have_no_label_or_UD_POS_or_role":True,
        "UD_grammar_not_natural_language_event_F1_gold":True,
        "UD_word_indices_REQUIRE_fail_closed_native_Calibri_BPE_alignment_later":True,
        "external_stanza_udpipe_not_required_by_NRA3_Calibri_inference":True,
        "old_S79_S86_spent_firstreads_and_S83_diag_NOT_imported":True,
        "original_43M_GT_616_J7_and_weights_not_touched_by_S94":True,
        "new_NRA3_training_steps":0,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return tuple(source),tuple(gold),manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--taiga-train-a",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    require(args.taiga_train_a.is_file() and not args.out.exists(),
            "no missing, mutable, or accidentally overwritten source")
    sources,labels,manifest=select_locked_UD_train256(
        args.taiga_train_a.read_bytes(),require_pinned_blob=True)
    args.out.mkdir(parents=True,exist_ok=False)
    for name,value in (
        ("NRA3_S94_TAIGA_TRAIN256_SOURCE_ONLY_NO_GOLD.json",sources),
        ("NRA3_S94_TAIGA_TRAIN256_OFFLINE_UD_GOLD_WORDINDEX_ONLY.json",labels),
        ("NRA3_S94_TAIGA_TRAIN256_LICENSE_PROVENANCE.json",manifest),
    ):
        (args.out/name).write_text(json.dumps(
            value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NRA3_S94_REAL_LICENSED_RUSSIAN_UD_TAIGA_TRAIN_ONLY",
          json.dumps({"count":len(sources),"strata":manifest["stratification_counts"],
                      "source_sha":manifest["source_only_sha256"],
                      "gold_sha":manifest["separately_supervised_only_gold_sha256"],
                      "release_commit":UPSTREAM_COMMIT,"output_gate":0},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
