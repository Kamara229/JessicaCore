"""S108: NEW original Russian-GSD TRAIN-only all-verb grammar gold-wall.

S107 structural competition weights 53,743 were physically registered
INSIDE Calibri and verified at zero gate, but NOT TRAINED. Taiga's
train-a/b/d/e selected sources and their dev labels are SPENT; original
Taiga TRAIN-c remains reserved/prospective. Taiga has no train-f split,
so use an independent *different* official UD Russian-GSD TRAIN corpus.
Neither GSD dev/test nor any previously spent diagnostic can be used.

Pin upstream Git commit and original file Git blob BEFORE selection.
Deterministically select 64 per active/passive/negated/intransitive
stratum (256 total), then generate SOURCE (row_id, text ONLY) versus
completely separate offline multi-VERB UD and morphology GOLD. Compare
source-text overlap against old Taiga-d/e prior256, fail closed. No
GOLD or UD parser available at model runtime. CC BY-SA 4.0 license
requires attribution/share-alike and legal review before commercial use.
Do not merge/deploy/update original Calibri, GreenTunnel or protected616.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    STRATA,git_blob_sha,parse_conllu,align_ud_surfaces_exact,
    select_locked_UD_train256,sha256_serialized,
)
from research.calibri_nra3.s103_new_Taiga_TRAINe_allVERB_source_gold_wall import (
    all_ud_verbs_roles_with_multiedges,offline_allword_morphology,
)

STAGE="CALIBRI_NRA3_S108_NEW_RUSSIAN_GSD_TRAIN256_SOURCE_VS_ALLVERB_GOLD"
UPSTREAM_REPO="UniversalDependencies/UD_Russian-GSD"
UPSTREAM_COMMIT="71b92ce4c8da2b779450b0297abada721796425f"
TRAIN_FILE="ru_gsd-ud-train.conllu"
TRAIN_GIT_BLOB="de392eb8d1a92fe1d12bc2e49c21cb717ae89859"
LICENSE_FILE_BLOB="9af2e0b2f73cf29fe1f60cb6376b176d9c2ba846"
LICENSE="CC BY-SA 4.0"
SOURCE_FILE="NRA3_S108_GSD_TRAIN256_SOURCE_ONLY_NO_GOLD.json"
GOLD_FILE="NRA3_S108_GSD_TRAIN256_ALLVERB_MORPH_OFFLINE_GOLD.json"
MANIFEST_FILE="NRA3_S108_GSD_TRAIN256_SOURCE_GOLD_PROVENANCE.json"

def require(ok,msg):
    if not bool(ok):
        raise ValueError("S108 new pinned GSD original TRAIN-only UD: "+msg)

def construct_gsd_new256(data:bytes,old_source_texts=()):
    require(isinstance(data,bytes) and git_blob_sha(data)==TRAIN_GIT_BLOB,
            "pin immutable original GSD upstream TRAIN Git blob")
    selected,selected_offline,_=select_locked_UD_train256(
        data,require_pinned_blob=False)
    lookup={}
    for sent_id,text,nodes in parse_conllu(data.decode("utf8")):
        if (sent_id,text) not in lookup:
            spans=align_ud_surfaces_exact(text,nodes)
            if spans is not None:
                lookup[(sent_id,text)]=(nodes,spans)
    out_source,out_gold=[],[]
    seen=set(old_source_texts)
    require(len(seen)==len(tuple(old_source_texts)),
            "previously licensed Taiga source text overlap audit must have unique texts")
    strat=Counter(); tags=Counter(); verbs=0; edges=0
    for ix,(s,chosen) in enumerate(zip(selected,selected_offline)):
        require(s["text"] not in seen,
                "no re-use or source-text overlap with previously spent Taiga data")
        seen.add(s["text"])
        key=(chosen["ud_sentence_ID_offline_only"],s["text"])
        require(key in lookup,"original GSD word offsets must exist exactly")
        nodes,spans=lookup[key]
        require([list(z) for z in spans]==
                [list(z) for z in chosen["word_surface_start_end_char_offsets_TRAIN_ONLY"]],
                "GSD source must preserve original char-offset alignment")
        events=all_ud_verbs_roles_with_multiedges(nodes)
        require(events and all(z["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"]
                               for z in events),
                "all selected GSD UD verbs require real positive linked arguments")
        morph=offline_allword_morphology(nodes)
        rid=f"NRA3_S108_GSD_{ix+1:03d}"
        out_source.append({"row_id":rid,"text":s["text"]})
        out_gold.append({
            "row_id":rid,
            "original_GSD_UD_sentence_ID_OFFLINE_ONLY":
                chosen["ud_sentence_ID_offline_only"],
            "complete_original_source_UD_word_start_end_chars_TRAIN_ONLY":
                [list(z) for z in spans],
            "all_UD_VERB_predicates_and_all_unambiguous_multi_role_edges_TRAIN_ONLY":
                events,
            "reported_speech_factuality_coreference_not_adjudicated_UNKNOWN":True,
            "no_negative_event_labels_invented":True,
            "multiple_distinct_arguments_PER_role_NOT_collapsed":True,
            "original_Calibri_BPE_alignment_NOT_YET_known":True,
            "Russian_GSD_UD_not_real_human_QA_gold":True,
            **morph,
        })
        strat[chosen["grammar_supervision_bucket_TRAIN_ONLY"]]+=1
        verbs+=len(events)
        edges+=sum(len(x["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"])
                   for x in events)
        for name,ls in morph.items():
            tags[name]+=sum(x is not None for x in ls)
    require(len(out_source)==len(out_gold)==256
            and len({z["text"] for z in out_source})==256
            and dict(strat)=={k:64 for k in STRATA}
            and verbs>=256 and edges>=300
            and min(tags.values())>=25
            and all(set(z)=={"row_id","text"} for z in out_source)
            and all(s["row_id"]==g["row_id"]
                    for s,g in zip(out_source,out_gold)),
            "strict no-duplicates 64x4 Russian GSD source/GOLD separated")
    manifest={
        "stage":STAGE,
        "source_upstream":UPSTREAM_REPO,
        "immutable_upstream_commit":UPSTREAM_COMMIT,
        "immutable_UD_GSD_TRAIN_git_blob":TRAIN_GIT_BLOB,
        "immutable_UD_GSD_LICENSE_git_blob":LICENSE_FILE_BLOB,
        "license":LICENSE,
        "ONLY_upstream_GSD_TRAIN_used_not_Taiga_or_GSD_dev_test":True,
        "old_Taiga_S100_d_S103_e_256_source_text_overlap":0,
        "all_old_S96_S99_S102_S105_internal_devs_SPENT_not_used":True,
        "old_S79_S86_independent_firstreads_SPENT_not_used":True,
        "original_Taiga_TRAIN_c_reserved_UNOPENED":True,
        "new_original_GSD_source_only_SHA256":sha256_serialized(out_source),
        "new_offline_GSD_allVERB_morph_GOLD_SHA256":sha256_serialized(out_gold),
        "new_GSD_source_rows":len(out_source),
        "four_precommitted_UD_strata_counts":dict(strat),
        "known_UD_VERB_events":verbs,
        "known_positive_UD_predicate_argument_links":edges,
        "known_UD_word_morph_tags":dict(tags),
        "source_only_output_contains_no_UD_POS_grammar_roles_or_parser":True,
        "offline_UD_gold_NEVER_passed_to_original_Calibri":True,
        "gold_ONLY_after_real_original12_source_SHA_pinned":True,
        "eight_original_NRA2_mini_areas_retained_in_new_53743_child":True,
        "all_12_original_Calibri_plus_GT_layers_required":True,
        "not_independent_naturalistic_event_or_factuality_evidence":True,
        "attribution_sharealike_legal_review_before_commercial_or_production":True,
        "new_GSD_neural_train_steps":0,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return out_source,out_gold,manifest

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--official-gsd-train",type=Path,required=True)
    a.add_argument("--previous-source-file",type=Path,action="append",required=True)
    a.add_argument("--out",type=Path,required=True)
    q=a.parse_args()
    require(q.official_gsd_train.is_file() and not q.out.exists()
            and len(q.previous_source_file)>=2,
            "new GSD corpus and two separate prior Taiga source-only files")
    previous=[]
    for f in q.previous_source_file:
        rows=json.loads(f.read_text(encoding="utf8"))
        require(len(rows)==256 and all(set(z)=={"row_id","text"} for z in rows)
                and len({z["text"] for z in rows})==256,
                "previous Taiga source-only 256 audit read, NEVER prior GOLD")
        previous.extend(z["text"] for z in rows)
    require(len(previous)==len(set(previous)),
            "previous Taiga-d and Taiga-e source-only corpus disjoint")
    source,gold,report=construct_gsd_new256(
        q.official_gsd_train.read_bytes(),previous)
    q.out.mkdir(parents=True,exist_ok=False)
    for filename,value in ((SOURCE_FILE,source),(GOLD_FILE,gold),
                           (MANIFEST_FILE,report)):
        (q.out/filename).write_text(
            json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
            encoding="utf8")
    print("NRA3_S108_NEW_UD_GSD_SOURCE_AND_GOLD_SHA_PINNED",
          json.dumps({"source_SHA":report["new_original_GSD_source_only_SHA256"],
                      "GOLD_SHA":report["new_offline_GSD_allVERB_morph_GOLD_SHA256"],
                      "events":report["known_UD_VERB_events"],
                      "positive_edges":report["known_positive_UD_predicate_argument_links"],
                      "morph":report["known_UD_word_morph_tags"],
                      "output_gate":0},ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
