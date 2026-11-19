"""NRA3 S97: separately pinned REAL Russian UD-Taiga TRAIN-b morphology.

NO old S96 TRAIN-a internal development, NO upstream dev/test,
NO older S79/S86 spent firstread or unsealed S66 EVAL data.
This is a NEW external third-party corpus file (train-b) at a
fixed verified original Git blob SHA. S97 source and morphology/
dependency labels are saved to separate research files. This is a
training-only teacher and NEVER imported by actual Calibri forward.

Four 64-row grammar strata selected with same fixed S94 algorithm,
but new S97 row namespace. TRAIN-b is an unreviewed mixture of UD
manual/automatic labels, NOT a gold real-user event benchmark.
License: CC BY-SA 4.0; NO production/commercial derivatives until
legal compliance/attribution/share-alike assessed independently.

MORPH label on EACH strict UD raw source word is UPOS, morphological
Case, Number, Voice when explicitly present (not present => UNKNOWN).
No invented negative pseudo-tags, and no teacher labels as model input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    UPSTREAM_REPO,UPSTREAM_COMMIT,LICENSE,STRATA,LIMIT_PER_STRATUM,
    git_blob_sha,parse_conllu,align_ud_surfaces_exact,
    select_locked_UD_train256,sha256_serialized,
)
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    UD_UPOS,UD_CASE,UD_NUMBER,UD_VOICE,
)

STAGE="CALIBRI_NRA3_S97_PINNED_UD_TAIGA_TRAIN_B_MORPHOLOGY_AND_ROLE_GOLD_WALL"
TRAIN_B_FILE="ru_taiga-ud-train-b.conllu"
TRAIN_B_BLOB_SHA="3b0f6947bd0d6ac022d91d5fb17269450a876bb3"
SOURCE_FILE="NRA3_S97_TAIGA_TRAINB256_SOURCE_ONLY_NO_GOLD.json"
GOLD_FILE="NRA3_S97_TAIGA_TRAINB256_OFFLINE_UD_ALLWORD_MORPH_GOLD.json"
REPORT_FILE="NRA3_S97_TAIGA_TRAINB256_SOURCE_GOLD_LICENSE_PROVENANCE.json"


def require(x,why):
    if not bool(x):
        raise ValueError("NRA3 S97 TRAIN-b license and offline morphology: "+why)


def one_feat(feats,key,possible):
    if feats=="_":
        return None
    values=[bit.split("=",1)[1] for bit in feats.split("|")
            if bit.startswith(key+"=") and bit.count("=")==1]
    require(len(values)<=1,"ambiguous multiple same-key UD features")
    if not values:
        return None
    require("," not in values[0],
            "ambiguous multi-valued feature must not become fake label")
    return values[0] if values[0] in possible else None


def new_training_only_corpus256(data:bytes,*,check_blob=True):
    if check_blob:
        require(git_blob_sha(data)==TRAIN_B_BLOB_SHA,
                "TRAIN-b immutable upstream Git blob mismatch")
    # Reuse fixed S94 source selection algorithm on *new* train-b,
    # but retain new morphological annotation on EVERY strict source word.
    source94,gold94,old_report=select_locked_UD_train256(
        data,require_pinned_blob=False)
    sentences={}
    for sent_id,text,nodes in parse_conllu(data.decode("utf8")):
        key=(sent_id,text)
        if key in sentences:
            continue
        # Raw Unicode word spans already normalized by S94.
        spans=align_ud_surfaces_exact(text,nodes)
        if spans is not None:
            sentences[key]=(nodes,spans)
    new_source=[]
    new_gold=[]
    n_morph_annotated={"upos":0,"case":0,"number":0,"voice":0}
    for original,old in zip(source94,gold94):
        key=(old["ud_sentence_ID_offline_only"],original["text"])
        require(key in sentences,"all previously frozen train-b source IDs found")
        nodes,spans=sentences[key]
        require([list(z) for z in spans]==
                    [list(z) for z in old["word_surface_start_end_char_offsets_TRAIN_ONLY"]],
                "exact preselected strict UD source Unicode offsets")
        new_id=original["row_id"].replace("S94_TAIGA_TRAIN_",
                                          "S97_TAIGA_TRAINB_")
        upos=[]
        case=[]
        number=[]
        voice=[]
        for word in nodes:
            p=word.upos if word.upos in UD_UPOS else None
            c=one_feat(word.feats,"Case",UD_CASE)
            n=one_feat(word.feats,"Number",UD_NUMBER)
            v=one_feat(word.feats,"Voice",UD_VOICE)
            upos.append(p);case.append(c);number.append(n);voice.append(v)
            for key,val in (("upos",p),("case",c),("number",n),("voice",v)):
                n_morph_annotated[key]+=int(val is not None)
        require(len(upos)==len(spans)>=5
                and old["predicate_UD_word_zero_index_TRAIN_ONLY"]<len(nodes),
                "full strict all-word UD morphology and event predicate")
        new_source.append({"row_id":new_id,"text":original["text"]})
        g={key:val for key,val in old.items()}
        g["row_id"]=new_id
        g["offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY"]=upos
        g["offline_Case_ALL_UD_WORDS_TRAIN_ONLY"]=case
        g["offline_Number_ALL_UD_WORDS_TRAIN_ONLY"]=number
        g["offline_Voice_ALL_UD_WORDS_TRAIN_ONLY"]=voice
        g["UD_morphology_not_independent_human_factuality_gold"]=True
        new_gold.append(g)
    require(len(new_source)==len(new_gold)==256
            and len({z["text"] for z in new_source})==256
            and all(set(z)=={"row_id","text"} for z in new_source)
            and all(g["row_id"]==s["row_id"]
                    for s,g in zip(new_source,new_gold))
            and n_morph_annotated["upos"]>=900
            and n_morph_annotated["case"]>=250
            and n_morph_annotated["number"]>=300
            and n_morph_annotated["voice"]>=90,
            "adequate morphological teacher coverage, source-only/gold separation")
    report={
        "stage":STAGE,
        "upstream_repo":UPSTREAM_REPO,
        "upstream_commit":UPSTREAM_COMMIT,
        "TRAIN_B_genuine_pinned_git_blob_sha":TRAIN_B_BLOB_SHA,
        "license":LICENSE,
        "attribution_and_share_alike_mandatory_for_derivatives":True,
        "commercial_production_use_forbidden_until_legal_review":True,
        "ONLY_original_Taiga_TRAIN_b_no_train_a_dev_test":True,
        "S96_old_Taiga_TRAIN_a_internal_dev_NEVER_reused":True,
        "S79_S86_old_firstread_TEST48_NEVER_reused":True,
        "NRA3_is_in_Calibri_neural_weights_NOT_external_teacher":True,
        "no_external_UD_parser_at_inference":True,
        "full_SOURCE_has_only_raw_rowIDs_and_text_no_UD_gold":True,
        "n_training_rows":len(new_source),
        "stratification_counts":old_report["stratification_counts"],
        "n_valid_UD_word_morph_labels_by_feature":n_morph_annotated,
        "strict_SOURCE_only_sha256":sha256_serialized(new_source),
        "strict_separate_offline_MORPH_and_role_GOLD_sha256":
            sha256_serialized(new_gold),
        "source_original_native_Calibri_BPE_alignment_NOT_YET_captured":True,
        "S97_untrained_new_morphological_neural_heads_NOT_grammar_quality":True,
        "new_train_steps":0,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return tuple(new_source),tuple(new_gold),report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--taiga-train-b",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    require(args.taiga_train_b.is_file() and not args.out.exists(),
            "source must be fresh immutable original licensed TRAIN-b")
    src,gold,report=new_training_only_corpus256(
        args.taiga_train_b.read_bytes(),check_blob=True)
    args.out.mkdir(parents=True,exist_ok=False)
    for name,payload in ((SOURCE_FILE,src),(GOLD_FILE,gold),
                         (REPORT_FILE,report)):
        (args.out/name).write_text(
            json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
            encoding="utf8")
    print("NRA3_S97_PINNED_NEW_UD_TAIGA_TRAIN_B256_MORPH_TEACHER",
          json.dumps({
              "source_SHA":report["strict_SOURCE_only_sha256"],
              "gold_SHA":report["strict_separate_offline_MORPH_and_role_GOLD_sha256"],
              "morphology_annotated":report["n_valid_UD_word_morph_labels_by_feature"],
              "licence":"CC BY-SA 4.0","output_gate":0,
          },sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
