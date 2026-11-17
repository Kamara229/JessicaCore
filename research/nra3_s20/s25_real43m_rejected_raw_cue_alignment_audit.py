"""S25: physical S22c word338 input-only cue audit for S24 failed inversion.

Diagnoses whether the text-only Russian "не"+dative indicator actually
reaches the *correct* annotated REJECTED argument word on original43M+GT
CPU representation, before changing any learnable architecture again.
Human TRAIN gold used ONLY for offline audit indexing, NEVER model forward.
No training, model inference, EVAL, LIVE or activation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    FEATURES,raw_surface_grammar,
)
from research.nra3_s19.s19f_real43m_train9_graph_targets import ROLE
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    build_s22_train256,FROZEN_S22_TRAIN256_SHA256,PATTERNS,
)
from research.nra3_s20.s22d_train256_grouped40_true43m_graph import (
    load_physical_training_source,
)
from research.nra3_s20.s24_neural_role_local_negation_scope import (
    PREV_NEG,LIKELY_DATIVE,
)


def verify(condition, message):
    if not condition:
        raise ValueError("S25 actual original43M source geometry: "+message)


def cue_audit(raw,valid,gold):
    rows=build_s22_train256()
    verify(len(rows)==256 and raw.shape[0]==256 and raw.shape[-1]==338,
           "not genuine source 256 raw338 geometry")
    found=defaultdict(list)
    truth=gold["argument_polarity_TRAIN_ONLY"]
    role=gold["predicate_argument_role_TRAIN_ONLY"]
    counted=0
    for row_id,g in enumerate(rows):
        words=raw_surface_grammar(g.text)
        verify(len(words)==int(valid[row_id].sum()),"text and original BPE word alignment drift")
        anchors={e.event_id:e.predicate-1 for e in g.events}
        for edge in g.arguments:
            if edge.polarity!="REJECTED":
                continue
            counted+=1
            src=anchors[edge.event_id]
            target=edge.token-1
            verify(int(truth[row_id,src,target])==1 and
                   int(role[row_id,src,target])==ROLE.index(edge.role) and
                   edge.role=="RECIPIENT" and target!=src,
                   "gold label is not REJECTED recipient for this word")
            surface=words[target]
            verify(g.text[surface.start:surface.end]==surface.form,
                   "missing exact raw surface word")
            real_cue=raw[row_id,target,320:].tolist()
            verify(all(real_cue[k]==surface.features[k] for k in range(18)),
                   "physical real 43M raw18 tokenizer alignment differs from text-only source")
            # The paired semantic polarity head predicts a (predicate,target) pair;
            # no annotations enter its actual model forward.
            i=(row_id % 32)
            found[PATTERNS[i]].append({
                "row_index":row_id,
                "TRAIN_only_lexicon_index":row_id//32,
                "predicate_to_rejected_argument_order":(
                    "ARGUMENT_BEFORE_PREDICATE" if target<src
                    else "PREDICATE_BEFORE_ARGUMENT"),
                "surface_previous_word_negative_particle":int(real_cue[PREV_NEG]),
                "surface_likely_dative_suffix":int(real_cue[LIKELY_DATIVE]),
                "raw_previous_word_is_ne":int(
                    words[target].features[FEATURES.index("previous_word_negative_particle")]),
                "raw_rejected_word":surface.form,
                "text_sha256":hashlib.sha256(g.text.encode("utf-8")).hexdigest(),
            })
    verify(counted==32 and
           set(found)==set(PATTERNS[24:28]) and
           all(len(z)==8 for z in found.values()),
           "S25 expected four eight-example REJECTED train patterns")
    groups={}
    for pat in PATTERNS[24:28]:
        records=found[pat]
        groups[pat]={
            "count":len(records),
            "negation_preceding_recipient":sum(z["surface_previous_word_negative_particle"] for z in records),
            "recipient_possible_dative":sum(z["surface_likely_dative_suffix"] for z in records),
            "negation_and_dative_both":sum(
                z["surface_previous_word_negative_particle"]*
                z["surface_likely_dative_suffix"] for z in records),
            "recipient_precedes_predicate":sum(
                z["predicate_to_rejected_argument_order"]=="ARGUMENT_BEFORE_PREDICATE"
                for z in records),
            "examples":records,
        }
    preposed=groups["rejected_recipient_preposed"]
    return {
        "stage":"NRA3_S25_REAL_43M_GT_RAW_INPUT_ONLY_CUE_AUDIT_NO_TRAINING",
        "source_SHA256":FROZEN_S22_TRAIN256_SHA256,
        "preposed_source_rejected_8":preposed,
        "other_rejected_surface_groups":{k:v for k,v in groups.items()
                                         if k!="rejected_recipient_preposed"},
        "architectural_hypothesis":"S24 failed 0/8; test whether raw_NEG reaches the preposed recipient; if 8/8 present, fix predicted-role probability gate or pair orientation, NOT data annotation.",
        "genuine_original43M_GT_original_616J7_unchanged":True,
        "model_forward_or_optimizer_executed":False,
        "human_train_gold_never_forwarded":True,
        "no_spent_independent_EVAL":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("features","gold","manifest","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    verify(not a.out.exists(),"do not overwrite existing diagnostics")
    raw,valid,y,m=load_physical_training_source(a.features,a.gold,a.manifest)
    result=cue_audit(raw,valid,y)
    result["actual_S22c_source_input_SHA256"]=m["raw338_word_features_SHA256"]
    result["actual_S22c_source_human_gold_SHA256"]=m["TRAIN_only_gold_SHA256"]
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"NRA3_S25_ACTUAL_43M_GT_RAW_REJECTED_SURFACE_CUE_AUDIT.json").write_text(
        json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8")
    print("S25_PREPOSED_REAL_RAW_REJECTED_SURFACE",
          json.dumps({k:v for k,v in result["preposed_source_rejected_8"].items()
                      if k!="examples"},sort_keys=True),flush=True)
    print("S25_OTHER_REJECTED_REAL_RAW_SURFACE",
          json.dumps({k:{x:y for x,y in v.items() if x!="examples"}
                      for k,v in result["other_rejected_surface_groups"].items()},
                     sort_keys=True),flush=True)
    print("S25_REAL_ORIGINAL43M_GT_RAW_INPUT_ONLY_SOURCE_PINNED",flush=True)


if __name__=="__main__":
    main()
