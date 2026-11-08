"""S26 independently frozen 96 wholly new noun color combinations, not TRAIN.

Goal: after S22/S23, test whether 36M can pick a correct *noun* after
learning continuation and keep previous skills. 8 withheld COLOR+NEW-NOUN
combinations from 4 previously unseen nouns, 12 variants each.
The entire S26 TRAIN source is forbidden from using these 8 exact
color+noun combinations. Do not use these questions or their answers
for gradients, hyperparameter selection, stopping or checkpoint choice.
Unlike S24 QA, these lexical combinations were frozen in PR #404
before any S26 training code existed. Responses are actual greedy
generated tokens, checked in full including EOS; first-slot-only credit
does NOT count as correct. No 135M FP32 comparison or teacher.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.fresh_heldout import generate
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from v174.run import save

from calibri2.s26.spec import COLORS, ALL_NOUNS, RESERVED
from calibri2.s26.frozen_new_qa import (
    cases, OPEN_QUESTIONS as FREE_QUESTIONS, manifest
)

def score(rows):
    stats=defaultdict(Counter)
    predictions=Counter()
    expression=re.compile(r"^\s*(red|blue|green|yellow)\s+"
       r"(cup|book|bag|lamp|chair|hat|plate|scarf|clock|spoon|pencil|bottle|fork|glove|shirt|key)\s*[.!]?\s*$",re.I)
    for row in rows:
        d=stats[row["domain"]];d["n"]+=1
        d["strict_full_EOS"]+=int(row["strict_full_EOS"])
        d["correct_first_token"]+=int(row["correct_first_token"])
        d["EOS_emitted"]+=int(row["EOS_emitted"])
        if row["domain"]!="reserved_novel_color_noun":continue
        m=expression.fullmatch(row["actual_text"])
        gold=row["gold"].split()
        first=bool(m and m.group(1).lower()==gold[0])
        noun=bool(m and m.group(2).lower()==gold[1])
        d["valid_two_word"]+=int(m is not None)
        d["correct_color_with_valid_format"]+=int(first)
        d["correct_noun_with_valid_format"]+=int(noun)
        d["both_semantic_words_correct"]+=int(first and noun)
        d["color_correct_noun_wrong"]+=int(first and not noun)
        d["noun_correct_color_wrong"]+=int(noun and not first)
        predictions[row["actual_text"].strip()]+=1
    return dict(stats),predictions.most_common(16)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--label",required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=a.archive.read_bytes()
    archive=CB59(raw)
    if len(raw)!=SOURCE_BYTES or archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S26 requires fixed physical 36M CB59")
    bpp=8*len(raw)/SOURCE_PARAMS
    if bpp>.7:raise AssertionError("S26 physically unpaid model")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases_now=cases()
    qa_sha=manifest()["qa_suite_sha256"]
    if len(cases_now)!=144 or len(RESERVED)!=8:
        raise AssertionError("S26 QA manifest unexpectedly changed")
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    model.eval()
    real=[]
    for idx,case in enumerate(cases_now):
        gold=tok.encode(case["gold"],add_special_tokens=False)+[tok.eos_token_id]
        if len(gold)>14:raise AssertionError("S26 target too long for 14-token cap")
        predicted=generate(model,sources,tok,case["question"],14)
        real.append({
          **case,
          "expected_token_ids_full_EOS":gold,"actual_token_ids":predicted,
          "actual_text":tok.decode(predicted,skip_special_tokens=True),
          "strict_full_EOS":predicted==gold,
          "correct_first_token":bool(predicted) and predicted[0]==gold[0],
          "EOS_emitted":bool(predicted) and predicted[-1]==tok.eos_token_id
        })
        if (idx+1)%48==0:
            print(json.dumps({"event":"S26_FROZEN_QA_PROGRESS",
                "label":a.label,"checked":idx+1,
                "complete_correct":sum(v["strict_full_EOS"] for v in real)}),flush=True)
    opened=[]
    for question in FREE_QUESTIONS:
        generated=generate(model,sources,tok,question,32)
        opened.append({"question":question,
          "actual_answer":tok.decode(generated,skip_special_tokens=True),
          "actual_tokens":generated,
          "EOS":bool(generated) and generated[-1]==tok.eos_token_id,
          "requires_human_coherence_review":True})
    per_domain,common=score(real)
    result={
      "stage":"S26_PRECOMMITTED_NOVEL_96_NOUN_BINDINGS_48_CONTROLS_16_OPEN",
      "label":a.label,
      "physical_archive_SHA256":hashlib.sha256(raw).hexdigest(),
      "complete_bytes":len(raw),"physical_bpp":bpp,
      "unique_params":SOURCE_PARAMS,"frozen_suite_sha256":qa_sha,
      "withheld_new_object_combinations":RESERVED,
      "no_QA_used_for_training_or_model_selection":True,
      "strict_full_EOS_only_claims_correct_complete_answer":True,
      "total_cases":len(real),
      "total_correct_full_EOS":sum(x["strict_full_EOS"] for x in real),
      "per_domain":per_domain,"common_real_answers":common,
      "every_144_actual_answer_and_gold":real,
      "every_16_open_actual_answer":opened,
      "no_original_135m_FP32_teacher":True,
    }
    save(a.out/"ACTUAL_CALIBRI2_S26_NOVEL_144_16_ANSWERS.json",result)
    print(json.dumps({key:value for key,value in result.items()
       if key not in ("every_144_actual_answer_and_gold",
                      "every_16_open_actual_answer","common_real_answers")}),flush=True)

if __name__=="__main__":
    main()
