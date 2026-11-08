"""CALIBRI S53: one-time *read-only* unseen paired-response robustness audit.

The physical S52 checkpoint is FROZEN BEFORE this independent evaluation.
S52 #42514912967: strict 29/64 new TRAIN full+EOS correct, 12/32 both
contrast sides correct, old 7/8 kept, 34 first-color correct, all physical
3,174,717 bytes / 0.698114373422 bpp; saved sidecar reload was confirmed.
These TRAIN numbers are not generalization. S53 checks whether the new
paired-noun behavior transfers to truly unseen natural-language templates,
names and context permutations. S45 and S47 act as read-only controls.

No model training; no checkpoint selection; no gradients, optimizer, or
QA-driven hyperparameter changes. This new 96-case diagnostic benchmark is
NOT S29's previously sealed final 96-new-noun QA and not previously
inspected S26 QA. The precommitted source itself IS the test definition,
independent of S52 predictions, and any later training must exclude its
content and never tune against its results. This diagnostic is not a claim
of generalization beyond these examples or parity with original FP32.
Do not touch Jessica chat, production, LIVE, Render, Supabase, APK or main.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s47 import active_small_green_tunnel as s47
from v176.operators import clear
from v174.run import save

S45_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
S47_SHA="02227f62d961b73ed1e7cf89e370d5a4852bf53839ce8fa8f608c44e286cdb43"
S52_SHA="80e52005a2c1ed97937a302c12b4d0b620264e13fe91246880544a9cb41eb4cd"
S47_SIDECAR_BYTES=22068
S52_SIDECAR_BYTES=32308
S52_INCLUSIVE_BYTES=3174717
S52_BPP="0.698114373422"
S52_MANIFEST_BYTES=896

# These 8 color+noun combinations were TRAIN-eligible in S30/S52.
# DO NOT use the other 8 S29 held-out combinations, nor read S29 files.
ELIGIBLE={
  "red":("bowl","drum"),
  "green":("bowl","rope"),
  "blue":("ring","rope"),
  "yellow":("ring","drum"),
}
NAMES=(
 ("Maya","Victor"),("Nina","Oskar"),("Felix","Olga"),
 ("Hugo","Marta"),("Irina","Daniel"),("Rita","Peter"),
 ("Zoe","Anton"),("Elena","Boris"),("Anita","Leo"),
 ("Sasha","Noah"),("Dina","Adam"),("Eva","Marco")
)
# Unlike TRAIN, these have no "TRAIN item/card/report/sample" marker,
# no FPTrainL/FPTrainR names, and a different fixed set of compositional
# templates. Constrain desired format to two content words: full+EOS only.
TEMPLATES=(
 "{a} owns the {left}. {b} owns the {right}. Reply with {a}'s color and object in exactly two words.",
 "A card lists {a}: {left}; {b}: {right}. What is {a}'s two-word item?",
 "The {left} belongs to {a}, while the {right} belongs to {b}. Which color and object are assigned to {a}?",
 "Between two tagged items, {a} selected {left} and {b} selected {right}. Say {a}'s item using only its color and noun.",
 "In this inventory, {a} takes {left} and {b} takes {right}. Name the two-word item for {a}.",
 "For {a}, the choice is {left}; for {b}, the choice is {right}. What did {a} choose? Two words."
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_precommitted_96():
    pairs=[]
    for color,nouns in ELIGIBLE.items():
        for k in range(12):
            a,b=NAMES[k]
            template=TEMPLATES[k%len(TEMPLATES)]
            left=f"{color} {nouns[0]}"
            right=f"{color} {nouns[1]}"
            first=template.format(a=a,b=b,left=left,right=right)
            second=template.format(a=a,b=b,left=right,right=left)
            pair={"id":8000+len(pairs),
                  "color":color,
                  "new_natural_template_index":k%len(TEMPLATES),
                  "left":{"question":first,"answer":left},
                  "right":{"question":second,"answer":right}}
            if first==second or left==right:
                raise AssertionError("not a real oppositional noun pair")
            pairs.append(pair)
    questions=[p[s]["question"] for p in pairs for s in ("left","right")]
    if len(pairs)!=48 or len(questions)!=96 or len(set(questions))!=96:
        raise AssertionError("unseen QA set must contain exactly 48 actual paired prompts")
    if any("TRAIN" in q or "FPTrain" in q for q in questions):
        raise AssertionError("train template leaked into frozen S53 QA")
    for p in pairs:
        if p["left"]["answer"].split()[0]!=p["right"]["answer"].split()[0]:
            raise AssertionError("paired contrast must share the same gold first color")
    return pairs

@torch.inference_mode()
def score_same_frozen_model(model,sources,tok,pairs,controller,head,label):
    per={}
    start=time.perf_counter()
    for p in pairs:
        for side in ("left","right"):
            data=p[side]
            key=f'{p["id"]}_{side}'
            actual=s47.greedy(model,sources,controller,head,tok,data["question"])
            expected=tok.encode(data["answer"],add_special_tokens=False)+[tok.eos_token_id]
            if len(expected)!=3:
                raise AssertionError("expected a color+noun+EOS gold sequence")
            per[key]={
               "prompt_id":p["id"],"side":side,
               "expected_text":data["answer"],
               "expected_full_tokens_EOS":expected,
               "actual_full_tokens":actual,
               "actual_text":tok.decode(actual,skip_special_tokens=True),
               "full_correct":actual==expected,
               "first_correct":bool(actual) and actual[0]==expected[0],
               "second_correct":len(actual)>1 and actual[1]==expected[1],
               "ends_in_EOS":bool(actual) and actual[-1]==tok.eos_token_id
            }
    elapsed=time.perf_counter()-start
    totals={
      "stage":label,"total_cases":96,"total_pairs":48,
      "strict_full_correct_with_EOS":sum(x["full_correct"] for x in per.values()),
      "first_correct":sum(x["first_correct"] for x in per.values()),
      "second_correct":sum(x["second_correct"] for x in per.values()),
      "ending_in_EOS":sum(x["ends_in_EOS"] for x in per.values()),
      "both_pair_sides_full_correct":sum(
        per[f'{p["id"]}_left']["full_correct"] and per[f'{p["id"]}_right']["full_correct"] for p in pairs),
      "same_full_token_sequence_on_opposing_gold_nouns":sum(
        per[f'{p["id"]}_left']["actual_full_tokens"]==per[f'{p["id"]}_right']["actual_full_tokens"]
        for p in pairs),
      "distinct_full_texts":len({x["actual_text"] for x in per.values()}),
      "top_repeated_texts":Counter(x["actual_text"] for x in per.values()).most_common(8),
      "CPU_single_run_wall_seconds":elapsed,
      "CPU_timing_exploratory_not_certification":True
    }
    return {"summary":totals,"all_actual_free_greedy_answers":per}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("s45_CB59",type=Path)
    ap.add_argument("s47_sidecar",type=Path)
    ap.add_argument("s52_sidecar",type=Path)
    ap.add_argument("s52_manifest",type=Path)
    ap.add_argument("output",type=Path)
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(5353)
    torch.set_num_threads(2)
    raw=a.s45_CB59.read_bytes()
    if len(raw)!=SOURCE_BYTES or sha(a.s45_CB59)!=S45_SHA:
        raise AssertionError("S45 sealed 36M CB59 source must match SHA+bytes")
    if a.s47_sidecar.stat().st_size!=S47_SIDECAR_BYTES or sha(a.s47_sidecar)!=S47_SHA:
        raise AssertionError("S47 saved accepted sidecar differs")
    if a.s52_sidecar.stat().st_size!=S52_SIDECAR_BYTES or sha(a.s52_sidecar)!=S52_SHA:
        raise AssertionError("S52 saved accepted sidecar differs")
    manifest=json.loads(a.s52_manifest.read_text(encoding="utf-8"))
    if (a.s52_manifest.stat().st_size!=S52_MANIFEST_BYTES
        or manifest["all_inclusive_total_bytes"]!=S52_INCLUSIVE_BYTES
        or manifest["all_inclusive_bpp"]!=S52_BPP
        or manifest["source_CB59_sha256"]!=S45_SHA
        or manifest["new_controller_sha256"]!=S52_SHA
        or manifest["new_controller_bytes"]!=S52_SIDECAR_BYTES
        or S52_INCLUSIVE_BYTES!=SOURCE_BYTES+S52_SIDECAR_BYTES+S52_MANIFEST_BYTES):
        raise AssertionError("S52 native paid physical manifest does not match saved checkpoint")
    if float(S52_BPP)>=.7:
        raise AssertionError("S52 real paid Green Tunnel crossed physical budget")
    pairs=build_precommitted_96()
    frozen_dataset_payload=json.dumps(pairs,sort_keys=True,ensure_ascii=False,separators=(",",":"))
    dataset_sha=hashlib.sha256(frozen_dataset_payload.encode("utf-8")).hexdigest()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    m,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    m.requires_grad_(False)
    head=m.lm_head.source().detach()
    frozenS47=s47.ActiveGreenTunnel(320,8)
    frozenS47.load_state_dict(load_file(str(a.s47_sidecar)))
    frozenS47.eval().requires_grad_(False)
    frozenS52=s47.ActiveGreenTunnel(320,12)
    frozenS52.load_state_dict(load_file(str(a.s52_sidecar)))
    frozenS52.eval().requires_grad_(False)
    none=score_same_frozen_model(m,sources,tok,pairs,None,None,"S45_frozen_packed_CB59")
    print(json.dumps({"event":"S53_FROZEN_NOVEL_96_QA",**none["summary"]},ensure_ascii=False),flush=True)
    s47ref=score_same_frozen_model(m,sources,tok,pairs,frozenS47,head,"S47_frozen_physically_paid_0p695842")
    print(json.dumps({"event":"S53_FROZEN_NOVEL_96_QA",**s47ref["summary"]},ensure_ascii=False),flush=True)
    s52ref=score_same_frozen_model(m,sources,tok,pairs,frozenS52,head,"S52_frozen_physically_paid_0p698114")
    print(json.dumps({"event":"S53_FROZEN_NOVEL_96_QA",**s52ref["summary"]},ensure_ascii=False),flush=True)
    output={
      "stage":"S53_ONCE_UNSEEN_NOVEL_48_PAIRED_96_QA_READONLY",
      "NO_TRAINING_OR_CHECKPOINT_SELECTION":True,
      "NO_QA_GRADIENTS_OR_ADAPTATION":True,
      "S29_FROZEN_FINAL_QA_NEVER_OPENED":True,
      "S26_PREVIOUSLY_INSPECTED_QA_NEVER_OPENED":True,
      "S53_DIAGNOSTIC_MUST_NEVER_BE_TUNED_ON":True,
      "S52_SAVED_BEFORE_S53_NEW_QA_WAS_DEFINED":True,
      "S52_original_train_new_full":29,
      "S52_original_train_both_32_pairs":12,
      "S52_original_train_old_full":7,
      "S52_original_train_first_color_correct":34,
      "S52_all_inclusive_bpp":S52_BPP,
      "source_SHA":S45_SHA,
      "S47_controller_SHA":S47_SHA,
      "S52_controller_SHA":S52_SHA,
      "novel_96_dataset_SHA256":dataset_sha,
      "new_prompt_templates":list(TEMPLATES),
      "new_person_names":list(NAMES),
      "new_questions_all96":pairs,
      "scores":{"S45":none,"S47":s47ref,"S52":s52ref},
      "not_FP32_parity_or_generalization_certification":True,
      "Jessica_chat_TEST_NOT_RUN":True,
      "LIVE_Render_Supabase_APK_UNCHANGED":True
    }
    save(a.output/"S53_96_NOVEL_PAIR_QA_FROZEN_CHECKPOINT_READONLY.json",output)
    print(json.dumps({
       "event":"S53_FINAL_FROZEN_NOVEL_PAIR_GENERALIZATION_DIAGNOSTIC",
       "S45_full":none["summary"]["strict_full_correct_with_EOS"],
       "S47_full":s47ref["summary"]["strict_full_correct_with_EOS"],
       "S52_full":s52ref["summary"]["strict_full_correct_with_EOS"],
       "S45_pairs_both":none["summary"]["both_pair_sides_full_correct"],
       "S47_pairs_both":s47ref["summary"]["both_pair_sides_full_correct"],
       "S52_pairs_both":s52ref["summary"]["both_pair_sides_full_correct"],
       "S52_physically_inclusive_bpp":S52_BPP,
       "new96_dataset_sha":dataset_sha,
       "S29_untouched":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
