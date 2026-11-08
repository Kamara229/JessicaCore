"""CALIBRI S55 — one-time independent QA of physically frozen S54.

S54 #42516715008 accepted physically paid S45+S54 rank12 controller:
29/64 original TRAIN full+EOS, 12/32 original opposing pairs both correct,
7/8 original old historical full+EOS; 15/64 extra distinct TRAIN full+EOS
and 7/32 extra paired both-correct; 0.698067535118 all-inclusive bpp.
ALL these are TRAIN scores only; independence/generalization not established.

Create a genuinely new read-only 48-pair / 96-response QA definition from
scratch with entirely new content structure and labels, frozen unchanged
for each S45/S47/S52/S54 model. No training, checkpoint selection,
gradient or optimization. The original sealed S29 heldout FINAL QA and
previously inspected S26 QA are untouched, as is separate earlier S53
read-only QA; S55's new QA is never used to optimize any future version.
No JSON/QA data is imported from S29/S26/S53, and no training process
touches the current independent QA. Do not claim model-language parity
with original FP32, or speed certification from four QA examples.
Jessica chat tests, LIVE/main/Render/Supabase/APK unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
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

SOURCE_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
S47_SHA="02227f62d961b73ed1e7cf89e370d5a4852bf53839ce8fa8f608c44e286cdb43"
S52_SHA="80e52005a2c1ed97937a302c12b4d0b620264e13fe91246880544a9cb41eb4cd"
S54_SHA="88fa9aab9966d5c757013c36ab80bc713e15650dcd8659561b73d5863c022c80"
S54_BPP="0.698067535118"
S54_TOTAL=3174504
S54_MANIFEST_SIZE=683
BASE_PAIRS=(("red","bowl","drum"),("green","bowl","rope"),
            ("blue","ring","rope"),("yellow","ring","drum"))
OWNER_PAIRS=(("Ivo","Kora"),("Pavel","Juno"),("Rhea","Arlo"),
             ("Sylvie","Orin"),("Freya","Luc"),("Tara","Jonah"),
             ("Nora","Uri"),("Milo","Ines"),("Theo","Ruth"),
             ("Oren","Vika"),("Lina","Gleb"),("Sara","Ben"))
# These are neither S30/S54 TRAIN strings nor earlier S53 QA strings.
# Multi-clause references and alternate object placement measure whether
# the adapter uses binding, not whether it reproduces TRAIN words.
PROBES=(
  "A display contains two entries: [{a}, {left}] and [{b}, {right}]. Provide the color and object beside {a}, two words only.",
  "The first parcel for {b} reads {right}. The second parcel for {a} reads {left}. State the two-word text on {a}'s parcel.",
  "In a swap log, the item {right} is recorded under {b}; the item {left} is recorded under {a}. Give only {a}'s color and item.",
  "Assume the labels below refer to different packages. Package {a} is marked {left}. Package {b} is marked {right}. What two words mark {a}?",
  "A ledger notes {b} received {right}, whereas {a} received {left}. For {a}, write the exact color followed by the object.",
  "Use the recipient key: recipient {a} corresponds to {left}, recipient {b} corresponds to {right}. Identify {a}'s item in two words."
)
assert len(BASE_PAIRS)==4 and len(OWNER_PAIRS)==12 and len(PROBES)==6

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def new_independent_pairs():
    out=[]
    for color,n1,n2 in BASE_PAIRS:
        for idx,(a,b) in enumerate(OWNER_PAIRS):
            left=f"{color} {n1}"
            right=f"{color} {n2}"
            pattern=PROBES[idx%len(PROBES)]
            qleft=pattern.format(a=a,b=b,left=left,right=right)
            qright=pattern.format(a=a,b=b,left=right,right=left)
            out.append({"id":11000+len(out),
                        "left":{"prompt":qleft,"answer":left},
                        "right":{"prompt":qright,"answer":right}})
    qs=[p[k]["prompt"] for p in out for k in ("left","right")]
    if len(out)!=48 or len(qs)!=96 or len(set(qs))!=96:
        raise AssertionError("S55 must contain 48 paired, 96 distinct unseen questions")
    if any("TRAIN" in q or "FPTrain" in q or "S53" in q for q in qs):
        raise AssertionError("S55 truly unseen QA must exclude training markers")
    if any(p["left"]["answer"].split()[0]!=p["right"]["answer"].split()[0] for p in out):
        raise AssertionError("S55 opposite nouns must share color")
    return out

@torch.inference_mode()
def measure(model,sources,tokenizer,rows,tunnel,head,label):
    t0=time.perf_counter()
    preds={}
    for pair in rows:
        for side in ("left","right"):
            item=pair[side]
            key=f'{pair["id"]}_{side}'
            actual=s47.greedy(model,sources,tunnel,head,tokenizer,item["prompt"])
            expected=tokenizer.encode(item["answer"],add_special_tokens=False)+[tokenizer.eos_token_id]
            if len(expected)!=3:
                raise AssertionError("QA gold is color/noun/EOS")
            preds[key]={"expected_text":item["answer"],
                        "expected_tokens":expected,
                        "actual_tokens":actual,
                        "actual_text":tokenizer.decode(actual,skip_special_tokens=True),
                        "full_EOS_exact":actual==expected,
                        "first_token_color_exact":bool(actual) and actual[0]==expected[0],
                        "noun_second_exact":len(actual)>1 and actual[1]==expected[1],
                        "terminates_EOS":bool(actual) and actual[-1]==tokenizer.eos_token_id}
    result={
      "model":label,"qa_cases":len(preds),"qa_opposing_pairs":len(rows),
      "full_EOS_exact":sum(x["full_EOS_exact"] for x in preds.values()),
      "first_token_color_exact":sum(x["first_token_color_exact"] for x in preds.values()),
      "second_token_noun_exact":sum(x["noun_second_exact"] for x in preds.values()),
      "terminates_EOS":sum(x["terminates_EOS"] for x in preds.values()),
      "opposing_pair_both_full_EOS_exact":sum(
          preds[f'{p["id"]}_left']["full_EOS_exact"] and
          preds[f'{p["id"]}_right']["full_EOS_exact"] for p in rows),
      "opposing_pair_same_actual_tokens":sum(
          preds[f'{p["id"]}_left']["actual_tokens"]==
          preds[f'{p["id"]}_right']["actual_tokens"] for p in rows),
      "distinct_actual_responses":len({x["actual_text"] for x in preds.values()}),
      "response_mode_top5":Counter(x["actual_text"] for x in preds.values()).most_common(5),
      "elapsed_seconds_exploratory_not_a_benchmark":time.perf_counter()-t0}
    return {"summary":result,"predictions":preds}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("s45",type=Path)
    ap.add_argument("s47",type=Path)
    ap.add_argument("s52",type=Path)
    ap.add_argument("s54",type=Path)
    ap.add_argument("s54_manifest",type=Path)
    ap.add_argument("out",type=Path)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(5555)
    if (a.s45.stat().st_size!=SOURCE_BYTES or sha(a.s45)!=SOURCE_SHA
        or a.s47.stat().st_size!=22068 or sha(a.s47)!=S47_SHA
        or a.s52.stat().st_size!=32308 or sha(a.s52)!=S52_SHA
        or a.s54.stat().st_size!=32308 or sha(a.s54)!=S54_SHA):
        raise AssertionError("S55 controller/source SHAs/sizes changed after checkpoint was frozen")
    m=json.loads(a.s54_manifest.read_text(encoding="utf-8"))
    if (a.s54_manifest.stat().st_size!=S54_MANIFEST_SIZE
        or m["source_CB59_sha256"]!=SOURCE_SHA
        or m["controller_sha256"]!=S54_SHA
        or m["all_inclusive_total_bytes"]!=S54_TOTAL
        or m["all_inclusive_bpp"]!=S54_BPP
        or SOURCE_BYTES+32308+S54_MANIFEST_SIZE!=S54_TOTAL or
        float(S54_BPP)>=.7):
        raise AssertionError("S54 physically paid manifest invalid")
    rows=new_independent_pairs()
    qa_data=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    qa_sha=hashlib.sha256(qa_data.encode()).hexdigest()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    model,sources=create_small_paid_model(CB59(a.s45.read_bytes()),build_config(MODEL_SPEC))
    model.requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    tunnel47=s47.ActiveGreenTunnel(320,8)
    tunnel47.load_state_dict(load_file(str(a.s47)))
    tunnel52=s47.ActiveGreenTunnel(320,12)
    tunnel52.load_state_dict(load_file(str(a.s52)))
    tunnel54=s47.ActiveGreenTunnel(320,12)
    tunnel54.load_state_dict(load_file(str(a.s54)))
    for g in (tunnel47,tunnel52,tunnel54):
        g.eval().requires_grad_(False)
    models=[("S45_frozen_CB59",None,None),
            ("S47_accepted_rank8",tunnel47,head),
            ("S52_accepted_rank12",tunnel52,head),
            ("S54_accepted_rank12",tunnel54,head)]
    results={}
    for label,tunnel,tied_head in models:
        record=measure(model,sources,tok,rows,tunnel,tied_head,label)
        results[label]=record
        print(json.dumps({"event":"S55_NOVEL_96_FROZEN_QA",**record["summary"]},
                         ensure_ascii=False),flush=True)
    final={
       "stage":"S55_UNSEEN_48_PAIRED_96_QA_ONE_TIME_ONLY",
       "no_training_or_gradient":True,
       "no_checkpoint_selection_from_QA":True,
       "S29_S26_S53_QA_never_opened_or_reused":True,
       "S55_QA_is_now_public_inspected_and_never_tune_to_it":True,
       "S54_controller_saved_before_this_QA":True,
       "frozen_sha_S45":SOURCE_SHA,
       "frozen_sha_S47":S47_SHA,
       "frozen_sha_S52":S52_SHA,
       "frozen_sha_S54":S54_SHA,
       "S54_all_inclusive_bytes":S54_TOTAL,
       "S54_all_inclusive_bpp":S54_BPP,
       "S54_original_TRAIN_full64":29,
       "S54_new_TRAIN_full64":15,
       "S54_old_TRAIN_full8":7,
       "unseen_QA_sha256":qa_sha,
       "new_QA_96_prompts":rows,
       "frozen_results":results,
       "not_original_FP32_parity":True,
       "not_generalization_guarantee_beyond_this_one_sample":True,
       "Jessica_chat_tests_never_run":True,
       "LIVE_main_Render_Supabase_APK_unchanged":True
    }
    save(a.out/"S55_ONE_TIME_UNSEEN_96_PAIRED_QA_READ_ONLY.json",final)
    print(json.dumps({
       "event":"S55_FROZEN_INDEPENDENT_NEW_QA_FINAL",
       "S45_full":results["S45_frozen_CB59"]["summary"]["full_EOS_exact"],
       "S47_full":results["S47_accepted_rank8"]["summary"]["full_EOS_exact"],
       "S52_full":results["S52_accepted_rank12"]["summary"]["full_EOS_exact"],
       "S54_full":results["S54_accepted_rank12"]["summary"]["full_EOS_exact"],
       "S54_both_pairs":results["S54_accepted_rank12"]["summary"]["opposing_pair_both_full_EOS_exact"],
       "S54_all_inclusive_bpp":S54_BPP,
       "S29_S53_embargo_respected":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
