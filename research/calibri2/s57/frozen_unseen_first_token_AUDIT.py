"""CALIBRI S57: sealed one-time first-token generalization check, READ ONLY.

S56 accepted TRAIN only, CI #42518099946: original full 29->42/64,
extra TRAIN full 15->27/64, 7 old full kept, >= original S54
both-correct pairs, physically paid 0.699552507279 bpp. It also improved
first colors 34->58/64 and 17->46/64 on TRAIN. These numbers do not prove
generalization. Evaluate whether adding a PAID first-token Green Tunnel
helps on 96 independent novel one-time QA prompts, with SAME physically
frozen source+S54+S56 and no gradient or checkpoint selection.

Do NOT import or access sealed S29, previous S26, S53 or S55 QA:
each has been inspected or reserved, and may NEVER steer optimizer or
select a subsequent checkpoint. After this first read-only S57 audit,
its own QA is equally permanently embargoed for future tuning. All S57
new questions constructed here use new names, new formats, original
8 TRAIN-eligible color/noun combinations, and no TRAIN tags.
No tests in Jessica chat; no LIVE, Render, Supabase, APK, main changes.
"""
from __future__ import annotations
import argparse,hashlib,json,time
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
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from v174.run import save

S45_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
S54_SHA="88fa9aab9966d5c757013c36ab80bc713e15650dcd8659561b73d5863c022c80"
S56_SHA="503926839336ed7161786fa6d8ccb2a4ae6fee876cb566d0e52db80ff128f6d4"
S56_MANIFEST_SHA="bb15a2638f9beb33318bbd97e156973043b220e8d8cda81874a69b61798d7c2b"
S56_BPP="0.699552507279"
S56_TOTAL=3181257
S56_SIDECAR_BYTES=39040
S56_MANIFEST_BYTES=704
COLORS=(("red","bowl","drum"),("green","bowl","rope"),
        ("blue","ring","rope"),("yellow","ring","drum"))
PEOPLE=(("Daria","Maksim"),("Jasmin","Theo"),("Antonella","Moris"),
        ("Carlo","Iman"),("Kira","Benno"),("Valeria","Gael"),
        ("Polina","Emil"),("Sofia","Ilia"),("Natalie","Cyril"),
        ("Mira","Levon"),("Anabel","Dmitri"),("Esme","Tobias"))
# ALL six QA sentence layouts written separately AFTER S56 was saved.
# Diverse order/context; never train or tune against these strings.
FRAMES=(
 "Two containers appear in a handwritten note. The one labeled {a} is {left}. The other, labeled {b}, is {right}. Read back only {a}'s color and object.",
 "Under {b}, the museum tag says {right}; under {a}, the tag says {left}. Give the exact two words from {a}'s tag.",
 "The event guest list pairs {a} with {left} and pairs {b} with {right}. Reply with just the color and noun paired with {a}.",
 "Someone compared the descriptions for {b} ({right}) and {a} ({left}). When asked for {a}, answer in precisely two words: color then object.",
 "For this memory puzzle, {b} was shown {right}. Meanwhile {a} was shown {left}. Identify what {a} saw in two words.",
 "The reference book gives two separate symbols: {a} means {left}, but {b} means {right}. Write the color and thing encoded by {a}."
)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def novel_96():
    pairs=[]
    for color,n1,n2 in COLORS:
        for i,(a,b) in enumerate(PEOPLE):
            l=f"{color} {n1}"
            r=f"{color} {n2}"
            fmt=FRAMES[i%len(FRAMES)]
            left=fmt.format(a=a,b=b,left=l,right=r)
            right=fmt.format(a=a,b=b,left=r,right=l)
            pairs.append({"id":12000+len(pairs),
                         "left":{"question":left,"answer":l},
                         "right":{"question":right,"answer":r}})
    qs=[p[s]["question"] for p in pairs for s in ("left","right")]
    if len(pairs)!=48 or len(qs)!=96 or len(set(qs))!=96:
        raise AssertionError("must have 96 new independent questions")
    if any("TRAIN" in q or "FPTrain" in q for q in qs):
        raise AssertionError("novel QA contains TRAIN leakage")
    return pairs

@torch.inference_mode()
def evaluate(model,sources,tok,head,pairs,first,later,label):
    rows={}
    start=time.perf_counter()
    for p in pairs:
        for side in ("left","right"):
            item=p[side]
            key=f'{p["id"]}_{side}'
            if first is None:
                actual=s47.greedy(model,sources,later,head,tok,item["question"])
            else:
                actual=s56.greedy_first_active(model,sources,first,later,head,tok,item["question"])
            expected=tok.encode(item["answer"],add_special_tokens=False)+[tok.eos_token_id]
            if len(expected)!=3:
                raise AssertionError("expected precisely color noun EOS")
            rows[key]={
              "expected":item["answer"],
              "expected_token_ids_including_EOS":expected,
              "actual_token_ids":actual,
              "actual":tok.decode(actual,skip_special_tokens=True),
              "strict_full_EOS_correct":actual==expected,
              "first_color_correct":bool(actual) and actual[0]==expected[0],
              "second_noun_correct":len(actual)>1 and actual[1]==expected[1],
              "ended_in_EOS":bool(actual) and actual[-1]==tok.eos_token_id
            }
    summary={
       "label":label,
       "fresh_cases":len(rows),
       "fresh_contrasting_pairs":len(pairs),
       "strict_full_correct":sum(int(v["strict_full_EOS_correct"]) for v in rows.values()),
       "first_color_correct":sum(int(v["first_color_correct"]) for v in rows.values()),
       "second_noun_correct":sum(int(v["second_noun_correct"]) for v in rows.values()),
       "EOS_terminated":sum(int(v["ended_in_EOS"]) for v in rows.values()),
       "both_paired_responses_strict_correct":sum(
          rows[f'{p["id"]}_left']["strict_full_EOS_correct"] and
          rows[f'{p["id"]}_right']["strict_full_EOS_correct"] for p in pairs),
       "paired_same_full_response_tokens":sum(
          rows[f'{p["id"]}_left']["actual_token_ids"]==
          rows[f'{p["id"]}_right']["actual_token_ids"] for p in pairs),
       "distinct_response_texts":len(set(r["actual"] for r in rows.values())),
       "most_repeated_texts":Counter(r["actual"] for r in rows.values()).most_common(6),
       "wall_seconds_single_run_exploratory_not_CPU_certification":time.perf_counter()-start
    }
    return {"summary":summary,"all_96_full_answers":rows}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("S45_core",type=Path)
    parser.add_argument("S54_controller",type=Path)
    parser.add_argument("S56_dual",type=Path)
    parser.add_argument("S56_manifest",type=Path)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(5757)
    torch.set_num_threads(2)
    if (a.S45_core.stat().st_size!=SOURCE_BYTES or digest(a.S45_core)!=S45_SHA or
        a.S54_controller.stat().st_size!=32308 or digest(a.S54_controller)!=S54_SHA or
        a.S56_dual.stat().st_size!=S56_SIDECAR_BYTES or digest(a.S56_dual)!=S56_SHA or
        a.S56_manifest.stat().st_size!=S56_MANIFEST_BYTES or
        digest(a.S56_manifest)!=S56_MANIFEST_SHA):
        raise AssertionError("physical weights frozen SHA or size changed")
    manifest=json.loads(a.S56_manifest.read_text(encoding="utf8"))
    if (manifest["all_inclusive_total_bytes"]!=S56_TOTAL or
        manifest["all_inclusive_bpp"]!=S56_BPP or
        manifest["dual_controller_sha256"]!=S56_SHA or
        manifest["source_CB59_sha256"]!=S45_SHA or
        SOURCE_BYTES+S56_SIDECAR_BYTES+S56_MANIFEST_BYTES!=S56_TOTAL or
        float(S56_BPP)>=.7):
        raise AssertionError("S56 inclusive paid size budget invalid")
    qa=novel_96()
    qa_sh=hashlib.sha256(json.dumps(qa,sort_keys=True).encode()).hexdigest()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    model,sources=create_small_paid_model(CB59(a.S45_core.read_bytes()),build_config(MODEL_SPEC))
    model.requires_grad_(False)
    head=model.lm_head.source().detach().cpu().contiguous()
    s54gate=s47.ActiveGreenTunnel(320,12)
    s54gate.load_state_dict(load_file(str(a.S54_controller)))
    s54gate.eval().requires_grad_(False)
    tensors=load_file(str(a.S56_dual))
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,12)
    first.load_state_dict({k.split(".",1)[1]:v for k,v in tensors.items() if k.startswith("first.")})
    later.load_state_dict({k.split(".",1)[1]:v for k,v in tensors.items() if k.startswith("later.")})
    first.eval().requires_grad_(False)
    later.eval().requires_grad_(False)
    # The S56 later controller must be an EXACT physical copy of S54.
    if any(not torch.equal(x,y) for (k,x),(j,y) in zip(
             sorted(s54gate.state_dict().items()),sorted(later.state_dict().items()))
           if k==j):
        raise AssertionError("S56 frozen S54 post-first paid neural weights changed")
    base=evaluate(model,sources,tok,head,qa,None,s54gate,"S54_saved_frozen_single_adapter")
    print(json.dumps({"event":"S57_FROZEN_NOVEL_QA","model":"S54",**base["summary"]},ensure_ascii=False),flush=True)
    new=evaluate(model,sources,tok,head,qa,first,later,"S56_saved_frozen_dual_first_later")
    print(json.dumps({"event":"S57_FROZEN_NOVEL_QA","model":"S56",**new["summary"]},ensure_ascii=False),flush=True)
    report={
      "stage":"S57_READ_ONLY_FIRST_TOKEN_NEVER_SEEN_QA",
      "completely_separate_from_original_AND_expanded_TRAIN":True,
      "S29_S26_S53_S55_QA_never_imported_or_read":True,
      "S57_QA_must_never_be_used_for_subsequent_optimization":True,
      "physical_S56_frozen_before_QA":True,
      "no_training_optimizer_gradients_checkpoint_selection":True,
      "frozen_SHA_S45":S45_SHA,"frozen_SHA_S54":S54_SHA,
      "frozen_SHA_S56_dual":S56_SHA,
      "S56_physically_paid_all_inclusive_bpp":S56_BPP,
      "qa_sha":qa_sh,"qa_96":qa,
      "actual_full_results_S54":base,
      "actual_full_results_S56":new,
      "Jessica_chat_test_not_run":True,
      "LIVE_main_Render_Supabase_APK_unchanged":True}
    save(a.output/"S57_NOVEL_96_READ_ONLY_FIRST_TOKEN_AUDIT.json",report)
    print(json.dumps({"event":"S57_FINAL_READONLY_96_PAIRED_NOVEL",
       "S54_full":base["summary"]["strict_full_correct"],
       "S56_full":new["summary"]["strict_full_correct"],
       "S54_first":base["summary"]["first_color_correct"],
       "S56_first":new["summary"]["first_color_correct"],
       "S54_pairs_both":base["summary"]["both_paired_responses_strict_correct"],
       "S56_pairs_both":new["summary"]["both_paired_responses_strict_correct"],
       "S56_bpp":S56_BPP,
       "S29_S53_S55_embargo_untouched":True},ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
