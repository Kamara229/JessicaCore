"""CALIBRI S79: FIRST and ONLY independent unseen test of FROZEN GPU-trained S74.

Pre-registration scientific contract:
 * This challenge is constructed AFTER S74 0.659328299132 bpp
   TRAIN checkpoint was frozen in CI #42562744529.
 * S72 and ALL previously inspected QA are permanently embargoed,
   never imported or reviewed. This study may be run ONLY ONCE.
 * S74 is immutable, and the source 36M is the same CB59/S61,
   temporarily restored to FP32 on Apple Silicon MPS FOR INFERENCE.
 * Reference S71 is frozen, and the original pretrained 135M FP32 is
   optional scientific context, NOT same architecture/same source.
 * Main objectives: 96 entirely new QA prompts, 48 opposing noun
   assignments; 48 seen TRAIN composition in completely new frames and
   48 new-noun out-of-distribution cases. All task prompts and names
   are novel, don't borrow previous QA questions/score feedback.
 * Score strict *full tokens and EOS*, first token, and each opposed
   pair fully correct; record all raw outputs including language on
   four new open questions. Separate semantic heuristic (exact answer
   phrase in output but not opposed phrase) is weak, not human rating.
 * Operational GPU->compressed CPU parity check on 8 PREEXISTING TRAIN
   prompts BEFORE independent challenge inference. If any mismatch,
   abort BEFORE seeing QA.
 * NO gradients, checkpoint selection, adaptation or retry/reuse of S79.
 * This is not a GPU speed or FP32 parity claim. LIVE/production untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer,AutoModelForCausalLM

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s47 import active_small_green_tunnel as s47
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s64.cpu_rank128_frozen_greedy_parity import instantiate
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s71 import neural_context_gate_preserve_s67 as s71
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS
from v186.pairs import tokenize_one

SEED=790079
NEW_NOUNS=("lantern","stapler")
NAMES=(("Rimala","Devron"),("Hetori","Salyon"),("Benivar","Aroli"),
       ("Morina","Calten"),("Uveren","Dalvor"),("Zelami","Ferio"))
FRAMES=(
 "INDEPENDENT Q79 exhibit {i}: According to the official label, {a}'s selected item is {left}, while {b}'s item is {right}. Reply using the two words for {a}.",
 "INDEPENDENT Q79 orchestra note {i}: The note attached to {b} reads {right}. The different note attached to {a} reads {left}. What exact two-word note belongs to {a}?",
 "INDEPENDENT Q79 warehouse dispatch {i}: Worker {a} collected product {left}; worker {b} collected product {right}. Which two-word product was collected by {a}?",
 "INDEPENDENT Q79 visual card {i}: Card {b} contains {right}, but card {a} contains {left}. Give only card {a}'s color and noun.",
 "INDEPENDENT Q79 expedition index {i}: For marker {a}, the recorded designation is {left}; for marker {b}, it is {right}. State the designation for marker {a} using two words.",
 "INDEPENDENT Q79 puzzle clue {i}: The key for {b} corresponds to {right}; separately the key for {a} corresponds to {left}. Transcribe the two-word key assigned to {a}.",
)
OPEN_QUESTIONS=(
  "Why is it important to check a bicycle's brakes before riding?",
  "Explain to a child why people should recycle paper.",
  "Write one clear instruction for keeping a kitchen counter clean.",
  "What could help a team resolve a disagreement fairly?",
)
assert len(FRAMES)==len(NAMES)==6 and len(OPEN_QUESTIONS)==4
FROZEN_S74_SOURCE_RUN=42562744529
S74_SELECTED_STEP=48

def sha(buf):return hashlib.sha256(buf).hexdigest()

def new_challenge(tok):
    assert set(s30.TRAIN_ELIGIBLE)=={"red","green","blue","yellow"}
    assert not any(x in s30.TRAIN_ELIGIBLE[c] for x in NEW_NOUNS for c in s30.TRAIN_ELIGIBLE)
    pairs=[]
    serial=79000
    for color in ("red","green","blue","yellow"):
        for cohort,nouns in (("seen_composition_unseen_frame",s30.TRAIN_ELIGIBLE[color]),
                             ("new_object_unseen_frame",NEW_NOUNS)):
            x,y=[f"{color} {z}" for z in nouns]
            for idx,frame in enumerate(FRAMES):
                name,other=NAMES[idx]
                q0=frame.format(i=serial,a=name,b=other,left=x,right=y)
                q1=frame.format(i=serial,a=name,b=other,left=y,right=x)
                a=tokenize_one(tok,q0,x)
                b=tokenize_one(tok,q1,y)
                if a["first_expected_id"]!=b["first_expected_id"] or q0==q1:
                    raise AssertionError("S79 must keep opposing noun pair, same first token")
                pairs.append({"pair_id":serial,"color":color,"cohort":cohort,
                              "left":a,"right":b})
                serial+=1
    if (len(pairs)!=48 or
        len({p[side]["question"] for p in pairs for side in ("left","right")})!=96):
        raise AssertionError("S79 exact 96 distinct new independent questions changed")
    return pairs

def build_unchanged_controllers(weights,with_first_increment):
    first=s47.ActiveGreenTunnel(320,2)
    later=s47.ActiveGreenTunnel(320,128)
    first.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("first.")})
    later.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith("later.")})
    mixer=s71.NeuralContextMixer(later,
        weights["later.up.weight"]+weights["context_delta.up.weight"])
    mixer.context_alpha.load_state_dict({
        "weight":weights["context_gate.weight"],
        "bias":weights["context_gate.bias"]})
    first.eval().requires_grad_(False)
    mixer.eval().requires_grad_(False)
    if with_first_increment:
        adapter=s73.FirstTokenResidual16(first)
        extras={k.removeprefix("first_increment."):v for k,v in weights.items()
                if k.startswith("first_increment.")}
        if set(extras)!={"down.weight","up.weight","gate.weight","gate.bias"}:
            raise AssertionError("S74 new trained rank16 must exist")
        adapter.load_state_dict({**{f"frozen.{k}":v for k,v in first.state_dict().items()},
                                 **extras},strict=True)
        first=adapter.eval().requires_grad_(False)
    return first,mixer

@torch.inference_mode()
def gpu_generate(model,tok,first,later,head,question,limit):
    prompt=tok.apply_chat_template([{"role":"user","content":question}],
                                   tokenize=False,add_generation_prompt=True)
    context=torch.tensor([tok.encode(prompt,add_special_tokens=False)],
                         device="mps",dtype=torch.long)
    tokens=[]
    past=None
    start=time.perf_counter()
    for step in range(limit):
        data=context if step==0 else torch.tensor(
            [[tokens[-1]]],device="mps",dtype=torch.long)
        output=model(input_ids=data,past_key_values=past,
                     use_cache=True,output_hidden_states=True)
        if output.hidden_states[-1].device.type!="mps":
            raise AssertionError("S79 intended GPU forward ran elsewhere")
        past=output.past_key_values
        if past is None:raise AssertionError("S79 GPU KV-cache not used")
        h=output.hidden_states[-1][0,-1]
        logits=output.logits[0,-1]
        delta=first(h) if step==0 else later(h)
        next_id=int((logits+F.linear(delta,head)).argmax().item())
        tokens.append(next_id)
        del output
        if next_id==tok.eos_token_id:break
    torch.mps.synchronize()
    return tokens,time.perf_counter()-start

@torch.inference_mode()
def pretrained_generate(model,tok,question,limit):
    prompt=tok.apply_chat_template([{"role":"user","content":question}],
                                   tokenize=False,add_generation_prompt=True)
    ids=torch.tensor([tok.encode(prompt,add_special_tokens=False)],
                      device="mps",dtype=torch.long)
    past=None
    result=[]
    t=time.perf_counter()
    for step in range(limit):
        new=ids if step==0 else torch.tensor([[result[-1]]],device="mps")
        output=model(input_ids=new,past_key_values=past,use_cache=True)
        past=output.past_key_values
        token=int(output.logits[0,-1].argmax().item())
        result.append(token)
        del output
        if token==tok.eos_token_id:break
    torch.mps.synchronize()
    return result,time.perf_counter()-t

def phrase_present(answer,desired,undesired):
    # Weak, lexical-only: does the response explicitly contain the target
    # two-word label without also citing its opposing possible label?
    normalize=lambda text:" ".join(re.findall(r"[a-zA-Z]+",text.lower()))
    doc=" "+normalize(answer)+" "
    expected=" "+normalize(desired)+" "
    wrong=" "+normalize(undesired)+" "
    return expected in doc and wrong not in doc

def score(rows,pairs):
    scores={}
    for cohort in ("seen_composition_unseen_frame","new_object_unseen_frame","all"):
        group=[p for p in pairs if cohort=="all" or p["cohort"]==cohort]
        ids=[f'{p["pair_id"]}_{side}' for p in group for side in ("left","right")]
        complete=lambda k:rows[k]["strict_full_plus_EOS"]
        first=lambda k:rows[k]["exact_first_answer_token"]
        sem=lambda k:rows[k]["weak_unambiguous_semantic_phrase"]
        scores[cohort]={
            "exact_full_plus_EOS":sum(complete(k) for k in ids),
            "first_answer_token":sum(first(k) for k in ids),
            "weak_unambiguous_semantic_phrase":sum(sem(k) for k in ids),
            "out_of_questions":len(ids),
            "both_opposing_full":sum(
                complete(f'{p["pair_id"]}_left') and complete(f'{p["pair_id"]}_right')
                for p in group),
            "out_of_pairs":len(group),
        }
    return scores

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_source_CB59",type=Path)
    p.add_argument("S71_full_paid_controller",type=Path)
    p.add_argument("S74_frozen_gpu_paid_controller",type=Path)
    p.add_argument("S74_all_physical_manifest",type=Path)
    p.add_argument("S61_immutable_compressed_source",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED);torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S79 real Apple MPS GPU required, no CPU fallback")
    raw=a.S45_source_CB59.read_bytes()
    baseline_bytes=a.S71_full_paid_controller.read_bytes()
    controller_bytes=a.S74_frozen_gpu_paid_controller.read_bytes()
    compact=a.S61_immutable_compressed_source.read_bytes()
    manifest=json.loads(a.S74_all_physical_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or sha(raw)!=s65.S45_SHA or
        sha(baseline_bytes)!=s73.S71_SHA or
        manifest["S71_frozen_controller_SHA256"]!=s73.S71_SHA or
        manifest["S74_GPU_full_controller_SHA256"]!=sha(controller_bytes) or
        manifest["S61_unchanged_exact_compressed_source_SHA256"]!=sha(compact) or
        manifest["selected_first_restoration_STEP_TRAIN_only"]!=S74_SELECTED_STEP or
        manifest["physical_all_inclusive_bytes"]!=len(compact)+len(controller_bytes)+
                                                  a.S74_all_physical_manifest.stat().st_size or
        float(manifest["physical_all_inclusive_bpp"])>.7):
        raise AssertionError("S79 refuses any unfrozen or unpaid S74 controller")
    w71=load_file(str(a.S71_full_paid_controller))
    w74=load_file(str(a.S74_frozen_gpu_paid_controller))
    for k,v in w71.items():
        if k not in w74 or not torch.equal(v,w74[k]):
            raise AssertionError("S79 S74 must freeze all original S71 weights")
    if set(w74)-set(w71)!={"first_increment.down.weight",
                            "first_increment.up.weight",
                            "first_increment.gate.weight",
                            "first_increment.gate.bias"}:
        raise AssertionError("S79 wrong physical S74 added parameter set")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    f71,l71=build_unchanged_controllers(w71,False)
    f74,l74=build_unchanged_controllers(w74,True)
    cpu_model,cpu_sources,first_cpu,later_cpu,cpu_head=instantiate(raw,w71,128)
    ref_mixer=s71.NeuralContextMixer(later_cpu,w71["later.up.weight"]+w71["context_delta.up.weight"])
    ref_mixer.context_alpha.load_state_dict({
        "weight":w71["context_gate.weight"],"bias":w71["context_gate.bias"]})
    cpu_s74=s73.FirstTokenResidual16(first_cpu)
    inc={k.removeprefix("first_increment."):v for k,v in w74.items()
         if k.startswith("first_increment.")}
    cpu_s74.load_state_dict({**{f"frozen.{k}":v for k,v in first_cpu.state_dict().items()},**inc},strict=True)
    dense=s30.dense_from_paid(CB59(raw)).to("mps").eval().requires_grad_(False)
    if sum(z.numel() for z in dense.parameters())!=SOURCE_PARAMS:
        raise AssertionError("S79 must temporarily reconstruct actual 36M source")
    f71,l71=f71.to("mps").eval(),l71.to("mps").eval()
    f74,l74=f74.to("mps").eval(),l74.to("mps").eval()
    head=dense.lm_head.weight.detach()
    if head.device.type!="mps":raise AssertionError("S79 original whole-model LM head not on GPU")

    # Safety check: absolutely NO unseen QA touched until REAL S74 and
    # S71 MPS full native tokens match compressed model old TRAIN.
    historical_train=s30.make_training_pairs(tok)[:4]
    checked=0
    for p0 in historical_train:
        for side in ("left","right"):
            q=p0[side]["question"]
            for name,first_cpu_,later_cpu_,first_gpu_,later_gpu_ in (
                ("S71",first_cpu,ref_mixer,f71,l71),
                ("S74",cpu_s74,ref_mixer,f74,l74)):
                ref=s56.greedy_first_active(cpu_model,cpu_sources,first_cpu_,later_cpu_,cpu_head,tok,q)
                generated,_=gpu_generate(dense,tok,first_gpu_,later_gpu_,head,q,s56.MAX_NEW)
                if ref!=generated:
                    raise AssertionError("S79 GPU/true compressed CPU mismatch on old TRAIN BEFORE QA: "+name)
                checked+=1
    if checked!=16:raise AssertionError("S79 GPU old TRAIN preflight insufficient")
    print(json.dumps({"event":"S79_GPU_S71_S74_OLD_TRAIN_PREFLIGHT_EXACT_PARITY",
       "real_compressed_CPU_to_GPU_full_token_sequences_identical":checked,
       "no_independent_QA_inference_yet":True}),flush=True)

    # QA is generated, fixed and persisted BEFORE the first unseen inference.
    challenge=new_challenge(tok)
    frozen={
       "comparison":"FROZEN_S71_v_FROZEN_S74_v_ORIGINAL_PRETRAINED_135M_FP32",
       "candidate_sha256":sha(controller_bytes),
       "source_sha256":sha(raw),
       "opposed_pairs":[{"id":p["pair_id"],"cohort":p["cohort"],
                         "left":{"question":p["left"]["question"],"answer":p["left"]["answer"]},
                         "right":{"question":p["right"]["question"],"answer":p["right"]["answer"]}}
                        for p in challenge],
       "new_open_prompts":OPEN_QUESTIONS}
    payload=json.dumps(frozen,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    (a.output_dir/"S79_FIRST_AND_ONLY_PERMANENTLY_EMBARGOED_NEW_QA.json").write_bytes(payload)
    qa_sha=sha(payload)
    print(json.dumps({"event":"S79_NEW_INDEPENDENT_QA_FROZEN_BEFORE_FIRST_INFERENCE",
       "qa_sha256":qa_sha,"task_questions":96,"opposing_pairs":48,
       "open_questions":4,
       "S74_fixed_controller_SHA256":sha(controller_bytes),
       "old_QA_never_loaded":True}),flush=True)

    fp32=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
        attn_implementation="eager").to("mps").eval()
    fp32.requires_grad_(False)
    params=sum(t.numel() for t in fp32.parameters())
    if not 100_000_000<params<150_000_000:
        raise AssertionError("S79 canonical pretrained reference must be ~135M, not CALIBRI same-architecture")
    systems={
        "S71_original_physically_paid_36M":(f71,l71),
        "S74_real_GPU_trained_physically_paid_36M":(f74,l74),
        "ORIGINAL_135M_pretrained_FP32_different_architecture":None,
    }
    outputs={name:{} for name in systems}
    seconds={name:[] for name in systems}
    for j,pair in enumerate(challenge):
        for side in ("left","right"):
            item=pair[side];q=item["question"]
            key=f'{pair["pair_id"]}_{side}'
            gold_ids=tok.encode(item["answer"],add_special_tokens=False)+[tok.eos_token_id]
            alt=pair["right" if side=="left" else "left"]["answer"]
            order=list(systems) if (j+(side=="right"))%2==0 else list(reversed(systems))
            for name in order:
                if systems[name] is None:
                    tokens,elapsed=pretrained_generate(fp32,tok,q,32)
                else:
                    f,l=systems[name]
                    tokens,elapsed=gpu_generate(dense,tok,f,l,head,q,s56.MAX_NEW)
                text_answer=tok.decode(tokens,skip_special_tokens=True)
                outputs[name][key]={
                    "actual_generated_ids":tokens,
                    "actual_generated_text":text_answer,
                    "gold_ids_including_EOS":gold_ids,
                    "gold_answer":item["answer"],
                    "opposite_answer":alt,
                    "exact_first_answer_token":bool(tokens) and tokens[0]==gold_ids[0],
                    "strict_full_plus_EOS":tokens==gold_ids,
                    "weak_unambiguous_semantic_phrase":
                        phrase_present(text_answer,item["answer"],alt),
                    "EOS_emitted":bool(tokens) and tokens[-1]==tok.eos_token_id,
                    "cohort":pair["cohort"],
                }
                seconds[name].append(elapsed)
        if (j+1)%8==0:
            print(json.dumps({"event":"S79_SINGLE_USE_FROZEN_QA_PROGRESS",
                "pairs_done":j+1,"out_of":len(challenge)}),flush=True)
    grades={name:score(rows,challenge) for name,rows in outputs.items()}
    ref=outputs["S71_original_physically_paid_36M"]
    cand=outputs["S74_real_GPU_trained_physically_paid_36M"]
    win=[k for k in ref if not ref[k]["strict_full_plus_EOS"] and cand[k]["strict_full_plus_EOS"]]
    lost=[k for k in ref if ref[k]["strict_full_plus_EOS"] and not cand[k]["strict_full_plus_EOS"]]
    open_outputs=[]
    for question in OPEN_QUESTIONS:
        row={"question":question}
        for name,system in systems.items():
            if system is None:
                got,t=pretrained_generate(fp32,tok,question,48)
            else:
                f,l=system
                got,t=gpu_generate(dense,tok,f,l,head,question,48)
            row[name]={"real_generated_token_ids":got,
                       "real_generated_text":tok.decode(got,skip_special_tokens=True),
                       "EOS_emitted":bool(got) and got[-1]==tok.eos_token_id}
        open_outputs.append(row)
    report={
        "stage":"S79_FIRST_AND_ONLY_INDEPENDENT_QA_FROZEN_S74_WITH_ORIGINAL_PRETRAINED_FP32_CONTEXT",
        "sealed_QA_SHA256":qa_sha,
        "S74_frozen_controller_sha256":sha(controller_bytes),
        "all_inclusive_S74_physical_bpp":manifest["physical_all_inclusive_bpp"],
        "strict_full_first_pair_semantic_heuristic_scores":grades,
        "S74_new_full_correct_vs_S71":win,
        "S74_lost_full_correct_vs_S71":lost,
        "S74_cpu_median_speed_NOT_measured_in_unseen":True,
        "real_GPU_median_speed_S74_over_S71":
            statistics.median(seconds["S71_original_physically_paid_36M"])/
            statistics.median(seconds["S74_real_GPU_trained_physically_paid_36M"]),
        "canonical_FP32_is_DIFFERENT_135M_architecture_NOT_iso_36M_parity":True,
        "semantic_exact_phrase_only_not_human_language_understanding":True,
        "unscored_open_language_outputs_for_human_review":open_outputs,
        "all_real_generated_token_sequences":outputs,
        "QA_permanently_EMBARGOED_from_future_training_selection_and_reuse":True,
        "NO_new_training_or_QA_retry_allowed":True,
        "LIVE_main_Jessica_Render_Supabase_unchanged":True,
    }
    (a.output_dir/"S79_FROZEN_ONESHOT_EVAL_FINAL_REAL_MODEL_OUTPUTS.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"event":"S79_FINAL_SEALED_GENERALIZATION_RESULT",
        "QA_sha":qa_sha,"strict_first_and_both_scores":grades,
        "S74_new_full_correct":len(win),"S74_lost_full_correct":len(lost),
        "all_prior_and_current_QA_permanently_embargoed":True,
        "not_FP32_iso_architecture_parity":True}),flush=True)

if __name__=="__main__":main()
