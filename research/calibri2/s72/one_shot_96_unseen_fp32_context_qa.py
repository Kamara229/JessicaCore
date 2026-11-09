"""CALIBRI S72: FIRST/ONLY new independent QA after S71 TRAIN freeze.

One sealed instrument, one evaluation. The data below are NEW EVAL,
created ONLY after S71 artifact 13638559856 was saved, fixed epoch 60.
Neither S68, S55, S53, S29, S26 QA, their prompts, nor their answer data
are imported, read, searched or reused. This file itself and its output
become PERMANENTLY EMBARGOED for future gradients, model selection,
prompts, architecture search, and hyperparameter tuning.

Compare exactly same unseen 96 objectives (48 opposed pairs) on:
 - frozen S67 physical <=0.7 bpp reference;
 - frozen, fully physically paid S71 controller (TRAIN only);
 - canonical ORIGINAL pretrained SmolLM2-135M-Instruct FP32 reference
   read-only and DIFFERENT architecture/parameter count. The 135M
   reference is contextual, NOT same-architecture parity.

Score TRUE entire autoregressive outputs incl EOS; first-color accuracy;
pairwise both-correct and new-correct/lost-correct vs S67. Also generate
4 new open answers from each real model for future manual assessment.
For open prompts no invented automatic coherence scores.

No model weights trained, no checkpoint search, no QA retry after
seeing results. CPU-only (no Kaggle / Groq), no Jessica chat/LIVE.
"""
from __future__ import annotations
import argparse,hashlib,json,re,statistics,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer,AutoModelForCausalLM

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s64.cpu_rank128_frozen_greedy_parity import instantiate
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s71 import neural_context_gate_preserve_s67 as s71
from v186.pairs import tokenize_one
from v176.operators import clear

SEED=720072
COLORS=("red","green","blue","yellow")
OBJECTS=("bowl","drum","ring","rope")
ACTORS=(
    ("Dervin","Polara"),
    ("Vessim","Taleno"),
    ("Rundel","Mevora"),
    ("Seltan","Lurvia"),
    ("Hesmon","Bolira"),
    ("Tarvin","Zefira"),
)
# Different names and wording from S69 TRAIN and every earlier EVAL.
# Six new syntactic frames with target mention in different positions.
FRAMES=(
  "QA72 museum inventory {i}: A container assigned to {a} carries the tag {left}; the container belonging to {b} carries {right}. Give only the two-word tag for {a}.",
  "QA72 classroom chart {i}: For {b}, the object code is {right}; for {a}, the code is {left}. Which two words identify {a}'s object?",
  "QA72 theater wardrobe {i}: The note next to {a} says {left}, but the note next to {b} says {right}. Reproduce the two-word note next to {a}.",
  "QA72 workshop bin {i}: {b} was allocated {right} while {a} was allocated {left}. Reply with {a}'s color and object, no other words.",
  "QA72 botanical exhibit {i}: The register states [{a} -> {left}] and [{b} -> {right}]. Transcribe the exact two words assigned to {a}.",
  "QA72 library catalog {i}: {a} is represented by the pair {left}. The separate listing for {b} shows {right}. What is the two-word pair for {a}?",
)
# Completely new, unlabeled, broad open questions. Human interpretation
# is mandatory; not useful as automatic checkpoint-choice signal.
OPEN_QUESTIONS=(
  "In two brief sentences, explain why neighbors might share tools instead of buying duplicates.",
  "Describe in plain English how a bicycle chain helps transfer motion to a wheel.",
  "Explain briefly how to check whether a plant has enough water without harming it.",
  "In two short sentences, describe a fair way to divide a group task among friends.",
)
assert len(FRAMES)==len(ACTORS)==6
assert len(OPEN_QUESTIONS)==4

def sha(b):return hashlib.sha256(b).hexdigest()

def make_new_blind_pairs(tok):
    assert set(s30.TRAIN_ELIGIBLE)==set(COLORS)
    rows=[]
    serial=72000
    for color in COLORS:
        seen=tuple(s30.TRAIN_ELIGIBLE[color])
        unseen=tuple(x for x in OBJECTS if x not in seen)
        if len(seen)!=2 or len(unseen)!=2:
            raise AssertionError("TRAIN composition provenance drift")
        for cohort,nouns in (("TRAIN_composition_new_prompt",seen),
                             ("withheld_composition_new_prompt",unseen)):
            gold0,gold1=[f"{color} {n}" for n in nouns]
            for idx,frame in enumerate(FRAMES):
                a,b=ACTORS[idx]
                q0=frame.format(i=serial,a=a,b=b,left=gold0,right=gold1)
                q1=frame.format(i=serial,a=a,b=b,left=gold1,right=gold0)
                if q0==q1 or gold0==gold1 or any("TRAIN route 6901" in z for z in (q0,q1)):
                    raise AssertionError("invalid independent opposed pair")
                v0=tokenize_one(tok,q0,gold0)
                v1=tokenize_one(tok,q1,gold1)
                if v0["first_expected_id"]!=v1["first_expected_id"]:
                    raise AssertionError("first answer token should be same for opposed pair")
                rows.append({"pair_id":serial,"cohort":cohort,"color":color,
                             "left":v0,"right":v1})
                serial+=1
    if len(rows)!=48 or len({p[s]["question"] for p in rows for s in ("left","right")})!=96:
        raise AssertionError("challenge must have exactly 96 distinct questions")
    return rows

def run_small(state,tok,question):
    model,sources,first,later,head=state
    t=time.perf_counter()
    tokens=s56.greedy_first_active(model,sources,first,later,head,tok,question)
    return tokens,time.perf_counter()-t

@torch.inference_mode()
def run_pretrained_135m_fp32(model,tok,question,cap):
    # Exact shared tokenizer/greedy/EOS protocol, but not a comparable size.
    prompt=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    ids=torch.tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)[None]
    tokens=[]
    t=time.perf_counter()
    for _ in range(cap):
        output=model(input_ids=ids,use_cache=False)
        logits=output.logits[0,-1]
        if not torch.isfinite(logits).all():
            raise AssertionError("nonfinite pretrained FP32 logits")
        nxt=int(logits.argmax())
        tokens.append(nxt)
        del output
        if nxt==tok.eos_token_id:break
        ids=torch.cat((ids,torch.tensor([[nxt]],dtype=torch.long)),dim=1)
    return tokens,time.perf_counter()-t

def answer_summary(rows,pairs,tag):
    subsets=(("TRAIN_composition_new_prompt",),
             ("withheld_composition_new_prompt",),
             ("TRAIN_composition_new_prompt","withheld_composition_new_prompt"))
    scores={}
    for cohorts in subsets:
        key="all" if len(cohorts)==2 else cohorts[0]
        subset=[p for p in pairs if p["cohort"] in cohorts]
        ids=[f'{p["pair_id"]}_{side}' for p in subset for side in ("left","right")]
        good=lambda k:rows[k]["full_correct_including_EOS"]
        first=lambda k:rows[k]["first_color_correct"]
        pairs_both=sum(good(f'{p["pair_id"]}_left') and good(f'{p["pair_id"]}_right') for p in subset)
        scores[key]={
            "full_answers":sum(good(i) for i in ids),
            "out_of":len(ids),
            "first_color":sum(first(i) for i in ids),
            "both_opposed_pair_answers":pairs_both,
            "out_of_opposed_pairs":len(subset)}
    return scores

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("S45_immutable_CB59",type=Path)
    ap.add_argument("S67_immutable_rank128",type=Path)
    ap.add_argument("S67_manifest",type=Path)
    ap.add_argument("S71_frozen_neural_candidate",type=Path)
    ap.add_argument("S71_fully_paid_manifest",type=Path)
    ap.add_argument("S61_compressed_envelope",type=Path)
    ap.add_argument("output_dir",type=Path)
    x=ap.parse_args()
    x.output_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw=x.S45_immutable_CB59.read_bytes()
    s67_bytes=x.S67_immutable_rank128.read_bytes()
    s71_bytes=x.S71_frozen_neural_candidate.read_bytes()
    compact=x.S61_compressed_envelope.read_bytes()
    man67=json.loads(x.S67_manifest.read_text())
    man71=json.loads(x.S71_fully_paid_manifest.read_text())
    if not (
       sha(raw)==s65.S45_SHA and len(raw)==s71.SOURCE_BYTES and
       sha(s67_bytes)==s69.CANDIDATE_SHA and
       man67["candidate_S67_rank128_controller_sha256"]==sha(s67_bytes) and
       man67["physical_CB61_source_sha256"]==sha(compact) and
       float(man67["all_inclusive_bpp"])<=.7 and
       man71["parent_S67_rank128_controller_SHA256"]==sha(s67_bytes) and
       man71["combined_controller_rank128_with_context_alpha_SHA256"]==sha(s71_bytes) and
       man71["physical_S61_CB61_SHA256"]==sha(compact) and
       man71["physical_all_inclusive_bytes"]==len(compact)+len(s71_bytes)+x.S71_fully_paid_manifest.stat().st_size and
       float(man71["physical_all_inclusive_bpp"])<=.7
    ):raise AssertionError("S72 refuses unfrozen/unpaid source or sidecar")
    weights67=load_file(str(x.S67_immutable_rank128))
    weights71=load_file(str(x.S71_frozen_neural_candidate))
    if set(weights71)!=(set(weights67)|{"context_gate.weight","context_gate.bias","context_delta.up.weight"}):
        raise AssertionError("S71 must have exact 321 paid gated parameters and frozen delta")
    if any(not torch.equal(v,weights71[k]) for k,v in weights67.items()):
        raise AssertionError("S71 changed original paid S67 weights")
    if weights71["context_gate.weight"].shape!=(1,320) or weights71["context_gate.bias"].shape!=(1,) or weights71["context_delta.up.weight"].shape!=(320,128):
        raise AssertionError("bad fully paid neural gate shape")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    challenge=make_new_blind_pairs(tok)
    frozen_data=[{"id":p["pair_id"],"cohort":p["cohort"],
                 "left":{"question":p["left"]["question"],"answer":p["left"]["answer"]},
                 "right":{"question":p["right"]["question"],"answer":p["right"]["answer"]}} for p in challenge]
    challenge_payload=json.dumps(
       {"opposed_pairs":frozen_data,"open_questions":OPEN_QUESTIONS},
       sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    challenge_sha=sha(challenge_payload)
    (x.output_dir/"S72_NEW_QA_PERMANENTLY_EMBARGOED.json").write_bytes(challenge_payload)
    print(json.dumps({"event":"S72_NEW_ONE_SHOT_CHALLENGE_FROZEN_BEFORE_FIRST_INFERENCE",
          "QA_sha256":challenge_sha,
          "candidate_sha256":sha(s71_bytes),
          "objective_questions":96,"opposed_pairs":48,"open_questions":len(OPEN_QUESTIONS),
          "immutable_FP32_model":MODEL+"@"+MODEL_SHA,
          "never_use_for_training_or_model_selection":True}),flush=True)

    old=instantiate(raw,weights67,128)
    fresh=instantiate(raw,weights67,128)
    mixer=s71.NeuralContextMixer(fresh[3],
                weights67["later.up.weight"]+weights71["context_delta.up.weight"])
    mixer.context_alpha.load_state_dict({
        "weight":weights71["context_gate.weight"],
        "bias":weights71["context_gate.bias"]})
    mixer.eval().requires_grad_(False)
    fresh=(fresh[0],fresh[1],fresh[2],mixer,fresh[4])
    # Fresh ORIGINAL pretrained FP32 135M: no training and never save weights.
    reference=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
        attn_implementation="eager").cpu().eval()
    reference.requires_grad_(False)
    total=sum(p.numel() for p in reference.parameters())
    if not 100_000_000<total<150_000_000 or any(p.dtype!=torch.float32 for p in reference.parameters()):
        raise AssertionError("this is NOT the canonical 135M FP32 reference")
    states={"S67_frozen_36M":old,"S71_context_36M":fresh,
            "ORIGINAL_pretrained_135M_FP32_other_architecture":reference}
    results={name:{} for name in states}
    measured={name:[] for name in states}
    keys=list(states)
    for index,pair in enumerate(challenge):
        for side in ("left","right"):
            row=pair[side]
            key=f'{pair["pair_id"]}_{side}'
            expected=tok.encode(row["answer"],add_special_tokens=False)+[tok.eos_token_id]
            # Fixed alternating execution order, so sequence does not bias speed
            order=keys if (index+(side=="right"))%2==0 else list(reversed(keys))
            for name in order:
                state=states[name]
                if name.startswith("ORIGINAL_"):
                    got,elapsed=run_pretrained_135m_fp32(state,tok,row["question"],s56.MAX_NEW)
                else:
                    got,elapsed=run_small(state,tok,row["question"])
                results[name][key]={
                    "token_ids":got,"full_text":tok.decode(got,skip_special_tokens=True),
                    "expected":row["answer"],"expected_ids":expected,
                    "full_correct_including_EOS":got==expected,
                    "first_color_correct":bool(got) and got[0]==expected[0],
                    "EOS_emitted":bool(got) and got[-1]==tok.eos_token_id,
                    "cohort":pair["cohort"]}
                measured[name].append(elapsed)
        if (index+1)%8==0:
            print(json.dumps({"event":"S72_NEW_QA_INDEPENDENT_PROGRESS",
             "opposed_pairs_checked":index+1,"out_of":48}),flush=True)
    scores={name:answer_summary(rows,challenge,name) for name,rows in results.items()}
    ref=results["S67_frozen_36M"]
    candidate=results["S71_context_36M"]
    wins=[k for k in ref if not ref[k]["full_correct_including_EOS"] and candidate[k]["full_correct_including_EOS"]]
    regress=[k for k in ref if ref[k]["full_correct_including_EOS"] and not candidate[k]["full_correct_including_EOS"]]
    speed=statistics.median(measured["S67_frozen_36M"])/statistics.median(measured["S71_context_36M"])
    open_samples=[]
    for i,q in enumerate(OPEN_QUESTIONS):
        record={"question":q,"sample":i}
        for name,state in states.items():
            if name.startswith("ORIGINAL_"):
                got,elapsed=run_pretrained_135m_fp32(state,tok,q,36)
            else:
                # Same 10-token generative S56 cap, not artificially increase
                got,elapsed=run_small(state,tok,q)
            record[name]={"tokens":got,"actual_text":tok.decode(got,skip_special_tokens=True),
                          "EOS_emitted":bool(got) and got[-1]==tok.eos_token_id}
        open_samples.append(record)
    outcome={
      "event":"S72_REAL_INDEPENDENT_QA_LOCKED_S71_VS_S67_AND_ORIGINAL_135M_FP32",
      "QA_SHA256":challenge_sha,
      "all_models_locked_before_QA":True,
      "candidate_S71_sha256":sha(s71_bytes),
      "physical_S71_all_inclusive_bpp":man71["physical_all_inclusive_bpp"],
      "physical_S67_all_inclusive_bpp":man67["all_inclusive_bpp"],
      "FP32_is_original_pretrained_135M_NOT_same_36M_architecture":True,
      "objective_evaluation":scores,
      "S71_new_right_vs_S67":wins,
      "S71_lost_right_vs_S67":regress,
      "S71_CPU_speed_ratio_vs_S67_on_unseen":speed,
      "CPU_speed_90percent_gate_pass":speed>=.90,
      "new_open_language_outputs_unscored_require_human_review":open_samples,
      "all_model_token_outputs":results,
      "S72_now_PERMANENTLY_EMBARGOED_from_future_training":True,
      "never_run_this_QA_again":True,
      "no_QA_finetuning_no_model_checkpoint_selection":True,
      "not_claiming_FP32_quality_parity":True,
      "no_gpu_groq_jessica_chat_tests_or_LIVE":True
    }
    (x.output_dir/"S72_SEALED_ONE_SHOT_REAL_QA_96_AND_OPEN_FINAL.json").write_text(
        json.dumps(outcome,indent=2,ensure_ascii=False))
    print(json.dumps({"event":"S72_FINAL_SCIENTIFIC_OUTCOME",
      "scores":scores,"new_correct":len(wins),"lost_correct":len(regress),
      "cpu_speed_ratio":speed,"bpp":man71["physical_all_inclusive_bpp"],
      "FP32_pretrained_135M_is_context_only_not_identical_architecture":True,
      "independent_QA_fully_embargoed_for_all_future_steps":True}),flush=True)

if __name__=="__main__":
    main()
