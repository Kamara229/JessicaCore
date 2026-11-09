"""CALIBRI S85 — FIRST AND ONLY independent novel-object QA of FROZEN S82.

Scientific contract: S74 real-GPU first-rank16, S80 frozen native
later-rank16, S82 frozen generic token context pointer are immutable,
TRAIN-only and physically paid <=0.7 bpp at the start. The exact
same-architecture 36M compressed source is used in all models.
S82's 74/256 new TRAIN answers are not independent generalization.
This QA tests out-of-training NOUN TRANSFER (64 swapped opposing pairs,
128 questions), plus four genuinely new open answers.

NO S79, S72, S68, S55, S53, S29 or ANY previous inspected QA prompts,
answers, run data, weights or graded results are loaded/imported, run
again, or used for challenge design/selection. S85 is permanently
embargoed for all future model development, and must be run once only.
All preregistered 128 prompts and 4 open tasks are SHA frozen BEFORE
first independent inference. No gradients, retraining, checkpoint
selection or retries on QA.

All three candidates run on same temporary GPU reconstruction of frozen
CB59 original 36M at full FP32; original paid source/model still
CB61+fully physically paid controller. Compare strict full generated
tokens incl EOS, first token, BOTH opposed tasks and actual raw text.
Don't assert semantic comprehension from lexical/first token scores.
Open output is not graded automatically. Original pretrained 135M FP32
is DIFFERENT architecture and not an identical reference; do not
misrepresent comparison as equality to FP32.

Preflight verifies frozen S74/S80/S82 fully paid sidecars and exactly
identical original S74 tensors, plus real CPU compressed-vs-GPU token
parity on 8 preexisting old TRAIN prompts *before* QA generation.

No changes to LIVE/main/Jessica chat/Render/Supabase/APK.
"""
from __future__ import annotations
import argparse,hashlib,json,re,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from calibri2.s78 import mps_36m_feature_cached_late_rank16_restoration as s78
from calibri2.s80 import real_mps_late_rank16_preserve_all_alpha_search as s80
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s84 import bitwise_certify_s80_physically_paid_232_real_cpu_mps as s84
from calibri2.s56 import paid_first_token_rank2_while_preserving_s54 as s56
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from v186.pairs import tokenize_one

SEED=850085
CASE_NAMES=(
   ("Norven","Amelia"),("Solven","Radila"),("Merin","Kavel"),
   ("Tureli","Deyan"),("Bertan","Olima"),("Revara","Quinel"),
   ("Cavira","Temric"),("Rensel","Alvira"),
)
# New writing, no imported templates, old QA examples, or target names.
CASE_FRAMES=(
   "Cabinet inspection entry {i}: Item code for {a} is {left}, but the different code for {b} is {right}. Which two-word item code is assigned to {a}?",
   "Festival prop record {i}: {b} will carry {right}. The prop issued to {a} is {left}. Answer with only the two words recorded for {a}.",
   "Garden plot checklist {i}: The label on {a}'s box says {left}. The label on {b}'s box instead says {right}. Give {a}'s label exactly.",
   "Film scene inventory {i}: For {b} the two-part designation is {right}; for {a} it is {left}. What are the two words for {a}?",
   "Parcel distribution {i}: Dispatcher {a} marked a parcel {left}; dispatcher {b} marked another {right}. State the two words assigned to {a}.",
   "Geography class worksheet {i}: {b}'s card is marked {right}. A separate card for {a} is marked {left}. Report {a}'s color and object only.",
   "Lost-and-found note {i}: Person {a} reports an item labeled {left}. Another person, {b}, reports {right}. Which exact item label belongs to {a}?",
   "Travel supply log {i}: The two-word tag {right} belongs to {b}, not {a}. The tag for {a} is {left}. Respond with only {a}'s tag.",
)
# Deliberately not S82 TRAIN nouns, S30 TRAIN composition nouns or the
# fixed older ones; tokenizer compatibility validated before QA creation.
NOUN_POOL=(
 "bottle","basket","brush","carpet","cushion","diamond","envelope",
 "feather","fence","helmet","jacket","ladder","marble","needle",
 "notebook","pillow","poster","ribbon","saddle","shovel","suitcase",
 "trophy","wallet","whistle","anchor","badge","brick","bucket",
 "fountain","glove","kettle","magnet","package","printer","shelf",
 "statue","tablet","violin","blanket","button","crown","guitar",
 "ladle","pocket","ruler","satchel","sculpture","compass","dinosaur",
 "lanternfish","spatula","telescope","trumpet","umbrella","zebra",
 "biscuit","cabinet","helmet","hammerhead","purse","rocket",
)
OPEN_TASKS=(
 "Explain briefly what you should do before lending someone your bicycle.",
 "Describe why a household might keep a written inventory of possessions.",
 "In two short sentences, explain how people can share a limited resource fairly.",
 "Explain the difference between a forecast and an observation in everyday language.",
)
assert len(CASE_NAMES)==len(CASE_FRAMES)==8
assert len(OPEN_TASKS)==4

def sha(b):return hashlib.sha256(b).hexdigest()

def independent_novel_object_matrix(tok):
    exclude=set(s82.NOUNS)|{
       x for names in s30.TRAIN_ELIGIBLE.values() for x in names}
    chosen=[]
    for word in NOUN_POOL:
        if word in exclude or word in chosen:continue
        if not all(len(tok.encode(f"{color} {word}",add_special_tokens=False))==2
                   for color in ("red","green","blue","yellow")):
            continue
        chosen.append(word)
        if len(chosen)==16:break
    if len(chosen)!=16:
        raise AssertionError("S85 not enough actually token-compatible fresh nouns")
    assert all(word not in exclude for word in chosen)
    rows=[];serial=85000
    for ci,color in enumerate(("red","green","blue","yellow")):
        four=chosen[ci*4:ci*4+4]
        for word_a,word_b in ((four[0],four[1]),(four[2],four[3])):
            first,second=f"{color} {word_a}",f"{color} {word_b}"
            for index,frame in enumerate(CASE_FRAMES):
                person,other=CASE_NAMES[index]
                p=frame.format(i=serial,a=person,b=other,
                               left=first,right=second)
                q=frame.format(i=serial,a=person,b=other,
                               left=second,right=first)
                a=tokenize_one(tok,p,first);b=tokenize_one(tok,q,second)
                if a["first_expected_id"]!=b["first_expected_id"] or p==q:
                    raise AssertionError("S85 opposed noun matrix invalid")
                rows.append({"id":serial,"color":color,
                             "left":a,"right":b})
                serial+=1
    if (len(rows)!=64 or
        len({p[x]["question"] for p in rows for x in ("left","right")})!=128):
        raise AssertionError("S85 64 pairs/128 independent questions violated")
    return rows,chosen

def make_pointer(w):
    network=s82.GenericContextPointer()
    actual={k.removeprefix("generic_pointer."):v for k,v in w.items()
            if k.startswith("generic_pointer.")}
    if set(actual)!=set(network.state_dict()):
        raise AssertionError("S85 pointer weight schema differs from S82")
    network.load_state_dict(actual,strict=True)
    return network.eval().requires_grad_(False)

def audit_manifest_and_weight_ancestry(a):
    raw=a.S45_frozen_CB59.read_bytes()
    compact=a.S61_frozen_compressed.read_bytes()
    side74=a.S74_frozen_controller.read_bytes()
    side80=a.S80_frozen_controller.read_bytes()
    side82=a.S82_frozen_controller.read_bytes()
    m74=json.loads(a.S74_frozen_manifest.read_text())
    m80=json.loads(a.S80_frozen_manifest.read_text())
    m82=json.loads(a.S82_frozen_manifest.read_text())
    if len(raw)!=SOURCE_BYTES or sha(raw)!=s65.S45_SHA:
        raise AssertionError("S85 original 36M physical source not immutable")
    if (m74["S74_GPU_full_controller_SHA256"]!=sha(side74) or
        m74["S61_unchanged_exact_compressed_source_SHA256"]!=sha(compact) or
        m74["physical_all_inclusive_bytes"]!=
            len(compact)+len(side74)+a.S74_frozen_manifest.stat().st_size or
        float(m74["physical_all_inclusive_bpp"])>.7):
        raise AssertionError("S85 S74 original physical manifest invalid")
    if (m80["original_S74_full_controller_SHA256"]!=sha(side74) or
        m80["new_controller_SHA256"]!=sha(side80) or
        m80["physically_paid_compressed_source_sha256"]!=sha(compact) or
        m80["full_physical_bytes"]!=
            len(compact)+len(side80)+a.S80_frozen_manifest.stat().st_size or
        float(m80["all_inclusive_physical_bpp"])>.7):
        raise AssertionError("S85 S80 paid manifest invalid")
    if (m82["S74_original_frozen_controller_SHA256"]!=sha(side74) or
        m82["S82_full_new_controller_SHA256"]!=sha(side82) or
        m82["original_S61_physically_compressed_source_SHA256"]!=sha(compact) or
        m82["all_inclusive_physical_bytes"]!=
            len(compact)+len(side82)+a.S82_frozen_manifest.stat().st_size or
        float(m82["all_inclusive_bpp"])>.7):
        raise AssertionError("S85 S82 paid manifest invalid")
    w74=load_file(str(a.S74_frozen_controller))
    w80=load_file(str(a.S80_frozen_controller))
    w82=load_file(str(a.S82_frozen_controller))
    if set(w80)-set(w74)!={
       "later_increment.down.weight","later_increment.up.weight",
       "later_increment.gate.weight","later_increment.gate.bias"}:
        raise AssertionError("S85 S80 late neural schema drift")
    if (set(w82)-set(w74))!={
       "generic_pointer.query.weight","generic_pointer.key.weight",
       "generic_pointer.gate.weight","generic_pointer.gate.bias",
       "generic_pointer.log_gain"}:
        raise AssertionError("S85 S82 generic pointer schema drift")
    for ref in (w80,w82):
        if any(k not in ref or not torch.equal(ref[k],v)
               for k,v in w74.items()):
            raise AssertionError("S85 original S74 frozen tensor was mutated")
    return raw,(w74,w80,w82),{
        "S74":sha(side74),"S80":sha(side80),"S82":sha(side82),
        "CB59_source":sha(raw),"CB61_source":sha(compact),
        "all_inclusive_bpp":{
            "S74":m74["physical_all_inclusive_bpp"],
            "S80":m80["all_inclusive_physical_bpp"],
            "S82":m82["all_inclusive_bpp"]}}

@torch.inference_mode()
def real_mps_gen(dense,tok,first,later,head,question,pointer=None,max_new=10):
    prompt=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    prefix=tok.encode(prompt,add_special_tokens=False)
    ids=torch.tensor([prefix],device="mps",dtype=torch.long)
    kept=s82.candidate_positions(tok,prefix) if pointer else []
    token_ids=torch.tensor([prefix[i] for i in kept],device="mps",dtype=torch.long)
    stored=None;past=None;tokens=[]
    for step in range(max_new):
        in_t=ids if step==0 else torch.tensor(
            [[tokens[-1]]],device="mps",dtype=torch.long)
        out=dense(input_ids=in_t,past_key_values=past,use_cache=True,
                  output_hidden_states=True)
        past=out.past_key_values
        if past is None or out.hidden_states[-1].device.type!="mps":
            raise AssertionError("S85 actual GPU KV-cache execution required")
        h=out.hidden_states[-1][0]
        if step==0 and pointer and kept:
            stored=h[torch.tensor(kept,device="mps")].detach()
        current=h[-1]
        scores=out.logits[0,-1]+torch.nn.functional.linear(
            first(current) if step==0 else later(current),head)
        if step==1 and pointer and stored is not None:
            if int(scores.argmax())!=tok.eos_token_id:
                extra=pointer.full_vocab_bonus(current,stored,token_ids)
                scores=scores+torch.zeros_like(scores).index_add(
                    0,token_ids,extra)
        nxt=int(scores.argmax());tokens.append(nxt)
        del out
        if nxt==tok.eos_token_id:break
    torch.mps.synchronize()
    return tokens

def weak_phrase(text,goal,opposite):
    normalize=lambda x:" ".join(re.findall(r"[a-zA-Z]+",x.lower()))
    x=f" {normalize(text)} "
    return f" {normalize(goal)} " in x and f" {normalize(opposite)} " not in x

def score(rows,casepairs):
    keys=list(rows)
    first=sum(rows[k]["first_expected_token_correct"] for k in keys)
    full=sum(rows[k]["strict_full_answer_with_EOS"] for k in keys)
    both=sum(
        rows[f'{p["id"]}_left']["strict_full_answer_with_EOS"] and
        rows[f'{p["id"]}_right']["strict_full_answer_with_EOS"]
        for p in casepairs)
    phrase=sum(rows[k]["weak_unambiguous_target_phrase_in_text"] for k in keys)
    return {"first_correct":first,"strict_full_with_EOS":full,
            "both_opposing_full_correct":both,
            "weak_lexical_phrase_not_human_semantic":phrase,
            "out_of_questions":len(keys),"out_of_pairs":len(casepairs)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_frozen_CB59",type=Path)
    p.add_argument("S61_frozen_compressed",type=Path)
    p.add_argument("S74_frozen_controller",type=Path)
    p.add_argument("S74_frozen_manifest",type=Path)
    p.add_argument("S80_frozen_controller",type=Path)
    p.add_argument("S80_frozen_manifest",type=Path)
    p.add_argument("S82_frozen_controller",type=Path)
    p.add_argument("S82_frozen_manifest",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S85 no CPU fallback: Apple Silicon GPU mandatory")
    torch.manual_seed(SEED);torch.set_num_threads(2)
    raw,(w74,w80,w82),frozen=audit_manifest_and_weight_ancestry(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    dense=s30.dense_from_paid(CB59(raw)).to("mps").eval().requires_grad_(False)
    if sum(z.numel() for z in dense.parameters())!=SOURCE_PARAMS:
        raise AssertionError("S85 original 36M same-architecture GPU source invalid")
    head=dense.lm_head.weight.detach()
    f74,l74=s78.controllers(w74)
    f74=f74.to("mps").eval().requires_grad_(False)
    l74=l74.to("mps").eval().requires_grad_(False)
    l80=s84.frozen_late_s80(s78.controllers(w74)[1])
    l80=l80.to("mps").eval().requires_grad_(False)
    pointer=make_pointer(w82).to("mps").eval().requires_grad_(False)
    systems={
      "S74_pretrained_first_neural_36M":(f74,l74,None),
      "S80_GPU_late_rank16_36M":(f74,l80,None),
      "S82_GENERIC_context_neural_pointer_36M":(f74,l74,pointer),
    }
    # Native CPU/GPU identity preflight uses old TRAIN only, not any QA.
    model,source=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    head_cpu=model.lm_head.source().detach().cpu().contiguous()
    first_cpu,later_cpu=s78.controllers(w74)
    late80_cpu=s84.frozen_late_s80(s78.controllers(w74)[1])
    ptr_cpu=make_pointer(w82)
    seen=s30.make_training_pairs(tok)[:4]
    matched=0
    for case in seen:
        for side in ("left","right"):
            q=case[side]["question"]
            golds={
              "S74_pretrained_first_neural_36M":
                  s56.greedy_first_active(model,source,first_cpu,later_cpu,head_cpu,tok,q),
              "S80_GPU_late_rank16_36M":
                  s56.greedy_first_active(model,source,first_cpu,late80_cpu,head_cpu,tok,q),
              "S82_GENERIC_context_neural_pointer_36M":
                  s82.full_cpu_generate(model,source,first_cpu,later_cpu,head_cpu,tok,q,ptr_cpu),
            }
            for name,(first,later,pointer_) in systems.items():
                on_gpu=real_mps_gen(dense,tok,first,later,head,q,pointer_,10)
                if on_gpu!=golds[name]:
                    raise AssertionError("S85 REFUSE independent QA if native GPU parity fails old TRAIN "+name)
                matched+=1
    if matched!=24:raise AssertionError("S85 incomplete 8x3 CPU/GPU old TRAIN preflight")
    print(json.dumps({"event":"S85_PREREGISTERED_MODEL_AND_REAL_GPU_PREFLIGHT",
        "all_3_models_SHA_frozen":frozen,
        "old_TRAIN_full_native_compressed_CPU_to_real_GPU_equal_count":matched,
        "sealed_QA_not_generated_or_inferred_yet":True}),flush=True)

    # Lock fresh challenge only AFTER immutable controller+CPU/GPU preflight,
    # and BEFORE any unseen model inference. Never reuse this instrument.
    casepairs,nouns=independent_novel_object_matrix(tok)
    challenge={"kind":"FIRST_AND_ONLY_S85_BLIND_128_NOVEL_NOUN_QA",
       "reference_frozen_model_SHA256":frozen,
       "novel_nouns":nouns,
       "opposing_pairs":[{
          "pair_id":p["id"],
          "left":{"question":p["left"]["question"],"gold":p["left"]["answer"]},
          "right":{"question":p["right"]["question"],"gold":p["right"]["answer"]}}
          for p in casepairs],
       "four_open_prompts":OPEN_TASKS}
    payload=json.dumps(challenge,ensure_ascii=False,sort_keys=True,
                       separators=(",",":")).encode()
    (a.output_dir/"S85_NEW_QA_PERMANENTLY_EMBARGOED.json").write_bytes(payload)
    qa_sha=sha(payload)
    print(json.dumps({"event":"S85_ONE_TIME_NEW_QA_FROZEN_BEFORE_FIRST_INFERENCE",
        "QA_SHA256":qa_sha,"questions":128,"opposing_pairs":64,
        "new_nouns":len(nouns),"open_questions":len(OPEN_TASKS),
        "NO_previous_QA_read_or_reused":True}),flush=True)
    outputs={name:{} for name in systems}
    for ci,case in enumerate(casepairs):
        for side in ("left","right"):
            item=case[side]
            key=f'{case["id"]}_{side}'
            gold=tok.encode(item["answer"],add_special_tokens=False)+[tok.eos_token_id]
            alternate=case["right" if side=="left" else "left"]["answer"]
            for name,(first,later,ptr) in systems.items():
                got=real_mps_gen(dense,tok,first,later,head,item["question"],ptr,10)
                response=tok.decode(got,skip_special_tokens=True)
                outputs[name][key]={
                    "actual_full_token_ids":got,
                    "actual_full_text":response,
                    "gold_full_ids_including_EOS":gold,
                    "gold_text":item["answer"],
                    "opposite_gold_text":alternate,
                    "first_expected_token_correct":bool(got) and got[0]==gold[0],
                    "strict_full_answer_with_EOS":got==gold,
                    "weak_unambiguous_target_phrase_in_text":
                        weak_phrase(response,item["answer"],alternate),
                    "EOS_emitted":bool(got) and got[-1]==tok.eos_token_id,
                }
        if (ci+1)%8==0:
            print(json.dumps({"event":"S85_ONE_SHOT_NOVEL_NOUN_GENERALIZATION_PROGRESS",
                "opposed_pairs_inferred":ci+1,"of":64}),flush=True)
    results={name:score(rows,casepairs) for name,rows in outputs.items()}
    baseline=outputs["S74_pretrained_first_neural_36M"]
    changes={}
    for name in ("S80_GPU_late_rank16_36M","S82_GENERIC_context_neural_pointer_36M"):
        experiment=outputs[name]
        changes[name]={
            "newly_correct_strict_full_vs_S74":[
                key for key in baseline if not baseline[key]["strict_full_answer_with_EOS"]
                and experiment[key]["strict_full_answer_with_EOS"]],
            "lost_previously_correct_strict_full_vs_S74":[
                key for key in baseline if baseline[key]["strict_full_answer_with_EOS"]
                and not experiment[key]["strict_full_answer_with_EOS"]],
        }
    open_outputs=[]
    for text_prompt in OPEN_TASKS:
        actual={"new_unseen_open_prompt":text_prompt}
        for name,(first,later,ptr) in systems.items():
            got=real_mps_gen(dense,tok,first,later,head,text_prompt,ptr,40)
            actual[name]={"actual_full_token_ids":got,
                         "actual_plain_text":tok.decode(got,skip_special_tokens=True),
                         "EOS_emitted":bool(got) and got[-1]==tok.eos_token_id}
        open_outputs.append(actual)
    final={
      "event":"S85_FINAL_FIRST_AND_ONLY_128_NOVEL_NOUN_QA_S74_S80_S82",
      "scientific_qualifier":"NONE of these QA prompts or labels may be used in any future training, selection or evaluation.",
      "locked_QA_SHA256":qa_sha,
      "physical_bpp_of_all_frozen_candidates":frozen["all_inclusive_bpp"],
      "all_frozen_models_share_exact_36M_source":True,
      "original_36M_gpu_FP32_expansion_temporary_not_saved":True,
      "not_F32_baseline_equal_quality":True,
      "all_existing_QA_embargoed_S79_S72_S68_S55_S53_S29":True,
      "all_new_S85_QA_NOW_PERMANENTLY_EMBARGOED":True,
      "new_object_generalization_STRICT_full_first_both":results,
      "actual_strict_answer_gains_losses_vs_S74":changes,
      "all_actual_new_QA_generated_token_ids_and_text":outputs,
      "new_open_freeform_actual_generations_UNSCORED_require_human_review":open_outputs,
      "no_model_training_checkpoint_selection_or_QA_retry":True,
      "no_LIVE_Jessica_main_Render_Supabase_APK_changes":True,
    }
    (a.output_dir/"S85_FROZEN_ONE_SHOT_REAL_UNSEEN_QA_S74_S80_S82_FINAL.json").write_text(
       json.dumps(final,ensure_ascii=False,indent=2))
    print(json.dumps({"event":"S85_FINAL_SINGLE_USE_QA_SCORE",
        "sealed_SHA256":qa_sha,
        "REAL_FULL_WORD_AND_OPPOSED_PAIR_RESULTS":results,
        "S80_gain_lost_count":[len(changes["S80_GPU_late_rank16_36M"][k]) for k in (
          "newly_correct_strict_full_vs_S74","lost_previously_correct_strict_full_vs_S74")],
        "S82_gain_lost_count":[len(changes["S82_GENERIC_context_neural_pointer_36M"][k]) for k in (
          "newly_correct_strict_full_vs_S74","lost_previously_correct_strict_full_vs_S74")],
        "all_QA_permanently_embargoed":True}),flush=True)

if __name__=="__main__":main()
