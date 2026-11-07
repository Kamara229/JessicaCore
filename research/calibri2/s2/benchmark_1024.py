"""CALIBRI2 S2: precommitted 960 unseen objective + 64 open-ended prompts.

Large independent diagnostic for S1 paid 22M model and later S2 checkpoints.
Training NEVER reads this suite. No FP32 comparison, teacher, QA benchmark
checkpoint selection, LIVE Jessica or Render/Supabase changes.
960 distinct deterministic synthetic held-out questions across eight domains,
plus 64 open-ended real generations. Gold is determined by simple executable
logic, with strict whole-answer AND EOS scoring. Not a claim of general LLM
competence or model ceiling. Physically packed CB59 input is mandatory.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v174.run import save
from v176.operators import clear
from v186.pairs import build_pairs
from calibri2.s0.bootstrap import CANDIDATES,build_config
from calibri2.s0.train import create_small_paid_model

SEED=2021
CASES_PER_DOMAIN=120
MAX_TOKENS_QA=12
MAX_TOKENS_OPEN=20
BPP_LIMIT=.700
EXPECTED_PARAMS=22368320
EXPECTED_BYTES=1932195
OPEN_TOPICS=(
 "Explain why the sky can look blue.", "Describe how rain forms.",
 "What happens to ice when it melts?", "Explain why plants need sunlight.",
 "What is the purpose of a library?", "Describe a friendly neighborhood.",
 "How would you prepare a cup of tea?", "Explain why wheels are round.",
 "What makes a good explanation?", "Describe a rainy street.",
 "How can a student practice arithmetic?", "Explain why time is measured.",
 "What is an example of a simple machine?", "Describe a day at the beach.",
 "Explain what a map is useful for.", "Describe the role of a teacher.",
 "What does a bridge connect?", "Explain what a computer does.",
 "Describe a busy train station.", "What is a scientific experiment?",
 "How can people save electricity?", "What is the difference between day and night?",
 "Explain why clean water matters.", "Describe what happens in a garden.",
 "What does it mean to compare two numbers?", "Describe a calm lake.",
 "How can people learn new words?", "Explain how a bicycle moves.",
 "What does it mean to solve a problem?", "Describe a mountain trail.",
 "How does a clock help people?", "Write a short story about finding a key.",
)
assert len(OPEN_TOPICS)==32

def frozen_suite():
    cases=[]
    def add(domain,i,question,gold):
        cases.append({"case_id":f"{domain}_{i:03d}","domain":domain,
                      "question":question,"answer":str(gold)})
    for i in range(CASES_PER_DOMAIN):
        a=1009+11*i
        b=a+2+i%17
        yes=(i%2==0)
        add("greater_than",i,
            f"Is {b if yes else a} greater than {a if yes else b}? Reply with only Yes or No.",
            "Yes" if yes else "No")
        yes=(i%2==1)
        add("less_than",i,
            f"Is {a if yes else b} less than {b if yes else a}? Reply with only Yes or No.",
            "Yes" if yes else "No")
        person=f"LumoProbe{i:03d}"
        other=f"KivaProbe{i:03d}"
        yes=(i%2==0)
        add("book_binding_yesno",i,
            f"In a fictional story, {person} carries a {'silver' if yes else 'gold'} book and {other} carries a {'gold' if yes else 'silver'} book. Does {person} carry a silver book? Reply with only Yes or No.",
            "Yes" if yes else "No")
        red=(i%2==0)
        add("folder_color_binding",i,
            f"In a fictional story, {person} has a {'red' if red else 'blue'} folder and {other} has a {'blue' if red else 'red'} folder. What color is {person}'s folder? Reply with one word.",
            "red" if red else "blue")
        d=1200+13*i
        e=2+i%9 if i%2==0 else 204+i%9
        add("addition",i,f"What is {d} plus {e}? Reply with only the number.",str(d+e))
        v=1600+15*i
        w=4+i%17
        add("subtraction",i,
            f"What is {v} minus {w}? Reply with only the number.",str(v-w))
        yes=(i%2==0)
        add("transitive_temporal",i,
            f"In a fictional story, {person} arrives before {other} and {other} arrives before SoriProbe{i:03d}. Does {'%s' % person if yes else f'SoriProbe{i:03d}'} arrive before {'SoriProbe%03d' % i if yes else person}? Reply with only Yes or No.",
            "Yes" if yes else "No")
        label="quiet river" if i%2==0 else "silver tower"
        add("two_word_copy",i,
            f"In a fictional story, {person} remembers the code words '{label}'. What are the code words? Reply with exactly two words.",
            label)
    domains=Counter(x["domain"] for x in cases)
    if len(cases)!=960 or len(domains)!=8 or set(domains.values())!={CASES_PER_DOMAIN}:
        raise AssertionError("Precommitted 960 QA cases must span 8 equal domains")
    if len({x["question"] for x in cases})!=len(cases):
        raise AssertionError("Held-out question duplication")
    opens=[]
    for i,topic in enumerate(OPEN_TOPICS):
        opens.append({"case_id":f"freeform_{2*i:03d}","question":topic})
        opens.append({"case_id":f"freeform_{2*i+1:03d}",
            "question":topic+" Use simple English and explain your reasoning."})
    if len(opens)!=64:
        raise AssertionError("Expected 64 fixed open-ended prompts")
    return cases,opens

def chat_ids(tok,question):
    message=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    return tok.encode(message,add_special_tokens=False)

@torch.inference_mode()
def generate(model,sources,prompt,tok,cap):
    context=torch.tensor(chat_ids(tok,prompt),dtype=torch.long)[None]
    if not context.numel() or context.numel()>=256:
        raise AssertionError("Malformed prompt too long for bounded test")
    produced=[]
    eos=tok.eos_token_id
    for _ in range(cap):
        clear(sources)
        logits=model(input_ids=context,use_cache=False).logits[0,-1]
        if not torch.isfinite(logits).all():
            raise AssertionError("Nonfinite full-vocabulary output")
        nxt=int(logits.argmax())
        produced.append(nxt)
        if nxt==eos:
            break
        context=torch.cat((context,context.new_tensor([[nxt]])),dim=1)
    clear(sources)
    return produced

def main():
    p=argparse.ArgumentParser()
    p.add_argument("physical_archive",type=Path)
    p.add_argument("output_dir",type=Path)
    p.add_argument("--label",required=True)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=a.physical_archive.read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    cb=CB59(raw)
    n=cb.meta["original_unique_parameter_count"]
    if n!=EXPECTED_PARAMS or len(raw)!=EXPECTED_BYTES or 8*len(raw)/n>BPP_LIMIT:
        raise AssertionError("No growth/no floating sidecar: require intact 22M <=.700bpp archive")
    cfg=build_config(next(s for s in CANDIDATES if s[0]=="small_320x6"))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    training=build_pairs(tok,"TRAIN")
    seen_train={pair[side]["question"] for pair in training for side in ("left","right")}
    cases,opens=frozen_suite()
    if any(x["question"] in seen_train for x in cases):
        raise AssertionError("A held-out question occurred in TRAIN")
    if any(x["question"] in seen_train for x in opens):
        raise AssertionError("A freeform prompt occurred in TRAIN")
    suite_hash=hashlib.sha256(json.dumps([cases,opens],ensure_ascii=False,
                                      sort_keys=True).encode()).hexdigest()
    model,sources=create_small_paid_model(cb,cfg)
    model.eval()
    correct=Counter()
    total=Counter()
    first=Counter()
    prompt_first_tokens=Counter()
    full=[]
    started=time.perf_counter()
    for k,x in enumerate(cases):
        exp=tok.encode(x["answer"],add_special_tokens=False)+[tok.eos_token_id]
        if not 2<=len(exp)<=MAX_TOKENS_QA:
            raise AssertionError("Unscorable held-out gold with EOS")
        predicted=generate(model,sources,x["question"],tok,MAX_TOKENS_QA)
        exact=(predicted==exp)
        key=x["domain"]
        total[key]+=1
        correct[key]+=int(exact)
        first[key]+=int(bool(predicted) and predicted[0]==exp[0])
        if predicted:prompt_first_tokens[predicted[0]]+=1
        full.append({
          "id":x["case_id"],"domain":key,
          "question":x["question"],"gold_text":x["answer"],
          "gold_tokens_with_EOS":exp,
          "actual_token_ids":predicted,
          "actual_text":tok.decode(predicted,skip_special_tokens=True),
          "strict_full_answer_and_EOS_correct":exact,
          "first_token_correct":bool(predicted) and predicted[0]==exp[0],
          "stopped_at_EOS":bool(predicted) and predicted[-1]==tok.eos_token_id,
          "adjacent_repeat_count":sum(
              v==w for v,w in zip(predicted[:-1],predicted[1:])),
        })
        if (k+1)%120==0:
            print(json.dumps({"stage":"CALIBRI2_S2_FROZEN_LARGE_TEST_DOMAIN",
                  "label":a.label,"completed":k+1,
                  "domain":key,
                  "overall_strict_full_correct":sum(correct.values())}),flush=True)
    creative=[]
    for x in opens:
        actual=generate(model,sources,x["question"],tok,MAX_TOKENS_OPEN)
        creative.append({"id":x["case_id"],"question":x["question"],
            "generated_token_ids":actual,
            "actual_text":tok.decode(actual,skip_special_tokens=True),
            "stopped_at_EOS":bool(actual) and actual[-1]==tok.eos_token_id,
            "adjacent_repeat_count":sum(
               p==q for p,q in zip(actual[:-1],actual[1:]))})
    elapsed=time.perf_counter()-started
    report={"suite_sha256":suite_hash,
       "model_label":a.label,"model_sha256":sha,
       "unique_original_parameters":n,"physical_bytes":len(raw),
       "physical_bits_per_original_parameter":8*len(raw)/n,
       "number_heldout_objective_prompts":len(cases),
       "number_heldout_freeform_prompts":len(opens),
       "all_prompts_are_unique_not_in_48_TRAIN_questions":True,
       "QA_or_VALID_used_during_training":False,
       "FP32_teacher_or_comparison":False,
       "strict_full_answer_and_EOS_correct":sum(correct.values()),
       "first_token_correct":sum(first.values()),
       "denominator_objective":len(cases),
       "by_domain":[{"domain":k,"full_EOS_correct":correct[k],
           "first_correct":first[k],"total":total[k]}
           for k in sorted(total)],
       "first_token_diversity":len(prompt_first_tokens),
       "most_common_first_generated_tokens":[
           {"token_id":i,"text":tok.decode([i]),"count":n}
           for i,n in prompt_first_tokens.most_common(20)],
       "all_960_actual_answers":full,
       "freeform_64_actual_answers_unscored":creative,
       "freeform_no_auto_claim_of_coherence":True,
       "runtime_python_proxy_seconds_NOT_native_C_speed":round(elapsed,3),
       "this_test_alone_cannot_establish_model_capacity_ceiling":True,
       "not_used_for_model_selection":True}
    save(a.output_dir/"ACTUAL_CALIBRI2_1024_HELDOUT_LARGE_TEST.json",report)
    print(json.dumps({"stage":"CALIBRI2_S2_LARGE_TEST_COMPLETED",
      "label":a.label,"test_sha":suite_hash,
      "strict_full_and_EOS":sum(correct.values()),
      "first_correct":sum(first.values()),
      "objective_total":len(cases),
      "freeform_total":len(creative),
      "diversity_first":len(prompt_first_tokens),
      "bpp":8*len(raw)/n,"by_domain":report["by_domain"]},
      ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
