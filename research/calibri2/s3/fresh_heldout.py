"""CALIBRI2 S3 precommitted new heldout generalization: 192 objective + 24 open.
Evaluate both immutable physical S2 and posttraining S3 checkpoint, but NEVER
use these questions for weight updates, early stopping or checkpoint selection.
No FP32 comparison; no LIVE changes; S2 960 is an exploratory reused diagnostic.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v186.pairs import build_pairs
from calibri2.s0.bootstrap import CANDIDATES,build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.diverse_task_replay import make_new_pairs
from v174.run import save

PARAMS=22368320
BYTES=1932195
S2_SHA="d0de536fa91146bf97fad77941aff51a13f28b476b5ec318b5d32489ab3cb575"
COPY_LABELS=("amber pearl","hidden ladder","teal rabbit","orange garden",
             "silver comet","polished shell","bright tulip","narrow brook")
TOPICS=(
 "Explain in a sentence why a bird needs wings.",
 "What does a compass help people do?",
 "What happens when the sun warms snow?",
 "Why do we take care of trees?",
 "What is a short explanation of why rain falls?",
 "Write one helpful sentence about keeping safe in rain.",
 "Explain what makes ice turn into water.",
 "What is a bridge useful for?",
 "How can someone find a path in a forest?",
 "Why is it helpful to read books?",
 "How do wheels help a bicycle move?",
 "What does a train carry?",
)

def suite():
    objective=[]
    def put(domain,i,q,a):
        objective.append({"id":f"{domain}_{i:03d}","domain":domain,
                          "question":q,"gold":a})
    for i in range(64):
        owner=f"JariEval{i:03d}";other=f"KoriEval{i:03d}"
        red=i%2==0
        c="red" if red else "blue"
        d="blue" if red else "red"
        if i%2==0:
            q=(f"During the school game, {owner} is holding a {c} folder, "
               f"and {other} is holding a {d} folder. "
               f"What color is the folder held by {owner}? Answer in one word.")
        else:
            q=(f"The folder next to {owner} is {c}; the folder next to "
               f"{other} is {d}. Which color folder is next to {owner}? One word.")
        put("color_binding",i,q,c)
        a=2207+i*43
        b=a+13+i%13
        positive=i%2==0
        if i%4<2:
            lhs,rhs=(b,a) if positive else (a,b)
            q=f"Is the number {lhs} greater than the number {rhs}? Say Yes or No."
        else:
            lhs,rhs=(a,b) if positive else (b,a)
            q=f"Is the number {lhs} smaller than the number {rhs}? Say Yes or No."
        put("numeric_relation",i,q,"Yes" if positive else "No")
        label=COPY_LABELS[i%len(COPY_LABELS)]
        q=(f"A note addressed to {owner} contains the two-word password "
           f"'{label}'. Write that password using exactly two words.")
        put("two_word_recall",i,q,label)
    creative=[{"id":f"open_{i:02d}","question":text+(
        " Reply with one short English sentence." if i%2 else "")}
        for i,text in enumerate(TOPICS*2)]
    if len(objective)!=192 or len(creative)!=24 or len(set(x["question"] for x in objective))!=192:
        raise AssertionError("S3 independent 192+24 suite malformed")
    return objective,creative

def prompt_tokens(tok,question):
    txt=tok.apply_chat_template([{"role":"user","content":question}],
                                  tokenize=False,add_generation_prompt=True)
    return tok.encode(txt,add_special_tokens=False)

@torch.inference_mode()
def generate(model,sources,tok,question,max_tokens):
    prefix=prompt_tokens(tok,question)
    if not 1<=len(prefix)<160:raise AssertionError("Malformed holdout prompt")
    context=torch.tensor(prefix,dtype=torch.long)[None]
    tokens=[]
    for _ in range(max_tokens):
        clear(sources)
        logits=model(input_ids=context,use_cache=False).logits[0,-1]
        if not torch.isfinite(logits).all():raise AssertionError("Nonfinite logits")
        token=int(logits.argmax())
        tokens.append(token)
        if token==tok.eos_token_id:break
        context=torch.cat((context,context.new_tensor([[token]])),dim=1)
    clear(sources)
    return tokens

def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--label",required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=args.checkpoint.read_bytes()
    cb=CB59(raw)
    if len(raw)!=BYTES or cb.meta["original_unique_parameter_count"]!=PARAMS or 8*len(raw)/PARAMS>.7:
        raise AssertionError("Require 22M physically paid CB59 <=0.7bpp")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    originals=build_pairs(tok,"TRAIN")
    added=make_new_pairs(tok)
    excluded={x[s]["question"] for x in originals+added for s in ("left","right")}
    cases,free=suite()
    if any(x["question"] in excluded for x in cases+free):
        raise AssertionError("HELDOUT sample contaminated by TRAIN question")
    suite_sha=hashlib.sha256(json.dumps([cases,free],sort_keys=True).encode()).hexdigest()
    cfg=build_config(next(x for x in CANDIDATES if x[0]=="small_320x6"))
    model,sources=create_small_paid_model(cb,cfg)
    model.eval()
    cnt=Counter();right=Counter();first=Counter()
    rows=[]
    for x in cases:
        expected=tok.encode(x["gold"],add_special_tokens=False)+[tok.eos_token_id]
        if not 2<=len(expected)<=12:raise AssertionError("Unscoreable exact gold")
        observed=generate(model,sources,tok,x["question"],12)
        ok=observed==expected
        cnt[x["domain"]]+=1;right[x["domain"]]+=int(ok)
        first[x["domain"]]+=int(bool(observed) and observed[0]==expected[0])
        rows.append({**x,"generated":tok.decode(observed,skip_special_tokens=True),
                     "token_ids":observed,"gold_ids":expected,
                     "full_plus_EOS":ok,"first_token_correct":bool(observed) and observed[0]==expected[0]})
    creative=[]
    for x in free:
        obs=generate(model,sources,tok,x["question"],20)
        creative.append({**x,"actual_answer":tok.decode(obs,skip_special_tokens=True),
                         "tokens":obs,"stopped_at_EOS":bool(obs and obs[-1]==tok.eos_token_id)})
    report={"label":args.label,"source_sha":hashlib.sha256(raw).hexdigest(),
        "suite_sha":suite_sha,"physical_bpp":8*len(raw)/PARAMS,
        "objective_total":len(cases),"strict_full_plus_EOS":sum(right.values()),
        "by_domain":[{"domain":k,"correct_full_EOS":right[k],
                    "first_correct":first[k],"total":cnt[k]} for k in sorted(cnt)],
        "all_192_actual_heldout_answers":rows,
        "all_24_open_generated_answers_unscored":creative,
        "not_used_for_training_or_model_selection":True,
        "FP32_teacher_comparison":False,
        "cannot_infer_language_capability_from_YesNo_chance":True}
    save(args.output/"ACTUAL_CALIBRI2_S3_FRESH_HELDOUT.json",report)
    print(json.dumps({"event":"CALIBRI2_S3_FRESH_HELDOUT_COMPLETE",
        "label":args.label,"suite_sha":suite_sha,
        "full_EOS":sum(right.values()),"total":len(cases),
        "by_domain":report["by_domain"]}),flush=True)

if __name__=="__main__":
    main()
