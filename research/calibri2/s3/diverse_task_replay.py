"""CALIBRI2 S3: precommitted diverse task learning from real S2 22M CB59.
TRAIN only, 0.691bpp, no FP32 teacher, no LIVE. Never use S2 QA or
fresh S3 heldout prompts for optimization or checkpoint selection.
One fixed 2560-step outcome, not a hyperparameter search.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.optim import propose_codes,apply_codes
from v185.codec import replace_discrete
from v186.pairs import build_pairs,tokenize_one
from v186.fit import paired_loss
from v200.rollout import probe_greedy_train
from calibri2.s0.bootstrap import CANDIDATES,build_config
from calibri2.s0.train import create_small_paid_model
from v174.run import save

START_SHA="d0de536fa91146bf97fad77941aff51a13f28b476b5ec318b5d32489ab3cb575"
PARAMS=22368320
BYTES=1932195
STEPS=2560
LR=.003
SEED=2033

# Distinct word pairs absent from both frozen S2 benchmark answer classes.
SENTENCES=(
 ("Why does a kite fly?","Moving air lifts kites.","Why does ice melt?","Warm air melts ice."),
 ("Why do people wear coats?","Coats keep bodies warm.","Why do boats float?","Boats displace water."),
 ("What helps a flower grow?","Sunlight helps flowers grow.","What helps birds fly?","Wings help birds fly."),
 ("Why do we wash our hands?","Washing removes harmful germs.","Why do we read books?","Reading helps people learn."),
 ("What happens when rain falls?","Rain makes the ground wet.","What happens when a bell rings?","The bell makes a sound."),
 ("Why do we use maps?","Maps show useful locations.","Why do we use clocks?","Clocks help us tell time."),
 ("How do fish breathe?","Fish take oxygen from water.","How do cars move?","Engines move car wheels."),
 ("What is a bridge for?","Bridges connect two places.","What is a lamp for?","Lamps give us light."),
)
WORDS_LEFT=("gentle","frosted","little","purple","golden","silvered","wooden","winter","yellow","distant","open","simple","brave","silent","smooth","fresh")
WORDS_RIGHT=("shining","quietly","bright","clear","light","green","tiny","strong","soft","hidden","plain","warm","small","cool","early","wide")
NOUNS_LEFT=("lantern","garden","island","bridge","meadow","window","pillow","forest")
NOUNS_RIGHT=("sailor","mountain","ribbon","castle","valley","engine","basket","tunnel")

def make_new_pairs(tok):
    raw=[]
    def add(domain,a,b):
        if a[1]==b[1] or a[0]==b[0]:
            raise AssertionError("Contrasting examples must have different labels and questions")
        raw.append((domain,a,b))
    for i in range(24):
        owner=f"MiraTraining{i:03d}";other=f"OrenTraining{i:03d}"
        if i%3==0:
            t="In a made-up scene, {owner} keeps a {color} folder, while {other} keeps a {other_color} folder. What is {owner}'s folder color? One word only."
        elif i%3==1:
            t="The {color} folder is owned by {owner}; the {other_color} folder belongs to {other}. Which folder color belongs to {owner}? Give one word."
        else:
            t="A {color} folder is with {owner}, and a {other_color} folder is with {other}. Name the color of {owner}'s folder in one word."
        def question(color,other_color):
            return t.format(owner=owner,other=other,color=color,other_color=other_color)
        add("s3_folder_training",(question("red","blue"),"red"),
            (question("blue","red"),"blue"))
    for i in range(16):
        a=8101+i*37;b=a+3+i%17
        if i%2==0:
            qa=f"Is {b} larger than {a}? Answer Yes or No."
            qb=f"Is {a} larger than {b}? Answer Yes or No."
        else:
            qa=f"Does {b} exceed {a}? Answer Yes or No."
            qb=f"Does {a} exceed {b}? Answer Yes or No."
        add("s3_compare_training",(qa,"Yes"),(qb,"No"))
    for i in range(16):
        left=WORDS_LEFT[i]+" "+NOUNS_LEFT[i%len(NOUNS_LEFT)]
        right=WORDS_RIGHT[i]+" "+NOUNS_RIGHT[i%len(NOUNS_RIGHT)]
        add("s3_two_word_training",
            (f"The secret phrase in card TRAIN{i:03d} is '{left}'. Repeat exactly the two words.",left),
            (f"The secret phrase in card TRAIN{i:03d} is '{right}'. Repeat exactly the two words.",right))
    for i,(q1,a1,q2,a2) in enumerate(SENTENCES):
        add("s3_sentence_training",
            (q1+" Reply with one short sentence.",a1),
            (q2+" Reply with one short sentence.",a2))
    if len(raw)!=64:raise AssertionError("64 predeclared new contrasting TRAIN pairs required")
    result=[]
    for index,(domain,first,second) in enumerate(raw,start=24):
        l=tokenize_one(tok,*first);r=tokenize_one(tok,*second)
        if l["first_expected_id"]==r["first_expected_id"]:
            raise AssertionError(f"TRAIN pair {index} lacks contrasting first token")
        result.append({"split":"TRAIN","pair_id":index,
                       "domain":domain,"left":l,"right":r})
    return result

def score(model,sources,pairs,label,step):
    result=probe_greedy_train(model,sources,pairs,max_new_tokens=12)
    correct=0;first=0;dom=defaultdict(lambda:{"full":0,"total":0})
    rows=[]
    for p in pairs:
        for side in ("left","right"):
            x=result[p["pair_id"]]["sides"][side]
            ok=x["exact_with_EOS"]
            correct+=int(ok)
            first+=int(x["generated_tokens"][0]==x["gold_tokens_with_EOS"][0])
            d=dom[p["domain"]];d["total"]+=1;d["full"]+=int(ok)
            rows.append({"id":p["pair_id"],"domain":p["domain"],"side":side,
                         "question":p[side]["question"],"gold":p[side]["answer"],
                         "generated_ids":x["generated_tokens"],"gold_ids":x["gold_tokens_with_EOS"],
                         "full_with_EOS":ok})
    summary={"stage":label,"step":step,"correct_full_EOS":correct,
             "first_correct":first,"total":len(rows),
             "by_domain":dict(dom),"actual_answers":rows}
    print(json.dumps({"event":"CALIBRI2_S3_FIXED_TRAIN_MILESTONE",
        "stage":label,"step":step,"correct_full_EOS":correct,
        "total":len(rows),"by_domain":dict(dom)}),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("initial_s2_paid",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw=args.initial_s2_paid.read_bytes()
    if len(raw)!=BYTES or hashlib.sha256(raw).hexdigest()!=START_SHA:
        raise AssertionError("Require exact physically paid S2 checkpoint")
    cb=CB59(raw)
    if cb.meta["original_unique_parameter_count"]!=PARAMS:raise AssertionError("22M architecture mismatch")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    historic=build_pairs(tok,"TRAIN")
    fresh=make_new_pairs(tok)
    assert len(historic)==24 and len(fresh)==64
    frozen_questions={p[s]["question"] for p in historic for s in ("left","right")}
    novel_questions={p[s]["question"] for p in fresh for s in ("left","right")}
    if len(novel_questions)!=128 or frozen_questions.intersection(novel_questions):
        raise AssertionError("Duplicated new TRAIN or historic source prompts")
    # These names are excluded from both precommitted public tests by construction;
    # never read the QA/VALID answer files inside the trainer.
    assert all("LumoProbe" not in q and "KivaProbe" not in q and "Eval" not in q for q in novel_questions)
    cfg=build_config(next(z for z in CANDIDATES if z[0]=="small_320x6"))
    model,sources=create_small_paid_model(cb,cfg)
    opt=torch.optim.Adam([src.centroids for src in sources.values()],lr=LR)
    keys=sorted(sources)
    checkpoints=[]
    loss_history=[]
    olds=news=0
    all_pairs=historic+fresh
    for step in range(STEPS+1):
        if step in (0,640,1280,1920,2560):
            checkpoints.append(score(model,sources,all_pairs,"old_24_plus_new_64_TRAIN",step))
        if step==STEPS:break
        if step%3==0:
            p=historic[olds%len(historic)];olds+=1
        else:
            p=fresh[news%len(fresh)];news+=1
        src=sources[keys[step%len(keys)]]
        captured=[]
        def hook(_module,_inputs,weight):
            weight.retain_grad()
            captured.append(weight)
        handle=src.register_forward_hook(hook)
        clear(sources)
        opt.zero_grad(set_to_none=True)
        try:
            loss,detail=paired_loss(model,src,p)
            loss_before=float(loss.detach())
            loss.backward()
        finally:
            handle.remove()
        if not captured or captured[-1].grad is None:
            raise AssertionError("Paid full-model dictionary gradient missing")
        gradient=captured[-1].grad.detach().cpu().numpy().copy()
        if sum(w.centroids.grad is not None for w in sources.values())!=43:
            raise AssertionError("All 43 paid codebooks must participate in training")
        norm=float(torch.nn.utils.clip_grad_norm_(
             [w.centroids for w in sources.values()],1.,error_if_nonfinite=True))
        opt.step()
        with torch.no_grad():
            for w in sources.values():w.centroids.clamp_(-16,16)
        clear(sources)
        swaps=0
        if (step+1)%8==0:
            with torch.inference_mode():
                baseline,_=paired_loss(model,src,p)
                matrix=src().detach().cpu().numpy().copy()
            proposed=propose_codes(src,gradient,matrix)
            if proposed["proposed_groups"]>512:
                raise AssertionError("5bit paid index proposal exceeds original constraint")
            if proposed["proposed_groups"]:
                with torch.no_grad():apply_codes(src,proposed,True)
                clear(sources)
                with torch.inference_mode():candidate_loss,_=paired_loss(model,src,p)
                if np.isfinite(float(candidate_loss)) and float(candidate_loss)+1e-7<float(baseline):
                    swaps=proposed["proposed_groups"]
                else:
                    with torch.no_grad():apply_codes(src,proposed,False)
        clear(sources)
        loss_history.append({"step":step+1,"id":p["pair_id"],"domain":p["domain"],
                             "loss":loss_before,"codebook_norm":norm,
                             "accepted_5bit_swaps":swaps})
    packed,info=replace_discrete(cb,sources,strict=False)
    dest=args.out/"CALIBRI2_S3_22M_2560steps_paid.cb59"
    dest.write_bytes(packed)
    if len(packed)!=BYTES or info["complete_bpp"]>.700:
        raise AssertionError("S3 physical weight budget exceeded")
    reloaded,rs=create_small_paid_model(CB59(packed),cfg)
    in_memory=score(model,sources,all_pairs,"last_in_memory",STEPS)
    post=score(reloaded,rs,all_pairs,"last_reloaded",STEPS)
    if [(a["id"],a["side"],a["generated_ids"]) for a in in_memory["actual_answers"]]!=[
        (a["id"],a["side"],a["generated_ids"]) for a in post["actual_answers"]]:
        raise AssertionError("S3 output changed upon physically packing/reloading")
    save(args.out/"ACTUAL_CALIBRI2_S3_TRAIN.json",{
        "experiment":"S3 22M systematic diverse-language expansion",
        "initial_S2_SHA256":START_SHA,"result_sha256":hashlib.sha256(packed).hexdigest(),
        "precommitted_steps":STEPS,"existing_original_TRAIN_pairs":24,
        "new_multi_domain_TRAIN_pairs":64,"new_pair_domains":dict(Counter(p["domain"] for p in fresh)),
        "old_steps":olds,"new_steps":news,"old_new_rehearsal_ratio":"1:2",
        "TRAIN_only_no_QA_VALID_checkpoint_selection":True,
        "no_FP32_teacher_or_comparison":True,"no_LIVE_changes":True,
        "physical_bpp":info["complete_bpp"],"physical_bytes":len(packed),
        "milestones":checkpoints,"final_reloaded_probe":post,
        "final_reload_full_answers_identical":True,"loss_curve":loss_history})
    print(json.dumps({"event":"CALIBRI2_S3_TRAIN_COMPLETE",
        "full_with_EOS":post["correct_full_EOS"],"total":post["total"],
        "bpp":info["complete_bpp"],"steps":STEPS,
        "sha256":hashlib.sha256(packed).hexdigest()}),flush=True)

if __name__=="__main__":
    main()
