"""CALIBRI2 S20: actually train 36M physical 5bit model to produce TWO WORDS.

Built in response to S11 independent 29->71/192 recovery with 0/80
new two/three-word answers, and S14 multi-hard-guard 50->49/192.
We do NOT pursue 135M FP32 parity prematurely. S20's single precommitted
source is SEALED S7 (chosen before evaluating this S20 suite), with a
96-case unseen-color/object QA already frozen separately before training.

Research hypothesis: the model's EOS preference and weak continuation
gradient prevent two-word outputs. A bounded curriculum trains 2-word
color+noun answers with later-token weighted CE, an explicit premature-EOS
margin, contrasting pairs and old-task rehearsal. A second multi-layer
protection stack uses projected TRAIN-gradient rehearsal, paid physical
5-bit code mutations guarded on old TRAIN tasks, and an EXACT size/bit
budget. NEVER train against QA or original 135M FP32/teacher logits.

One fixed 960-step training program, never QA-select checkpoint; retain
negative results. No extra runtime adapter/sidecar. All 85 codebooks paid.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.optim import propose_codes,apply_codes
from v185.codec import replace_discrete
from v186.pairs import build_pairs,tokenize_one
from calibri2.s3.diverse_task_replay import make_new_pairs as s3_pairs
from calibri2.s6.train_29m import new_pairs as s4_pairs
from calibri2.s8.broad_replay import (
    SOURCE_SHA,SOURCE_BYTES,SOURCE_PARAMS,PAID_OPERATORS,MODEL_SPEC,
    weighted_pair_loss,freeze_language_corpus
)
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s11.green_tunnel_protected import training_probe
from v174.run import save

SEED=2120
STEPS=960
LR=.0017
GUARD_WEIGHT=.15
EOS_MARGIN=3.0
LATE_WEIGHT=6.0
EOS_FINAL_WEIGHT=2.0
INDEX_PERIOD=16
PAIR_COUNT=192
CONTEXT=48
SCHEDULE=("two_word","two_word","two_word","two_word",
          "two_word","two_word","old_replay","two_word",
          "two_word","old_replay","two_word","language")
COLORS=("red","blue","green","yellow")
OBJECTS=("cup","book","bag","lamp","chair","hat")
TRAIN_PATTERNS=(
 "Practice label {idx}. {a} has a {p} and {b} has a {q}. Give {a}'s thing in exactly two words: the color then the noun.",
 "Practice label {idx}. A shelf for {a} shows '{p}'. A shelf for {b} shows '{q}'. Write the two-word description for {a}.",
 "A made-up classroom inventory for {idx}: {a} received a {p}; {b} received a {q}. Respond with two words describing {a}'s object.",
 "Training entry {idx}: {a} holds the {p}. {b} holds the {q}. Give only {a}'s item's two-word color-and-object name.",
 "Rehearsal note {idx} displays '{p}' next to {a} and '{q}' next to {b}. Copy the two words next to {a}.",
 "In exercise {idx}, {a} is associated with {p} while {b} is associated with {q}. Name {a}'s item using two words, no explanation.",
)


def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def training_combinations():
    result=[(c,o) for c in range(len(COLORS)) for o in range(len(OBJECTS))
            if (c+o)%3!=0]
    if len(result)!=16:
        raise AssertionError("Exactly 16 TRAIN color-object combinations")
    for c in range(4):
        if sum(x==c for x,_ in result)!=4:
            raise AssertionError("Wrong balanced train noun inventory")
    return result


def make_training_pairs(tok):
    allowed=training_combinations()
    by_color={c:[(x,o) for x,o in allowed if x==c] for c in range(4)}
    data=[]
    for i in range(PAIR_COUNT):
        c=i%4
        d=(c+1+(i//4)%3)%4
        if c==d:
            raise AssertionError("Not opposing first-token contrast colors")
        left=by_color[c][(i//4)%len(by_color[c])]
        right=by_color[d][(i//5+i//16)%len(by_color[d])]
        p=COLORS[left[0]]+" "+OBJECTS[left[1]]
        q=COLORS[right[0]]+" "+OBJECTS[right[1]]
        template=TRAIN_PATTERNS[(i//3)%len(TRAIN_PATTERNS)]
        a=f"TrainDori{i:03d}"
        b=f"TrainPeli{i:03d}"
        question=template.format(idx=i,p=p,q=q,a=a,b=b)
        reverse=template.format(idx=i,p=q,q=p,a=a,b=b)
        l=tokenize_one(tok,question,p)
        r=tokenize_one(tok,reverse,q)
        if l["first_expected_id"]==r["first_expected_id"]:
            raise AssertionError("S20 opposing two-word TRAIN samples share first token")
        if len(tok.encode(p,add_special_tokens=False))<2:
            raise AssertionError("Two-word TRAIN tokenization cannot be one token")
        data.append({
            "pair_id":3000+i,
            "domain":"S20_TRAIN_TWO_WORD_BINDING",
            "split":"TRAIN","left":l,"right":r,
            "actual_color_noun_left":list(left),
            "actual_color_noun_right":list(right),
        })
    questions=[pair[s]["question"] for pair in data for s in ("left","right")]
    if len(data)!=PAIR_COUNT or len(set(questions))!=PAIR_COUNT*2:
        raise AssertionError("S20 train set incomplete or duplicate")
    return data


def get_train_loss(model,pair):
    """TRAIN-only teacher-forced full answer: first, ALL later, then EOS.

    Explicitly counteracts 'red'+EOS collapse by supervising the second
    word and every continuation token. The no-early-EOS margin is charged
    where the gold continuation is *not* EOS. The final correct EOS
    remains trained at the actual final answer position.
    """
    scored=[]
    first_logits=[]
    targets=[]
    for side in ("left","right"):
        row=pair[side]
        logits=model(input_ids=row["ids"][None],use_cache=False).logits[0][row["positions"]]
        target=row["targets"]
        if len(target)<3 or int(target[-1])!=int(model.config.eos_token_id):
            raise AssertionError("S20 two-word target must contain continuation and EOS")
        ce=F.cross_entropy(logits,target,reduction="none")
        weights=torch.ones_like(ce)
        weights[1:-1]=LATE_WEIGHT
        weights[-1]=EOS_FINAL_WEIGHT
        weighted=(weights*ce).sum()/weights.sum()
        eos_id=int(model.config.eos_token_id)
        continuation=logits[1:-1]
        expected=target[1:-1]
        if continuation.shape[0]<1:
            raise AssertionError("No second-word token available for training")
        rowidx=torch.arange(len(expected),dtype=torch.long)
        preferred=continuation[rowidx,expected]
        competing=continuation[:,eos_id]
        delayed_eos=(F.relu(EOS_MARGIN + competing - preferred)).mean()
        scored.append(weighted + 0.35*delayed_eos)
        first_logits.append(logits[0])
        targets.append(int(target[0]))
    margins=[]
    for j in range(2):
        chosen=first_logits[j][targets[j]]
        other=first_logits[j][targets[1-j]]
        margins.append(F.relu(1.0-chosen+other))
    return .5*(scored[0]+scored[1])+.15*.5*(margins[0]+margins[1])


def get_language_loss(model,ids):
    logits=model(input_ids=ids[None],use_cache=False).logits[0]
    return F.cross_entropy(logits[:-1],ids[1:])


def take_grads(model,sources,keys,loss_generator,target_name=None):
    clear(sources)
    model.zero_grad(set_to_none=True)
    captured=[]
    hook=None
    if target_name:
        def retain(_m,_x,dense):
            dense.retain_grad()
            captured.append(dense)
        hook=sources[target_name].register_forward_hook(retain)
    try:
        val=loss_generator()
        if not torch.isfinite(val):
            raise AssertionError("Nonfinite TRAIN loss")
        observed=float(val.detach())
        val.backward()
    finally:
        if hook:
            hook.remove()
    grads=[]
    for name in keys:
        x=sources[name].centroids.grad
        if x is None or not torch.isfinite(x).all():
            raise AssertionError("Training missed one of 85 paid dictionaries")
        grads.append(x.detach().clone())
    matrix=None
    if target_name:
        captured=[x.grad for x in captured if x.grad is not None]
        if not captured:
            raise AssertionError("No source matrix gradient for paid 5bit index proposal")
        matrix=torch.stack(captured).sum(dim=0).detach().cpu().numpy().copy()
    clear(sources)
    return observed,grads,matrix


def combine_projected(main,guard,params):
    dot=sum(torch.sum(a.double()*b.double()) for a,b in zip(main,guard))
    norm=sum(torch.sum(g.double().square()) for g in guard)
    if not torch.isfinite(norm) or norm<=0:
        raise AssertionError("Unusable historical TRAIN guard")
    rate=float(dot/norm) if dot<0 else 0.
    combined=[a-rate*b+GUARD_WEIGHT*b for a,b in zip(main,guard)]
    for p,g in zip(params,combined):
        p.grad=g.to(dtype=p.dtype).clone()
    after=float(sum(torch.sum(g.double()*b.double()) for g,b in zip(combined,guard)))
    if after<-1e-8:
        raise AssertionError("Old TRAIN projection is not protected")
    return bool(dot<0),after


def train():
    pass


def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    setup()
    raw=args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA or len(raw)!=SOURCE_BYTES:
        raise AssertionError("S20 must start from original pre-S8, pre-S11 S7 36M")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Unexpected source architecture")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    small=make_training_pairs(tok)
    original=build_pairs(tok,"TRAIN")+s3_pairs(tok)+s4_pairs(tok)
    if len(original)!=144:
        raise AssertionError("Historical old TRAIN replay missing")
    if any(p[s]["question"] in {q[t]["question"] for q in small for t in ("left","right")}
           for p in original for s in ("left","right")):
        raise AssertionError("TRAIN overlaps new S20 inputs")
    # Three old-skill protective gradients, rotated across color, yes/no
    # and copying; these are OLD TRAIN only, NEVER independent QA.
    s3=original[24:88]
    guards={
      "old_color":s3[:24],
      "old_numeric":original[:12]+s3[24:40],
      "old_copy":original[88:88+32]
    }
    if any(not v for v in guards.values()):
        raise AssertionError("Missing historical guard classes")
    public_tokens,public_meta=freeze_language_corpus(tok)
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    keys=sorted(sources)
    if len(keys)!=PAID_OPERATORS:
        raise AssertionError("All 85 paid 36M matrices required")
    train_probe=small[:8]+small[-8:]
    old_probe=original[:8]+s3[:4]
    before_new=training_probe(model,sources,train_probe)
    before_old=training_probe(model,sources,old_probe)
    params=[sources[name].centroids for name in keys]
    opt=torch.optim.Adam(params,lr=LR)
    counter=Counter()
    events=[]
    accepted_indices=0
    rejected_index_proposals=0
    gradient_projections=0
    for step in range(STEPS):
        kind=SCHEDULE[step%len(SCHEDULE)]
        sample_index=counter[kind]
        counter[kind]+=1
        if kind=="two_word":
            current=small[sample_index%len(small)]
            sample_ids=None
            target=lambda: get_train_loss(model,current)
        elif kind=="old_replay":
            current=original[sample_index%len(original)]
            sample_ids=None
            target=lambda: weighted_pair_loss(model,current)
        else:
            current=None
            index=(sample_index%(len(public_tokens)//CONTEXT))*CONTEXT
            sample_ids=public_tokens[index:index+CONTEXT]
            target=lambda: get_language_loss(model,sample_ids)
        guard_kind=("old_color","old_numeric","old_copy")[step%3]
        guardset=guards[guard_kind]
        guard=guardset[(step//3)%len(guardset)]
        key=keys[(step//INDEX_PERIOD)%len(keys)]
        target_name=key if (step+1)%INDEX_PERIOD==0 else None
        new_loss,primary,matrix_grad=take_grads(
            model,sources,keys,target,target_name)
        old_loss,old_grad,_=take_grads(
            model,sources,keys,lambda:weighted_pair_loss(model,guard))
        projected,_=combine_projected(primary,old_grad,params)
        gradient_projections+=int(projected)
        norm=float(torch.nn.utils.clip_grad_norm_(
            params,1.0,error_if_nonfinite=True))
        opt.step()
        with torch.no_grad():
            for param in params:
                param.clamp_(-16,16)
        clear(sources)
        changed=0
        if target_name:
            with torch.inference_mode():
                baseline=float(target())
                clear(sources)
                old_guard_baseline=float(weighted_pair_loss(model,guard))
                clear(sources)
                current_weight=sources[key]().detach().cpu().numpy().copy()
            proposal=propose_codes(sources[key],matrix_grad,current_weight)
            if proposal["proposed_groups"]>512:
                raise AssertionError("Paid discrete index budget drift")
            if proposal["proposed_groups"]:
                with torch.no_grad():
                    apply_codes(sources[key],proposal,True)
                clear(sources)
                with torch.inference_mode():
                    cand=float(target())
                    clear(sources)
                    old_after=float(weighted_pair_loss(model,guard))
                if np.isfinite(cand) and cand+1e-7<baseline and old_after<=old_guard_baseline+1e-3:
                    changed=proposal["proposed_groups"]
                    accepted_indices+=changed
                else:
                    rejected_index_proposals+=1
                    with torch.no_grad():
                        apply_codes(sources[key],proposal,False)
                clear(sources)
        events.append({
          "step":step+1,"kind":kind,
          "train_id":None if current is None else current["pair_id"],
          "guard_kind":guard_kind,"guard_id":guard["pair_id"],
          "actual_target_loss":new_loss,
          "historical_replay_loss":old_loss,
          "projection_applied":projected,
          "preclip_norm":norm,
          "physically_paid_5bit_groups_accepted":changed
        })
        if (step+1)%120==0:
            print(json.dumps({
              "event":"S20_MULTIWORD_GREEN_TUNNEL_TRAIN_PROGRESS",
              "step":step+1,"counts":dict(counter),
              "new_loss":new_loss,"guard_loss":old_loss,
              "old_skill_gradient_projections":gradient_projections,
              "physically_paid_index_updates":accepted_indices,
              "rejected_5bit_guard_proposals":rejected_index_proposals
            }),flush=True)
    packed,info=replace_discrete(archive,sources,strict=False)
    dest=args.out/"CALIBRI2_S20_36M_960_MULTIWORD_PROTECTED_PAID.cb59"
    dest.write_bytes(packed)
    hash_new=hashlib.sha256(packed).hexdigest()
    if len(packed)!=SOURCE_BYTES or info["complete_bpp"]>.700 or hash_new==SOURCE_SHA:
        raise AssertionError("S20 36M physical payload unchanged or violates bit budget")
    reloaded,reloaded_src=create_small_paid_model(
        CB59(dest.read_bytes()),build_config(MODEL_SPEC))
    memory_new=training_probe(model,sources,train_probe)
    disk_new=training_probe(reloaded,reloaded_src,train_probe)
    if memory_new!=disk_new:
        raise AssertionError("S20 paid CB59 archive changed the multiword generated responses")
    disk_old=training_probe(reloaded,reloaded_src,old_probe)
    if len(before_new)!=len(disk_new) or len(before_old)!=len(disk_old):
        raise AssertionError("TRAIN old/new probe length changed")
    new_old_lost=sum(a["correct_full_EOS"] and not b["correct_full_EOS"]
                     for a,b in zip(before_old,disk_old))
    multi_before=sum(z["correct_full_EOS"] for z in before_new)
    multi_after=sum(z["correct_full_EOS"] for z in disk_new)
    old_before=sum(z["correct_full_EOS"] for z in before_old)
    old_after=sum(z["correct_full_EOS"] for z in disk_old)
    report={
      "stage":"S20_36M_EOS_AWARE_MULTIWORD_GREEN_TUNNEL_ACTUAL_TRAIN",
      "original_S7_sha256":SOURCE_SHA,
      "physical_final_SHA256":hash_new,
      "source_unique_params":SOURCE_PARAMS,
      "full_physical_bytes":len(packed),
      "complete_physical_bpp":info["complete_bpp"],
      "changed_existing_paid_5bit_indices":info["actual_changed_5bit_group_indices"],
      "changed_paid_8D_FP16_centroid_cells":info["actually_changed_FP16_codebook_cells"],
      "steps":STEPS,"learning_rate":LR,
      "train_color_object_pairs":PAIR_COUNT,
      "old_predeclared_TRAIN_pairs":len(original),
      "wiki_TRAIN_provenance":public_meta,
      "old_skills_gradient_rehearsal_and_projection":True,
      "second_answer_token_and_premature_EOS_margin":True,
      "old_gradient_projection_events":gradient_projections,
      "accepted_5bit_group_indices":accepted_indices,
      "historical_guard_rejected_index_proposals":rejected_index_proposals,
      "actual_TRAIN_new_probe_complete_before":multi_before,
      "actual_TRAIN_new_probe_complete_after":multi_after,
      "actual_TRAIN_old_probe_complete_before":old_before,
      "actual_TRAIN_old_probe_complete_after":old_after,
      "actual_TRAIN_old_correct_lost":new_old_lost,
      "all_original_TRAIN_multiword_actual_answers":before_new,
      "all_FINAL_TRAIN_multiword_actual_answers":disk_new,
      "all_original_TRAIN_old_actual_answers":before_old,
      "all_FINAL_TRAIN_old_actual_answers":disk_old,
      "full_960_step_actual_TRAIN_history":events,
      "no_FP32_135M_comparison_or_teacher":True,
      "no_QA_or_VALID_or_TEST_in_training":True,
      "precommitted_checkpoint_no_QA_early_stop":True,
      "no_unpaid_inference_sidecar":True,
      "quality_jump_NOT_claimed_without_frozen_QA":True,
    }
    save(args.out/"ACTUAL_CALIBRI2_S20_960_MULTIWORD_PROTECTED_TRAIN.json",report)
    print(json.dumps({
      "event":"S20_36M_FULL_MULTIWORD_PAID_CHECKPOINT_SEALED",
      "sha256":hash_new,"physical_bpp":info["complete_bpp"],
      "train_two_words_before":multi_before,
      "train_two_words_after":multi_after,
      "train_old_before":old_before,
      "train_old_after":old_after,
      "train_old_correct_lost":new_old_lost,
      "paid_index_updates":accepted_indices,
      "gradient_projections":gradient_projections,
    }),flush=True)


if __name__=="__main__":
    main()
