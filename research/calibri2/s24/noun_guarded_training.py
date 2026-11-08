"""S24: paid 36M noun-specific contrastive continuation with Green Tunnel guards.

Evidence motivating THIS new experiment:
  S20 learned complete TRAIN color+noun 0 -> 8/32 (S22), first token
  6 -> 23/32 and no longer chose early EOS; nevertheless S23 measured
  0/96 independent full two-word answers and just 2/96 correct nouns.
  S20 trained opposing *colors* on either side of each pair. This does
  not isolate second-word selection. S24 instead explicitly trains pairs
  sharing the SAME color but opposing nouns, including same-object color
  contrasts as a control. The loss directly margins the first NOUN token
  under a teacher-forced correct color. Its multi-layer Green Tunnel:
   (1) project each primary gradient against TWO TRAIN-only anchors,
       one historical skill and one previous S20 two-word skill;
   (2) exact paid-FP16 trial backtracking on the actual current TASK
       and both selected old TRAIN guard losses (not only dot product);
   (3) paid 5bit group swaps only on actual new TRAIN gain and guarded
       old TRAIN loss;
   (4) keep physical <=0.7 bpp, paired CPU speed, sealed final weights.
  A failed guard backtracks; not all historical knowledge is provably
  preserved by rotating two guards. No original FP32 or its logits.
  No QA/VALID/TEST in updates, no LIVE code or inference adapter.

ONE deterministic, precommitted 640-step program. Start from SEALED S20
physically paid model SHA, not a QA-selected checkpoint. S24 independent
QA, frozen before training, uses 8 COLOR+NEW-NOUN combinations excluded
from these new TRAIN questions. All S20 old withheld combinations remain
excluded. Check success only on heldout full answers, never first token.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.optim import propose_codes,apply_codes
from v185.codec import replace_discrete
from v186.pairs import build_pairs,tokenize_one
from calibri2.s3.diverse_task_replay import make_new_pairs as s3_pairs
from calibri2.s6.train_29m import new_pairs as s4_pairs
from calibri2.s8.broad_replay import (
 SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC,PAID_OPERATORS,
 weighted_pair_loss,freeze_language_corpus,
)
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s11.green_tunnel_protected import training_probe
from calibri2.s20.multiword_curriculum import (
 make_training_pairs as s20_train_pairs,
 get_language_loss, take_grads,
)
from v174.run import save

SEED=2124
START_SHA="4b62bcec5e40c2b7b73d3be38e4c365d1cf9a62614420497490795534477e93d"
COLORS=("red","blue","green","yellow")
NOUNS=("cup","book","bag","lamp","chair","hat",
       "plate","scarf","clock","spoon","pencil","bottle")
STEPS=640
PAIR_NOUN=320
PAIR_COLOR=128
LR_CANDIDATES=(.16,.08,.04,.02,.01)
GUARD_TOL=.003
TASK_IMPROVE=1e-7
NOUN_MARGIN=2.25
NOUN_WEIGHT=9.0
COLOR_WEIGHT=1.0
EOS_WEIGHT=2.5
INDEX_PERIOD=16
PROPOSED_INDEX_CAP=512
CONTEXT=48
SCHEDULE=("noun","noun","noun","noun","color","noun","noun",
          "noun","old","noun","noun","language")
PATTERNS=(
 "TRAIN exercise {n}: {a} carries a {left} and {b} carries a {right}. Give only the two-word item for {a}.",
 "TRAIN item {n}: In the classroom, {a} received a {left}; {b} received a {right}. Name the object belonging to {a}, color then noun.",
 "TRAIN card {n}: The {left} is assigned to {a}; the {right} belongs to {b}. What is assigned to {a}? Give two words.",
 "TRAIN inventory {n}: beside {a} is a {left}, beside {b} is a {right}. Return the color and the noun for {a}'s item.",
 "TRAIN object study {n}. Record: {a} -> {left}; {b} -> {right}. Write exactly the two words that identify {a}'s item.",
 "TRAIN store sample {n}: {a} chose a {left}. Separately, {b} chose a {right}. Reply using a two-word description of {a}'s choice.",
)

def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def allowed_combos():
    # Explicitly EXCLUDE the eight original S20 withheld combinations
    # AND every new-noun/color combination held out for S24.
    ok=[(ci,oi) for ci in range(4) for oi in range(len(NOUNS))
        if (ci+oi)%3!=0]
    if len(ok)!=32:raise AssertionError("Expected 32 TRAIN combinations")
    assert all((c+o)%3!=0 for c,o in ok)
    return ok

def _prefix_len(tok, color, noun):
    tokens=tok.encode(color+" "+noun,add_special_tokens=False)
    prefix=tok.encode(color,add_special_tokens=False)
    if not tokens[:len(prefix)]==prefix or len(tokens)<=len(prefix):
        raise AssertionError("S24 tokenization cannot isolate noun after exact color")
    return len(prefix)

def create_train_pairs(tok):
    allowed=allowed_combos()
    bycolor={c:[o for x,o in allowed if c==x] for c in range(4)}
    bynoun={o:[c for c,x in allowed if x==o] for o in range(len(NOUNS))}
    if any(len(v)<2 for v in list(bycolor.values())+list(bynoun.values())):
        raise AssertionError("Cannot construct contrasting TRAIN noun/color choices")
    examples=[]
    for kind,count in (("noun",PAIR_NOUN),("color",PAIR_COLOR)):
        for i in range(count):
            if kind=="noun":
                c=i%4
                nouns=bycolor[c]
                o1=nouns[(i//4)%len(nouns)]
                possible=[o for o in nouns if o!=o1 and
                     tok.encode(COLORS[c]+" "+NOUNS[o],add_special_tokens=False)[
                          _prefix_len(tok,COLORS[c],NOUNS[o])]
                       !=tok.encode(COLORS[c]+" "+NOUNS[o1],add_special_tokens=False)[
                          _prefix_len(tok,COLORS[c],NOUNS[o1])]]
                if not possible:raise AssertionError("No distinct noun-token negative")
                o2=possible[(i//16+i//4)%len(possible)]
                first=(c,o1);second=(c,o2)
            else:
                o=(i*7+i//4)%len(NOUNS)
                cs=bynoun[o]
                c1=cs[(i//len(NOUNS))%len(cs)]
                c2=cs[(i//len(NOUNS)+1)%len(cs)]
                first=(c1,o);second=(c2,o)
            A=f"{COLORS[first[0]]} {NOUNS[first[1]]}"
            B=f"{COLORS[second[0]]} {NOUNS[second[1]]}"
            if A==B:raise AssertionError("TRAIN contrasts identical")
            pattern=PATTERNS[(i//4)%len(PATTERNS)]
            owner=f"TrainKemi{kind}{i:03d}";other=f"TrainHaro{kind}{i:03d}"
            left=tokenize_one(tok,
                pattern.format(n=i,a=owner,b=other,left=A,right=B),A)
            right=tokenize_one(tok,
                pattern.format(n=i,a=owner,b=other,left=B,right=A),B)
            m1=_prefix_len(tok,COLORS[first[0]],NOUNS[first[1]])
            m2=_prefix_len(tok,COLORS[second[0]],NOUNS[second[1]])
            if kind=="noun" and left["first_expected_id"]!=right["first_expected_id"]:
                raise AssertionError("Same-color pairs must have identical first answer token")
            if kind=="color" and left["first_expected_id"]==right["first_expected_id"]:
                raise AssertionError("Different-color pairs must have opposite first token")
            if kind=="noun" and int(left["targets"][m1])==int(right["targets"][m2]):
                raise AssertionError("Same-color contrasting noun first token identical")
            examples.append({"pair_id":4000+len(examples),
                 "domain":"S24_NOUN_BINDING_TRAIN" if kind=="noun" else "S24_COLOR_RECALL_TRAIN",
                 "split":"TRAIN","kind":kind,"left":left,"right":right,
                 "noun_token_position_left":m1,
                 "noun_token_position_right":m2,
                 "first_color_noun":list(first),
                 "second_color_noun":list(second),
            })
    questions=[p[side]["question"] for p in examples for side in ("left","right")]
    if len(examples)!=PAIR_NOUN+PAIR_COLOR or len(set(questions))!=len(questions):
        raise AssertionError("S24 duplicate or incomplete TRAIN")
    # Absolutely no withheld combo appears in TRAIN answer, *including*
    # the foil, so this test can measure true combinatorial transfer.
    for p in examples:
        for side in ("left","right"):
            reply=p[side]["answer"].split()
            c=COLORS.index(reply[0]);o=NOUNS.index(reply[1])
            if (c+o)%3==0:raise AssertionError("Heldout answer leaked into TRAIN")
    return {"noun":examples[:PAIR_NOUN],"color":examples[PAIR_NOUN:]}

def noun_loss(model,pair):
    # The PRIMARY distinctive S24 change: compare two answers with an
    # IDENTICAL first color but different nouns and require exact lexical
    # noun next-token selection (and only then EOS).
    losses=[];positions=[];logits_list=[]
    for side in ("left","right"):
        row=pair[side]
        logits=model(input_ids=row["ids"][None],use_cache=False).logits[0][row["positions"]]
        target=row["targets"]
        noun_pos=pair["noun_token_position_"+side]
        if not 0<noun_pos<len(target)-1:
            raise AssertionError("Noun target position was not at continuation")
        ce=F.cross_entropy(logits,target,reduction="none")
        weights=torch.full_like(ce,COLOR_WEIGHT)
        weights[noun_pos:-1]=NOUN_WEIGHT
        weights[-1]=EOS_WEIGHT
        loss=(ce*weights).sum()/weights.sum()
        eos_id=int(model.config.eos_token_id)
        noun_gold_logit=logits[noun_pos,int(target[noun_pos])]
        early_eos=F.relu(2.25+logits[noun_pos,eos_id]-noun_gold_logit)
        losses.append(loss+.25*early_eos)
        positions.append(noun_pos)
        logits_list.append(logits)
    if pair["kind"]=="noun":
        l,r=pair["left"],pair["right"]
        lpos,rpos=positions
        own_l=int(l["targets"][lpos]);own_r=int(r["targets"][rpos])
        m1=F.relu(NOUN_MARGIN-logits_list[0][lpos,own_l]+logits_list[0][lpos,own_r])
        m2=F.relu(NOUN_MARGIN-logits_list[1][rpos,own_r]+logits_list[1][rpos,own_l])
        return .5*(losses[0]+losses[1])+.60*.5*(m1+m2)
    if pair["kind"]=="color":
        l,r=pair["left"],pair["right"]
        t1=int(l["targets"][0]);t2=int(r["targets"][0])
        m1=F.relu(1.25-logits_list[0][0,t1]+logits_list[0][0,t2])
        m2=F.relu(1.25-logits_list[1][0,t2]+logits_list[1][0,t1])
        return .5*(losses[0]+losses[1])+.20*.5*(m1+m2)
    raise AssertionError("Unexpected S24 learning domain")

def loss(model,kind,pair,tokens):
    if kind in ("noun","color"):return noun_loss(model,pair)
    if kind=="old":return weighted_pair_loss(model,pair)
    if kind=="language":return get_language_loss(model,tokens)
    raise AssertionError("Unknown TRAIN curriculum type")

def safe_measure(model,sources,which,kind="old"):
    clear(sources)
    with torch.inference_mode():
        val=float(weighted_pair_loss(model,which))
    clear(sources)
    if not np.isfinite(val):raise AssertionError("Nonfinite TRAIN guard")
    return val

def safe_objective(model,sources,kind,pair,tokens):
    clear(sources)
    with torch.inference_mode():
        val=float(loss(model,kind,pair,tokens))
    clear(sources)
    if not np.isfinite(val):raise AssertionError("Nonfinite TRAIN task objective")
    return val

def train():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    setup()
    raw=args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=START_SHA or len(raw)!=SOURCE_BYTES:
        raise AssertionError("S24 MUST start from immutable physically paid S20, not QA-selected alternate")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Wrong original 36M architecture")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    new=create_train_pairs(tok)
    old=build_pairs(tok,"TRAIN")+s3_pairs(tok)+s4_pairs(tok)
    prev=s20_train_pairs(tok)
    if len(old)!=144 or len(prev)!=192:raise AssertionError("Historical TRAIN protection source missing")
    if set(p[s]["question"] for kind in new for p in new[kind] for s in ("left","right")) & \
       set(p[s]["question"] for p in old+prev for s in ("left","right")):
        raise AssertionError("TRAIN question exact overlap")
    public,public_meta=freeze_language_corpus(tok)
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    keys=sorted(sources)
    if len(keys)!=PAID_OPERATORS:raise AssertionError("S24 missing paid codebooks")
    params=[sources[k].centroids for k in keys]
    prev_probe=prev[:6]+prev[-6:]
    old_probe=old[:6]+old[-6:]
    new_probe=new["noun"][:6]+new["noun"][-6:]
    before_prev=training_probe(model,sources,prev_probe)
    before_old=training_probe(model,sources,old_probe)
    before_new=training_probe(model,sources,new_probe)
    trace=[]
    counters=Counter()
    projections=0;accepted=0;accepted_groups=0;candidate_group_rejects=0
    for step in range(STEPS):
        kind=SCHEDULE[step%len(SCHEDULE)]
        idx=counters[kind];counters[kind]+=1
        if kind in ("noun","color"):
            pair=new[kind][idx%len(new[kind])]
            tokens=None
        elif kind=="old":
            pair=old[idx%len(old)]
            tokens=None
        else:
            pair=None
            start=(idx%(len(public)//CONTEXT))*CONTEXT
            tokens=public[start:start+CONTEXT]
        old_guard=old[(step*7)%len(old)]
        word_guard=prev[(step*11)%len(prev)]
        # At most two actual skill barriers: five hard barriers S14
        # overconstrained updates; one S12 barrier was too weak.
        index_key=keys[(step//INDEX_PERIOD)%len(keys)]
        captured=index_key if (step+1)%INDEX_PERIOD==0 else None
        objective=lambda: loss(model,kind,pair,tokens)
        task_loss,task_g,matrix_g=take_grads(model,sources,keys,objective,captured)
        _,past_g,_=take_grads(model,sources,keys,
                              lambda:weighted_pair_loss(model,old_guard))
        _,word_g,_=take_grads(model,sources,keys,
                              lambda:weighted_pair_loss(model,word_guard))
        # PCGrad-like projection separately against both guards, plus
        # modest TRAIN replay gradient. Real accepted loss is checked
        # independently below, so dot-product alone is never "proof".
        update=[x.double().clone() for x in task_g]
        conflicts=0
        for guard in (past_g,word_g):
            norm=sum(torch.sum(x.double()*x.double()) for x in guard)
            dot=sum(torch.sum(x.double()*y.double()) for x,y in zip(update,guard))
            if norm<=0 or not torch.isfinite(norm):
                raise AssertionError("S24 zero/nonfinite protection gradient")
            if dot<0:
                coef=dot/norm
                update=[u-coef*g.double() for u,g in zip(update,guard)]
                conflicts+=1
        update=[u+.12*(g.double()+h.double())/2 for u,g,h in
                zip(update,past_g,word_g)]
        projections+=conflicts
        norm=float(torch.sqrt(sum(torch.sum(u*u) for u in update)))
        if norm<=1e-12 or not np.isfinite(norm):
            raise AssertionError("S24 nonfinite projected update")
        normfactor=max(1.0,norm)
        directions=[u.float()/normfactor for u in update]
        old_task=safe_objective(model,sources,kind,pair,tokens)
        old_scores=[safe_measure(model,sources,p) for p in (old_guard,word_guard)]
        with torch.no_grad():
            backup=[param.detach().clone() for param in params]
        chosen=None
        trials=[]
        for lr in LR_CANDIDATES:
            with torch.no_grad():
                for param,prev_val,delta in zip(params,backup,directions):
                    param.copy_((prev_val-lr*delta).clamp(-16,16).half().float())
            task_after=safe_objective(model,sources,kind,pair,tokens)
            guards_after=[safe_measure(model,sources,p)
                           for p in (old_guard,word_guard)]
            task_pass=task_after+TASK_IMPROVE<old_task
            guards_pass=all(x<=oldv+GUARD_TOL for x,oldv in zip(guards_after,old_scores))
            trials.append({"lr":lr,"TRAIN_task_loss":task_after,
                           "TRAIN_old_guard_losses":guards_after,
                           "task_improved":task_pass,"two_guards_preserved":guards_pass})
            if task_pass and guards_pass:
                chosen=lr;accepted+=1
                break
        if chosen is None:
            with torch.no_grad():
                for p,original in zip(params,backup):
                    p.copy_(original)
            clear(sources)
        changed=0;index_accepted=False
        if chosen is not None and captured:
            source=sources[captured]
            base_task=safe_objective(model,sources,kind,pair,tokens)
            base_guard=[safe_measure(model,sources,p) for p in (old_guard,word_guard)]
            with torch.inference_mode():
                weight=source().detach().cpu().numpy().copy()
            proposal=propose_codes(source,matrix_g,weight)
            if proposal["proposed_groups"]>PROPOSED_INDEX_CAP:
                raise AssertionError("Paid 5bit proposal capacity violated")
            if proposal["proposed_groups"]:
                with torch.no_grad():apply_codes(source,proposal,True)
                clear(sources)
                changed_task=safe_objective(model,sources,kind,pair,tokens)
                changed_guard=[safe_measure(model,sources,p) for p in (old_guard,word_guard)]
                if (changed_task+TASK_IMPROVE<base_task and
                     all(a<=b+GUARD_TOL for a,b in zip(changed_guard,base_guard))):
                    changed=int(proposal["proposed_groups"])
                    accepted_groups+=changed
                    index_accepted=True
                else:
                    candidate_group_rejects+=1
                    with torch.no_grad():apply_codes(source,proposal,False)
                    clear(sources)
        trace.append({
            "step":step+1,"kind":kind,
            "TRAIN_pair_id":None if pair is None else pair["pair_id"],
            "historical_guard_pair_id":old_guard["pair_id"],
            "multiword_guard_pair_id":word_guard["pair_id"],
            "task_loss_before":old_task,"guard_loss_before":old_scores,
            "two_gradient_conflicts_projected":conflicts,
            "gradient_l2":norm,
            "all_trials":trials,
            "accepted_physically_quantized_lr":chosen,
            "accepted_real_paid_5bit_groups":changed,
            "rejected_code_proposal":bool(chosen is not None and captured and not index_accepted)
        })
        if (step+1)%80==0:
            print(json.dumps({
              "event":"S24_PAID_36M_NOUN_GUARDED_TRAIN_PROGRESS",
              "step":step+1,"counts":dict(counters),
              "accepted":accepted,"gradient_conflicts":projections,
              "accepted_5bit_groups":accepted_groups,
              "rejected_code_proposals":candidate_group_rejects,
              "last_task":kind,"last_loss":old_task
            }),flush=True)
    packed,phys=replace_discrete(archive,sources,strict=False)
    final=args.output/"CALIBRI2_S24_36M_NOUN_GUARDED_640_PAID.cb59"
    final.write_bytes(packed)
    sha=hashlib.sha256(packed).hexdigest()
    if len(packed)!=SOURCE_BYTES or phys["complete_bpp"]>.700:
        raise AssertionError("Physically paid 36M 0.7bpp contract violated")
    if sha==START_SHA:
        raise AssertionError("S24 learned no physical weight updates")
    reloaded,reload_src=create_small_paid_model(CB59(final.read_bytes()),build_config(MODEL_SPEC))
    mem_new=training_probe(model,sources,new_probe)
    reload_new=training_probe(reloaded,reload_src,new_probe)
    if mem_new!=reload_new:
        raise AssertionError("Paid CB59 reload changed actual generated TRAIN words")
    after_prev=training_probe(reloaded,reload_src,prev_probe)
    after_old=training_probe(reloaded,reload_src,old_probe)
    def n_correct(rows):return sum(x["correct_full_EOS"] for x in rows)
    def lost(before,after):
        return sum(b["correct_full_EOS"] and not a["correct_full_EOS"] for b,a in zip(before,after))
    report={
      "stage":"S24_PHYSICAL_36M_SAME_COLOR_OPPOSING_NOUN_GREEN_TUNNEL",
      "starting_sealed_S20_sha256":START_SHA,"final_SHA256":sha,
      "original_params":SOURCE_PARAMS,"fully_paid_archive_bytes":len(packed),
      "complete_bpp":phys["complete_bpp"],
      "updated_paid_FP16_codebook_cells":phys["actually_changed_FP16_codebook_cells"],
      "updated_paid_5bit_index_groups":phys["actual_changed_5bit_group_indices"],
      "total_steps":STEPS,"accepted_FP16_steps":accepted,
      "rejected_FP16_steps":STEPS-accepted,
      "gradient_conflicts_projected":projections,
      "accepted_paid_5bit_proposed_groups":accepted_groups,
      "5bit_candidate_rejections_on_guard_or_task":candidate_group_rejects,
      "training_noun_opposite_same_color_pairs":PAIR_NOUN,
      "training_color_opposite_same_noun_pairs":PAIR_COLOR,
      "excluded_all_eight_new_noun_combinations_from_TRAIN":True,
      "excluded_all_previous_S20_withheld_combinations_from_TRAIN":True,
      "original_old_TRAIN_pairs":len(old),
      "rehearsal_S20_TRAIN_pairs":len(prev),
      "actual_TRAIN_noun_pre_correct":n_correct(before_new),
      "actual_TRAIN_noun_post_correct":n_correct(reload_new),
      "actual_TRAIN_S20_pre_correct":n_correct(before_prev),
      "actual_TRAIN_S20_post_correct":n_correct(after_prev),
      "actual_TRAIN_old_pre_correct":n_correct(before_old),
      "actual_TRAIN_old_post_correct":n_correct(after_old),
      "lost_old_TRAIN_answers":lost(before_old,after_old),
      "lost_previous_multiword_TRAIN_answers":lost(before_prev,after_prev),
      "actual_pre_new_TRAIN_answers":before_new,
      "actual_post_new_TRAIN_answers":reload_new,
      "actual_pre_S20_TRAIN_answers":before_prev,
      "actual_post_S20_TRAIN_answers":after_prev,
      "actual_pre_old_TRAIN_answers":before_old,
      "actual_post_old_TRAIN_answers":after_old,
      "precommitted_single_final_checkpoint":True,
      "no_independent_QA_in_training_or_checkpoint_selection":True,
      "no_pretrained_135m_FP32_or_logits":True,
      "no_runtime_inference_adapter":True,
      "all_640_per_step_trial_decisions":trace,
      "real_generalization_NOT_yet_claimed":True,
      "public_language_TRAIN_metadata":public_meta,
    }
    save(args.output/"ACTUAL_CALIBRI2_S24_REAL_GUARDED_NOUN_TRAIN.json",report)
    print(json.dumps({
      "event":"S24_PAID_36M_NOUN_BINDING_CHECKPOINT_SEALED",
      "SHA256":sha,"complete_bpp":phys["complete_bpp"],
      "accepted_steps":accepted,"rejected":STEPS-accepted,
      "TRAIN_noun_before":report["actual_TRAIN_noun_pre_correct"],
      "TRAIN_noun_after":report["actual_TRAIN_noun_post_correct"],
      "TRAIN_S20_before":report["actual_TRAIN_S20_pre_correct"],
      "TRAIN_S20_after":report["actual_TRAIN_S20_post_correct"],
      "old_correct_lost":report["lost_old_TRAIN_answers"],
      "updated_5bit_groups":accepted_groups
    }),flush=True)

if __name__=="__main__":
    train()
