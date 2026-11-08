"""CALIBRI2 S11: a real, paid-weight, gradient-protected 36M recovery attempt.

Direct response to S10 negative evidence: S7 36M has 50/192 correctly
generated full frozen S8 answers, but 768 mixed S8 TRAIN steps reduced
that to 22/192, primarily by turning colors into Yes/No. S10 measured
signed task gradient directions (no training). S11 attempts a repair,
not a guaranteed breakthrough. Start ONLY from pre-S8 sealed S7, never
from QA-selected candidate. One deterministic, fixed 2048-step recipe,
frozen before evaluation, with old TRAIN rehearsal plus balanced diverse
new TRAIN pairs. A TRAIN-only guard gradient is computed every step;
when the new task gradient opposes it, remove that negative projection
before updating physical FP16 K32xV8 dictionaries. Periodic physical
5-bit swaps are accepted only if they improve new TRAIN loss without
increasing the protected TRAIN sample loss. No FP32 teacher, no QA use
in updates, no sidecar at inference, no LIVE or GitHub main updates.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.optim import propose_codes,apply_codes
from v185.codec import replace_discrete
from v186.pairs import build_pairs, tokenize_one
from v200.rollout import probe_greedy_train
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.diverse_task_replay import make_new_pairs as s3_pairs
from calibri2.s6.train_29m import new_pairs as s4_pairs
from calibri2.s8.broad_replay import (
    SOURCE_SHA, SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC,
    PAID_OPERATORS, freeze_language_corpus, weighted_pair_loss,
    SENTENCE_CONTRASTS,
)
from v174.run import save

SEED=2110
STEPS=2048
LR=.0011
GUARD_COEFF=.35
MAX_INDEX_PROPOSALS=512
SCHEDULE_EARLY=("color","color","numeric","copy")
SCHEDULE_LATE=("color","numeric","copy","color","copy","numeric","copy",
               "color","sentence","language")
STAGE_TRANSITION=768
# Do not use words or prompt templates from the new frozen S11 evaluation.
ADJ_A=("amber","soft","faint","glowing","pale","quiet","bright","silver",
       "small","distant","sturdy","smooth","plain","fresh","early","green")
NOUN_A=("lantern","meadow","tunnel","valley","notebook","basket","harbor",
        "castle","river","garden","window","planet","engine","mountain",
        "bridge","flower")
ADJ_B=("warm","golden","wooden","cool","gentle","shining","dark","open",
       "silent","tiny","wide","clear","hidden","brave","purple","winter")
NOUN_B=("forest","village","bicycle","ribbon","station","ocean","island",
        "signal","pillow","beacon","sailor","rabbit","orchard","sunset",
        "basketball","rainfall")
COLORS=("red","blue","green","yellow")
TEMPLATES_COLOR=(
 "A work note reads: {a} stored the {ca} box; {b} stored the {cb} box. Which color box did {a} store? Reply with one color.",
 "An inventory states that {a} owns a {ca} folder and {b} owns a {cb} folder. State the folder color for {a}.",
 "During class, {a} received a {ca} card while {b} received a {cb} card. Give the color of {a}'s card.",
 "According to the record, the {ca} pen belongs to {a}; the {cb} pen belongs to {b}. What color pen belongs to {a}?",
 "Two people have objects: {a} has the {ca} cup and {b} has the {cb} cup. Return only the color owned by {a}.",
 "A list associates {a} with a {ca} lamp and {b} with a {cb} lamp. Answer with the color of the lamp for {a}.",
 "At the campsite, {a} brought a {ca} tent and {b} brought a {cb} tent. Which color tent was brought by {a}?",
 "A report says {a} has a {ca} coat, whereas {b} has a {cb} coat. What is the color of {a}'s coat?",
)
TEMPLATES_COPY=(
 "Read the phrase '{phrase}' from training sheet {i}; reproduce it exactly without explanation.",
 "Training card {i}: '{phrase}'. Answer by repeating only the words inside quotation marks.",
 "On a paper is '{phrase}'. Type that exact phrase as your response to exercise {i}.",
 "Practice exercise {i} asks you to copy '{phrase}' exactly and then stop.",
 "From the practice line '{phrase}', repeat the quoted words, including their order.",
 "A classroom exercise {i} displays '{phrase}'. Respond with those same words.",
 "Repeat, word for word, what follows in this example {i}: '{phrase}'.",
 "In this sample {i} the phrase is '{phrase}'. Return only that phrase.",
)
TEMPLATES_NUMERIC=(
 "For practice problem {i}, is {a} larger than {b}? Say Yes or No.",
 "In a numbers exercise {i}, compare {a} with {b}. Is the first value greater? Reply Yes or No.",
 "Practice sample {i}: does {a} exceed {b}? Give Yes or No.",
 "Take the two numbers {a} and {b}. Is the left one bigger? Return Yes or No. Item {i}.",
 "Exercise {i}: Is {a} strictly greater than {b}? One word, Yes or No.",
 "In example {i}, check whether {a} is greater than {b}. Respond Yes or No.",
)
TRAIN_PER_DOMAIN={"color":256,"numeric":128,"copy":256,"sentence":64}
PROBE_MAX=12


def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def new_train_pairs(tok):
    data={k:[] for k in TRAIN_PER_DOMAIN}
    index=1000
    def add(domain,a,b):
        nonlocal index
        left=tokenize_one(tok,*a)
        right=tokenize_one(tok,*b)
        if left["first_expected_id"]==right["first_expected_id"]:
            raise AssertionError(f"S11 {domain}: both TRAIN sides same first token")
        row={"split":"TRAIN","domain":"s11_"+domain,"pair_id":index,
             "left":left,"right":right}
        data[domain].append(row)
        index+=1

    for i in range(TRAIN_PER_DOMAIN["color"]):
        a=f"AriPractice{i:04d}";b=f"BelaPractice{i:04d}"
        col=COLORS[(i//2)%len(COLORS)]
        opposing=COLORS[((i//2)+1+(i//13)%2)%len(COLORS)]
        if col==opposing:
            raise AssertionError("Color TRAIN labels identical")
        template=TEMPLATES_COLOR[(i//3)%len(TEMPLATES_COLOR)]
        add("color",
            (template.format(a=a,b=b,ca=col,cb=opposing), col),
            (template.format(a=a,b=b,ca=opposing,cb=col), opposing))

    for i in range(TRAIN_PER_DOMAIN["numeric"]):
        base=11000+i*19
        diff=(i%23)+3
        t=TEMPLATES_NUMERIC[i%len(TEMPLATES_NUMERIC)]
        add("numeric",
            (t.format(i=i,a=base+diff,b=base),"Yes"),
            (t.format(i=i,a=base,b=base+diff),"No"))

    for i in range(TRAIN_PER_DOMAIN["copy"]):
        a=ADJ_A[i%16]+" "+NOUN_A[(i*7+i//16)%16]
        b=ADJ_B[i%16]+" "+NOUN_B[(i*5+i//16)%16]
        if i%3==0:
            a+=" softly";b+=" slowly"
        t=TEMPLATES_COPY[(i//4)%len(TEMPLATES_COPY)]
        add("copy",(t.format(phrase=a,i=i),a),(t.format(phrase=b,i=i),b))

    for i in range(TRAIN_PER_DOMAIN["sentence"]):
        a,b,c,d=SENTENCE_CONTRASTS[i%len(SENTENCE_CONTRASTS)]
        context=f"Training example {i}: "
        add("sentence",
            (context+a+" Answer in a sentence.",b),
            (context+c+" Answer in a sentence.",d))

    if any(len(data[k])!=count for k,count in TRAIN_PER_DOMAIN.items()):
        raise AssertionError("S11 new TRAIN set incomplete")
    allpairs=[p for val in data.values() for p in val]
    prompts=[p[side]["question"] for p in allpairs for side in ("left","right")]
    if len(set(prompts))!=len(prompts):
        raise AssertionError("Repeated TRAIN question")
    return data


def training_probe(model,sources,pairs):
    outcomes=probe_greedy_train(model,sources,pairs,max_new_tokens=PROBE_MAX)
    result=[]
    for p in pairs:
        for side in ("left","right"):
            sample=outcomes[p["pair_id"]]["sides"][side]
            result.append({
              "pair_id":p["pair_id"],"domain":p["domain"],"side":side,
              "gold":p[side]["answer"],"actual_generated_ids":sample["generated_tokens"],
              "correct_full_EOS":bool(sample["exact_with_EOS"])})
    return result


def loss_of(model, pair, language_tokens):
    if pair is not None:
        return weighted_pair_loss(model,pair)
    scores=model(input_ids=language_tokens[None],use_cache=False).logits[0]
    return F.cross_entropy(scores[:-1],language_tokens[1:])


def take_paid_gradient(model,sources,keys,pair=None,tokens=None,target_name=None):
    """Return exact paid-codebook gradients and optional target 2D gradient.

    Sources are re-built each forward and cached to save dense matrix lifetime.
    Gradients here are for TRAIN pairs, never heldout. The indexed CB59 2D
    operator gradient is consumed for protected, paid 5-bit code proposals.
    """
    clear(sources)
    model.zero_grad(set_to_none=True)
    captured=[]
    holder=None
    if target_name is not None:
        def retain(_mod,_args,weight):
            weight.retain_grad()
            captured.append(weight)
        holder=sources[target_name].register_forward_hook(retain)
    try:
        value=loss_of(model,pair,tokens)
        if not torch.isfinite(value).all():
            raise AssertionError("TRAIN objective not finite")
        result=float(value.detach())
        value.backward()
    finally:
        if holder is not None:
            holder.remove()
    gradients=[]
    for key in keys:
        g=sources[key].centroids.grad
        if g is None or not torch.isfinite(g).all():
            raise AssertionError(f"TRAIN paid centroid has no gradient: {key}")
        gradients.append(g.detach().clone())
    matrix=None
    if target_name is not None:
        good=[a.grad for a in captured if a.grad is not None]
        if not good:
            raise AssertionError("Targeted packed 2D source has no TRAIN gradient")
        matrix=torch.stack(good).sum(dim=0).detach().float().cpu().numpy().copy()
    clear(sources)
    return result,gradients,matrix


def merge_protected(primary,guard,params):
    dot=sum(torch.sum(x.double()*y.double()) for x,y in zip(primary,guard))
    anchor_sq=sum(torch.sum(g.double().square()) for g in guard)
    if anchor_sq<=0 or not torch.isfinite(anchor_sq):
        raise AssertionError("Cannot protect a zero/nonfinite guard gradient")
    projected=dot < 0
    correction=float(dot/anchor_sq) if projected else 0.
    merged=[a-correction*g + GUARD_COEFF*g
            for a,g in zip(primary,guard)]
    for p,g in zip(params,merged):
        p.grad=g.to(dtype=p.dtype).clone()
    final_dot=sum(torch.sum(g.double()*q.double())
                  for g,q in zip(merged,guard))
    if final_dot <= -1e-10:
        raise AssertionError("TRAIN safeguard failed first-order guard")
    return float(dot),float(final_dot),bool(projected)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    setup()
    raw=args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA or len(raw)!=SOURCE_BYTES:
        raise AssertionError("S11 only starts from sealed S7, not QA-selected S8")
    cb=CB59(raw)
    if cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S11 physical architecture drift")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    first=build_pairs(tok,"TRAIN")
    second=s3_pairs(tok)
    third=s4_pairs(tok)
    assert [len(first),len(second),len(third)]==[24,64,56]
    replay=first+second+third
    fresh=new_train_pairs(tok)
    training_questions=[p[s]["question"] for p in replay+
                        [q for group in fresh.values() for q in group]
                        for s in ("left","right")]
    if len(training_questions)!=len(set(training_questions)):
        raise AssertionError("Old/new TRAIN overlap")
    # 24 original pairs include numeric/yes-no/color; S3 and S4 add older
    # color and multi-token TRAIN to protect across changed task distribution.
    color_old=[r for r in second if r["domain"] in ("s3_folder_training","s3_compare_training")]
    copy_old=[r for r in third if "copy" in r["domain"]]
    guards=first[:16]+color_old[:8]+copy_old[:8]
    if len(guards)!=32:
        raise AssertionError("Missing predeclared 32 TRAIN-only replay guards")
    probe_pairs=first[:8]+color_old[:4]+copy_old[:4]
    if len(probe_pairs)!=16:
        raise AssertionError("S11 TRAIN probe changed")
    corpus,corpus_meta=freeze_language_corpus(tok)
    model,sources=create_small_paid_model(cb,build_config(MODEL_SPEC))
    keys=sorted(sources)
    if len(keys)!=PAID_OPERATORS:
        raise AssertionError("Not exactly 85 physically paid codebooks")
    start_probe=training_probe(model,sources,probe_pairs)
    params=[sources[k].centroids for k in keys]
    optimizer=torch.optim.Adam(params,lr=LR)
    counts=Counter()
    trace=[]
    projection_events=0
    accepted_codes=0
    attempted_codes=0
    for step in range(STEPS):
        plan=SCHEDULE_EARLY if step<STAGE_TRANSITION else SCHEDULE_LATE
        kind=plan[(step if step<STAGE_TRANSITION else step-STAGE_TRANSITION)%len(plan)]
        index=counts[kind]
        counts[kind]+=1
        guard=guards[step%len(guards)]
        if kind=="language":
            i=(index%(len(corpus)//48))*48
            train_pair=None
            text_ids=corpus[i:i+48]
        else:
            group=fresh[kind]
            train_pair=group[index%len(group)]
            text_ids=None
        target_name=keys[(step//8)%len(keys)]
        guard_loss,guard_grads,_=take_paid_gradient(
            model,sources,keys,pair=guard)
        first_loss,task_grads,weight_grad=take_paid_gradient(
            model,sources,keys,pair=train_pair,
            tokens=text_ids,
            target_name=target_name if (step+1)%8==0 else None)
        before,after,projected=merge_protected(task_grads,guard_grads,params)
        projection_events+=int(projected)
        norm=float(torch.nn.utils.clip_grad_norm_(
            params,1.0,error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for p in params:
                p.clamp_(-16,16)
        clear(sources)
        changed=0
        rejected_by_guard=0
        if (step+1)%8==0:
            source=sources[target_name]
            with torch.inference_mode():
                current_task=float(loss_of(model,train_pair,text_ids))
                clear(sources)
                current_guard=float(loss_of(model,guard,None))
                clear(sources)
                matrix=source().detach().cpu().numpy().copy()
            proposal=propose_codes(source,weight_grad,matrix)
            if proposal["proposed_groups"]>MAX_INDEX_PROPOSALS:
                raise AssertionError("S11 unexpected packed code proposal size")
            attempted_codes+=proposal["proposed_groups"]
            if proposal["proposed_groups"]:
                with torch.no_grad():
                    apply_codes(source,proposal,True)
                clear(sources)
                with torch.inference_mode():
                    next_task=float(loss_of(model,train_pair,text_ids))
                    clear(sources)
                    next_guard=float(loss_of(model,guard,None))
                if (np.isfinite(next_task) and np.isfinite(next_guard)
                        and next_task+1e-7<current_task
                        and next_guard <=current_guard+1e-4):
                    changed=proposal["proposed_groups"]
                    accepted_codes+=changed
                else:
                    rejected_by_guard=int(next_guard>current_guard+1e-4)
                    with torch.no_grad():
                        apply_codes(source,proposal,False)
                clear(sources)
        trace.append({"step":step+1,"task":kind,
                      "train_pair_id":None if train_pair is None else train_pair["pair_id"],
                      "guard_pair_id":guard["pair_id"],
                      "new_TASK_loss":first_loss,"rehearsal_TRAIN_loss":guard_loss,
                      "guard_task_gradient_dot_before":before,
                      "guard_task_gradient_dot_after":after,
                      "gradient_projection_applied":projected,
                      "clipped_gradient_norm":norm,
                      "index_updates_accepted":changed,
                      "index_candidate_rejected_on_guard":rejected_by_guard})
        if (step+1)%256==0:
            print(json.dumps({"event":"S11_PROTECTED_TRAIN_PROGRESS",
                "step":step+1,"task":kind,
                "projections":projection_events,"accepted_code_indices":accepted_codes,
                "guard_rejected_index_proposals":sum(r["index_candidate_rejected_on_guard"]
                                                     for r in trace),
                "schedule_counts":dict(counts),
                "task_loss":first_loss,"guard_loss":guard_loss}),flush=True)
    packed,phys=replace_discrete(cb,sources,strict=False)
    final=args.out/"CALIBRI2_S11_36M_PROTECTED_2048steps_paid.cb59"
    final.write_bytes(packed)  # save before *all* diagnostics, never lose real weights
    sha=hashlib.sha256(packed).hexdigest()
    if len(packed)!=SOURCE_BYTES or phys["complete_bpp"]>.700 or sha==SOURCE_SHA:
        raise AssertionError("S11 physically paid model invalid or unchanged")
    reload_model,reload_sources=create_small_paid_model(
        CB59(final.read_bytes()),build_config(MODEL_SPEC))
    a=training_probe(model,sources,probe_pairs)
    b=training_probe(reload_model,reload_sources,probe_pairs)
    if a!=b:
        raise AssertionError("S11 physical CB59 reload changed actual TRAIN sequences")
    lost=sum(x["correct_full_EOS"] and not y["correct_full_EOS"]
             for x,y in zip(start_probe,b))
    gained=sum(not x["correct_full_EOS"] and y["correct_full_EOS"]
               for x,y in zip(start_probe,b))
    report={
        "stage":"S11_PROTECTED_36M_ACTUAL_TRAIN",
        "source_SHA256":SOURCE_SHA,"trained_SHA256":sha,
        "source_parameters":SOURCE_PARAMS,
        "full_saved_file_bytes":len(packed),
        "full_physical_bpp":phys["complete_bpp"],
        "all_85_paid_dictionary_cells_and_indices_only":True,
        "total_2048_steps":STEPS,"new_LR":LR,"guard_coeff":GUARD_COEFF,
        "predeclared_768_then_1280_curriculum":True,
        "new_TRAIN_pair_counts":TRAIN_PER_DOMAIN,
        "old_TRAIN_rehearsal_pairs":len(replay),
        "protected_TRAIN_guard_pairs":len(guards),
        "public_text_TRAIN_provenance":corpus_meta,
        "gradients_projected_away_from_guard":projection_events,
        "gradient_projection_numerical_first_order_checks":True,
        "accepted_physically_paid_5bit_groups":accepted_codes,
        "proposed_physically_paid_5bit_groups":attempted_codes,
        "index_swaps_guard_rejections":sum(x["index_candidate_rejected_on_guard"] for x in trace),
        "TRAIN_original_correct":sum(x["correct_full_EOS"] for x in start_probe),
        "TRAIN_final_correct":sum(x["correct_full_EOS"] for x in b),
        "TRAIN_previous_correct_lost":lost,
        "TRAIN_new_correct_gained":gained,
        "preservation_gate_PASS":(lost==0),
        "all_baseline_TRAIN_answers":start_probe,
        "all_final_TRAIN_answers":b,
        "all_paid_reloaded_full_tokens_equal":True,
        "no_QA_VALID_TEST_for_train_or_checkpoint_selection":True,
        "no_FP32_teacher_original_weights":True,
        "no_inference_sidecar":True,
        "quality_jump_NOT_yet_proven":True,
        "full_step_trace":trace,
    }
    save(args.out/"ACTUAL_CALIBRI2_S11_PROTECTED_TRAIN.json",report)
    print(json.dumps({"event":"S11_PAID_CHECKPOINT_SEALED",
        "sha256":sha,"bytes":len(packed),"physical_bpp":phys["complete_bpp"],
        "projections":projection_events,"accepted_index_updates":accepted_codes,
        "previously_correct_TRAIN_lost":lost,
        "TRAIN_before":report["TRAIN_original_correct"],
        "TRAIN_after":report["TRAIN_final_correct"],
        "guard_preservation_pass":report["preservation_gate_PASS"]}),flush=True)


if __name__=="__main__":
    main()
