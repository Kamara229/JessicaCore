"""S30: same-architecture *trainable DENSE FP32* 36M capability probe.

Purpose: distinguish a lack of FP32 learning capacity from the restrictions of
the physically paid compressed Green Tunnel updates. Start from EXACT same
sealed S26 CB59 reconstructed weights, not an unrelated pretrained 135M.
Verify pretraining full next-token logits against the compact 36M model.

Learn only a small, separately generated set of S30 TRAIN prompts containing
the eight training-eligible combinations of four new nouns. No S29 QA import,
peek, gradient, stopping rule, model selection, early stopping or scoring;
S29 QA was SHA-frozen in a different earlier PR/CI. This is ONLY a TRAIN
capacity experiment; it cannot claim generalization or 0.7 bpp preservation
because weights are now fully dense 32-bit in RAM.

No original FP32 teacher, no public pretrained weights, no LIVE Jessica.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59,unpack5
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from v186.pairs import tokenize_one,build_pairs
from v176.operators import clear
from v174.run import save

SOURCE_SHA="eaa71093d89cd658a1e9d7a7e5297586f2d4549778d1b9bfae3cf54c26ce502c"
STEPS=192
LR=0.00015
NEW_PAIRS=32
SCHEDULE=("new","new","old","new")
TRAIN_ELIGIBLE={
  "red":("bowl","drum"),
  "green":("bowl","rope"),
  "blue":("ring","rope"),
  "yellow":("ring","drum"),
}
assert set(TRAIN_ELIGIBLE)=={"red","green","blue","yellow"}
assert len(set((color,n) for color,nouns in TRAIN_ELIGIBLE.items() for n in nouns))==8
S29_FROZEN_SHA="812a8165f50945e57e126bd02b7dca7fec52fd46f13202d4bf26ae18d34e7223"


def dense_from_paid(cb):
    """Convert exactly the paid same-architecture 36M model to independent FP32."""
    cfg=build_config(MODEL_SPEC)
    model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).cpu().eval()
    if set(model.state_dict())!=set(cb.meta["full_original_state_keys"]):
        raise AssertionError("Not SAME 36M architecture / tied keys")
    if sum(p.numel() for p in model.parameters())!=SOURCE_PARAMS:
        raise AssertionError("Not unique 36M parameters")
    if model.model.embed_tokens.weight.data_ptr()!=model.lm_head.weight.data_ptr():
        raise AssertionError("Lost tied embeddings")
    with torch.no_grad():
        for name,dest in model.state_dict().items():
            if name in cb.handles:
                h=cb.handles[name]
                m,n=h.shape
                ix=unpack5(h.packed.tobytes(),m*n//8)
                w=h.centroids[ix].reshape(m,n)*h.row_scales[:,None]
            else:
                w=cb.raw[name]
            if dest.shape!=w.shape or not np.isfinite(w).all():
                raise AssertionError("S30 source weights incomplete/nonfinite")
            dest.copy_(torch.from_numpy(np.array(w,copy=True)).float())
    return model


def make_training_pairs(tok):
    """S30 TRAIN only; withhold 8 other combinations, never import S29 QA."""
    patterns=(
      "TRAIN item {i}: {a} holds a {left}; {b} holds a {right}. State {a}'s color and object in two words.",
      "TRAIN sample {i}: In a storeroom {a} receives the {left}, while {b} receives the {right}. Name {a}'s object.",
      "TRAIN card {i}: {b} = {right}; {a} = {left}. What belongs to {a}? Give the two-word item.",
      "TRAIN report {i}: {a} chose {left}. Someone called {b} chose {right}. Reply with {a}'s two-word choice.",
    )
    rows=[]
    for ci,(color,nouns) in enumerate(TRAIN_ELIGIBLE.items()):
        first=f"{color} {nouns[0]}"
        second=f"{color} {nouns[1]}"
        for i in range(8):
            q=patterns[(i+ci)%len(patterns)]
            owner=f"FPTrainL{ci:01d}{i:02d}"
            other=f"FPTrainR{ci:01d}{i:02d}"
            left=tokenize_one(tok,q.format(i=i,a=owner,b=other,left=first,right=second),first)
            right=tokenize_one(tok,q.format(i=i,a=owner,b=other,left=second,right=first),second)
            if left["first_expected_id"]!=right["first_expected_id"]:
                raise AssertionError("Contrasting nouns must share the SAME first color token")
            if first.split()[1]==second.split()[1]:
                raise AssertionError("No noun contrast")
            rows.append({"pair_id":5000+len(rows),"kind":"new",
                         "color":color,"left":left,"right":right})
    if len(rows)!=NEW_PAIRS or len({r[s]["question"] for r in rows for s in ("left","right")})!=64:
        raise AssertionError("TRAIN pair size/uniqueness wrong")
    return rows


def double_check_original_logits(cb,dense,tok):
    paid,sources=create_small_paid_model(cb,build_config(MODEL_SPEC))
    sample="TRAIN capacity preflight: One item is a red bowl. What color is it?"
    prompt=tok.apply_chat_template(
        [{"role":"user","content":sample}],tokenize=False,add_generation_prompt=True)
    ids=torch.tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)[None]
    with torch.inference_mode():
        gold=paid(input_ids=ids,use_cache=False).logits[0,-1]
        actual=dense(input_ids=ids,use_cache=False).logits[0,-1]
        maximum=float(torch.max(torch.abs(gold-actual)))
        both_equal=int(gold.argmax())==int(actual.argmax())
    clear(sources)
    del paid,sources
    if maximum>0.001 or not both_equal:
        raise AssertionError(f"Dense 36M initial logits not matched to compressed source: {maximum}")
    return maximum


def train_loss(model,pair,boost_noun):
    items=[]
    for side in ("left","right"):
        r=pair[side]
        logits=model(input_ids=r["ids"][None],use_cache=False).logits[0][r["positions"]]
        ce=F.cross_entropy(logits,r["targets"],reduction="none")
        if boost_noun:
            prefix=r["answer"].split()[0]
            ncolor=len(tok_encode_color(prefix))
            if ncolor>=len(r["targets"])-1:raise AssertionError("No noun token")
            weights=torch.ones_like(ce)
            weights[ncolor:-1]=4.
            items.append((ce*weights).sum()/weights.sum())
        else:
            items.append(ce.mean())
    return sum(items)/len(items)


# A tiny globally set tokenizer is sufficient for this isolated one-process CI
# experiment; the TRAIN-only labels never reference S29 QA.
TOK=None

def tok_encode_color(color):
    if TOK is None:raise AssertionError("Tokenizer not initialized")
    return TOK.encode(color,add_special_tokens=False)


@torch.inference_mode()
def greedy_actual(model,tok,question,max_new=10):
    prompt=tok.apply_chat_template(
        [{"role":"user","content":question}],tokenize=False,add_generation_prompt=True)
    ids=torch.tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)[None]
    result=[]
    for _ in range(max_new):
        logits=model(input_ids=ids,use_cache=False).logits[0,-1]
        token=int(logits.argmax())
        result.append(token)
        if token==tok.eos_token_id:break
        ids=torch.cat((ids,torch.tensor([[token]],dtype=torch.long)),dim=1)
    return result


def train_probe(model,tok,new_pairs,old_pairs):
    # PURELY TRAIN prompts. A correct forced prefix or keyword is not a hit.
    sample_new=new_pairs[::4]
    sample_old=old_pairs[:4]
    outcomes=[]
    for kind,pairs in (("new_noun_TRAIN",sample_new),("historical_TRAIN",sample_old)):
        for pair in pairs:
            for side in ("left","right"):
                q=pair[side]
                predicted=greedy_actual(model,tok,q["question"])
                gold=tok.encode(q["answer"],add_special_tokens=False)+[tok.eos_token_id]
                outcomes.append({
                    "kind":kind,"id":f'{pair["pair_id"]}_{side}',
                    "expected_full_tokens_EOS":gold,
                    "actual_full_tokens":predicted,
                    "actual_text":tok.decode(predicted,skip_special_tokens=True),
                    "strict_correct_full_EOS":predicted==gold,
                    "correct_first_token":bool(predicted) and predicted[0]==gold[0]
                })
    return {"answers":outcomes,"per_kind":{
        kind:{"n":sum(x["kind"]==kind for x in outcomes),
              "full_correct":sum(x["kind"]==kind and x["strict_correct_full_EOS"] for x in outcomes),
              "first_token_only":sum(x["kind"]==kind and x["correct_first_token"] for x in outcomes)}
        for kind in ("new_noun_TRAIN","historical_TRAIN")}}


def snapshot_sha(model):
    checksum=hashlib.sha256()
    for name,t in model.named_parameters():
        checksum.update(name.encode("utf-8"))
        checksum.update(t.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(3030)
    torch.set_num_threads(2)
    raw=args.source.read_bytes()
    if len(raw)!=SOURCE_BYTES or hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise AssertionError("S30 matched FP32 starts ONLY from immutable S26")
    cb=CB59(raw)
    if cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S30 36M physical source wrong")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    global TOK
    TOK=tok
    new_pairs=make_training_pairs(tok)
    old_pairs=build_pairs(tok,"TRAIN")
    if len(old_pairs)!=24:raise AssertionError("Historic TRAIN guard missing")
    if {r[s]["question"] for r in new_pairs for s in ("left","right")} & \
       {r[s]["question"] for r in old_pairs for s in ("left","right")}:
        raise AssertionError("TRAIN question duplication")
    dense=dense_from_paid(cb)
    parity=double_check_original_logits(cb,dense,tok)
    before_sha=snapshot_sha(dense)
    before=train_probe(dense,tok,new_pairs,old_pairs)
    optimizer=torch.optim.AdamW(dense.parameters(),lr=LR,weight_decay=0.0)
    gradients_seen=0
    tasks=Counter()
    trace=[]
    for step in range(STEPS):
        kind=SCHEDULE[step%len(SCHEDULE)]
        index=tasks[kind]
        tasks[kind]+=1
        if kind=="new":
            pair=new_pairs[index%len(new_pairs)]
            boost=True
        else:
            pair=old_pairs[index%len(old_pairs)]
            boost=False
        optimizer.zero_grad(set_to_none=True)
        objective=train_loss(dense,pair,boost)
        if not torch.isfinite(objective):raise AssertionError("Nonfinite TRAIN objective")
        before_step=float(objective.detach())
        objective.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(
            dense.parameters(),1.0,error_if_nonfinite=True))
        if norm>0:gradients_seen+=1
        optimizer.step()
        if step==0 or (step+1)%24==0:
            current=float(train_loss(dense,pair,boost).detach())
            trace.append({"step":step+1,"kind":kind,
                          "TRAIN_pair_id":pair["pair_id"],
                          "TRAIN_loss_before":before_step,
                          "TRAIN_loss_after":current,
                          "full_dense_grad_norm":norm})
            print(json.dumps({"event":"S30_DENSE_FP32_TRAIN",
                              **trace[-1]},ensure_ascii=False),flush=True)
    after=train_probe(dense,tok,new_pairs,old_pairs)
    after_sha=snapshot_sha(dense)
    if before_sha==after_sha or gradients_seen!=STEPS:
        raise AssertionError("No authentic full FP32 training happened")
    report={
        "stage":"S30_MATCHED_36M_DENSE_FP32_TRAIN_CAPACITY_ONLY",
        "source_S26_5bit_SHA256":SOURCE_SHA,
        "source_S26_complete_bpp":8*len(raw)/SOURCE_PARAMS,
        "initial_paid_vs_dense_logits_max_abs":parity,
        "same_36M_architecture_and_starting_decompressed_weights":True,
        "original_unique_parameters":SOURCE_PARAMS,
        "FP32_dense_payload_bytes":4*SOURCE_PARAMS,
        "dense_FP32_is_NOT_0_7bpp":True,
        "not_an_original_pretrained_FP32_reference":True,
        "S29_QA_immutable_sha_frozen_prior_to_this_source":S29_FROZEN_SHA,
        "S29_QA_never_imported_or_scored":True,
        "S26_inspected_QA_not_imported_or_used_for_training":True,
        "all_train_only_questions":True,
        "steps":STEPS,"learning_rate":LR,
        "pre_training_FP32_SHA256":before_sha,
        "post_training_FP32_SHA256":after_sha,
        "all_steps_real_full_gradient":gradients_seen,
        "train_scheduled_task_counts":dict(tasks),
        "TRAIN_results_before":before,
        "TRAIN_results_after":after,
        "periodic_TRAIN_loss_trace":trace,
        "no_training_quality_parity_claim_without_future_frozen_QA":True,
    }
    save(args.out/"ACTUAL_S30_MATCHED_36M_DENSE_FP32_TRAIN_CAPACITY.json",report)
    print(json.dumps({
        "stage":report["stage"],
        "steps":STEPS,
        "fp32_full_weights_updated":before_sha!=after_sha,
        "parity_max_abs":parity,
        "train_before":before["per_kind"],
        "train_after":after["per_kind"],
        "S29_QA_not_touched":True},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
