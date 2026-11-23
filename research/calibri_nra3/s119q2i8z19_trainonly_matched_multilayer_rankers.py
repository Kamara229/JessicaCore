"""S119q2i8z19: matched 1,280-param ORIGINAL S142 native top64 rerankers.

Same pretrained original43M, same source Z18 28 TRAIN, same frozen
native49k logits, same original native top64 and token embeddings,
same TRAIN gold conditional eligibility and same 25-epoch fixed AdamW.
Only variable: final ORIGINAL LM head h320 vs pre-registered FIXED
3 original 12-layer intermediates (L3/L7/L11) + final h320 mean.
Each candidate scorer is a SEPARATE compact shadow, not a new
region/weights of original CALIBRI. DEV is NEVER part of training.
"""
from __future__ import annotations
import torch
from research.calibri_nra3.s119q2i8z16_trainonly_lowrank_native64_reranker import (
    NativeBPE2CandidateReranker,
    VOCAB,HIDDEN,K,RANK,TRAIN_EPOCHS,TRAIN_ADAMW_LR,TRAIN_ADAMW_WD,
)
from research.calibri_nra3.s119q2i8z18_fresh28train14dev_multilayer_source import (
    BASELINE,ALTERNATE,
)

MODES=(BASELINE,ALTERNATE)
HIDDEN_KEYS={BASELINE:"hidden_final",ALTERNATE:"hidden_multilayer"}
SOURCE_PREFIX="S119Q2I8Z18_"

def check(ok,reason):
    if not ok:raise ValueError("S119q2i8z19 protected matched TRAIN-only protocol: "+reason)


def train_matched_shadow_rerankers(records,original_native_embedding):
    check(isinstance(records,(tuple,list)) and len(records)==28
          and len({r["source_id"] for r in records})==28
          and all(r["split"]=="TRAIN" and
                  r["source_id"].startswith(SOURCE_PREFIX) and
                  type(r["target_native_BPE2"]) is int and
                  0<=r["target_native_BPE2"]<VOCAB and
                  type(r["own_first_correct"]) is bool for r in records),
          "exact Z18 28 new human TRAIN only, no Z16 or DEV labels")
    check(isinstance(original_native_embedding,torch.Tensor)
          and original_native_embedding.shape==(VOCAB,HIDDEN)
          and not original_native_embedding.requires_grad,
          "original S142 full native embeddings strictly frozen")
    emb=original_native_embedding.detach().float().clone()
    eligible=[]
    for r in records:
        check(r["logits"].shape==(1,VOCAB)
              and all(r[k].shape==(1,HIDDEN) and not r[k].requires_grad
                      for k in HIDDEN_KEYS.values()),
              "same real original43M native logits/h320 and mixed h320 cached once")
        ids=torch.topk(r["logits"][0].detach(),K).indices
        eligible.append(bool(r["own_first_correct"] and
                             (ids==r["target_native_BPE2"]).any()))
    check(sum(eligible)>=2,"cannot train lowrank2 without 2+ real original TRAIN BPE2 candidates")
    models={}
    statistics={}
    for mode in MODES:
        model=NativeBPE2CandidateReranker()
        opt=torch.optim.AdamW(model.parameters(),lr=TRAIN_ADAMW_LR,
                              weight_decay=TRAIN_ADAMW_WD)
        for epoch in range(TRAIN_EPOCHS):
            for r,use in zip(records,eligible):
                if not use:continue
                opt.zero_grad(set_to_none=True)
                loss,meta=model.trainonly_loss(
                    r["logits"],r[HIDDEN_KEYS[mode]],emb,
                    r["target_native_BPE2"],split="TRAIN",
                    own_first_correct=r["own_first_correct"])
                check(meta["eligible_for_gradient"] and
                      torch.isfinite(loss) and loss.requires_grad,
                      "only source Z18 TRAIN conditional gold within immutable original top64")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
        check(sum(x.numel() for x in model.parameters())==1280 and
              all(torch.isfinite(x).all() for x in model.parameters()) and
              model.output_gate==0 and not model.candidate_accepted,
              "exact 1280 cheap trained shadow weights unaccepted")
        model.eval()
        models[mode]=model
        statistics[mode]={
            "only_human_Z18_28_TRAIN":True,
            "new_TRAIN_examples":28,
            "eligible_correct_own_first_and_gold_in_original_native_top64":
                sum(eligible),
            "TRAIN_epochs_fixed_before_DEV":TRAIN_EPOCHS,
            "AdamW_lr":TRAIN_ADAMW_LR,
            "AdamW_weight_decay":TRAIN_ADAMW_WD,
            "trainable_params":1280,
            "original_S142_43m_optimizer_steps":0,
            "mode":mode,
            "candidate_accepted":False,"output_gate":0,
        }
    check(set(models)==set(HIDDEN_KEYS) and
          statistics[BASELINE]["eligible_correct_own_first_and_gold_in_original_native_top64"]==
          statistics[ALTERNATE]["eligible_correct_own_first_and_gold_in_original_native_top64"],
          "exact same eligible TRAIN source for final and fixed multilayer")
    return models,statistics
