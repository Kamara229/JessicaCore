"""S119q2i8z21: compact future BPE2/BPE3/BPE4 embedding state, not semantic proof.

TRAIN-only native original43M h320 -> latent16 -> predicted future h320,
trained against mean of frozen true ORIGINAL native embedding rows of
human next BPE2..BPE4, with matched SHUFFLED-label negative control.
Does not use gold in DEV projection/reranking, has no gradient path to
original43M/GT12/Core/NRA3/NPA1/native embedding/logits. Candidate
top64 is taken from ORIGINAL native49152 logits before prediction.
A shadow of 10240 params, not an original neural area. EOS unmasked.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z21_fresh28train14dev_future_state_source import (
    TOP_K,LATENT_DIM,PREDICTED_FUTURE_POSITIONS,
    FIXED_LOGIT_COSINE_SCALE,TRAIN_EPOCHS,TRAIN_LR,
    TRAIN_WD,PERMUTED_TRAIN_SEED,
)

HIDDEN=320
VOCAB=49152
EOS_ID=2
SHADOW_INIT_SEED=119122
STATES=("ALIGNED_NEXT3_NATIVE_EMBEDDINGS","PERMUTED_NEXT3_NATIVE_EMBEDDINGS")

def check(ok,reason):
    if not ok:
        raise ValueError("Z21 predictive state fail closed: "+reason)


def target_next3_original_embeddings(original_native_embed,target_ids):
    check(isinstance(original_native_embed,torch.Tensor)
          and original_native_embed.shape==(VOCAB,HIDDEN)
          and not original_native_embed.requires_grad
          and isinstance(target_ids,(list,tuple))
          and len(target_ids)>=max(PREDICTED_FUTURE_POSITIONS)+1
          and all(type(x) is int and 0<=x<VOCAB for x in target_ids),
          "TRAIN-only original native gold BPE2..BPE4 and frozen embeddings")
    ids=torch.tensor([target_ids[i] for i in PREDICTED_FUTURE_POSITIONS],
                     dtype=torch.long)
    with torch.no_grad():
        mean=original_native_embed.detach()[ids].float().mean(dim=0)
        check(mean.shape==(HIDDEN,) and torch.isfinite(mean).all()
              and torch.linalg.vector_norm(mean).item()>0,
              "future original native BPE2..4 embedding mean finite nonzero")
        return F.normalize(mean,dim=0).unsqueeze(0).detach().clone()


class TinyFutureState(nn.Module):
    def __init__(self):
        super().__init__()
        g=torch.Generator(device="cpu").manual_seed(SHADOW_INIT_SEED)
        self.down=nn.Parameter(torch.randn((HIDDEN,LATENT_DIM),generator=g)*.02)
        self.up=nn.Parameter(torch.randn((LATENT_DIM,HIDDEN),generator=g)*.02)
        self.output_gate=0
        self.candidate_accepted=False
        check(sum(x.numel() for x in self.parameters())==HIDDEN*LATENT_DIM*2,
              "fixed 10240-param lowrank future-state shadow only")

    def forward(self,original_last_h320):
        check(self.output_gate==0 and self.candidate_accepted is False
              and isinstance(original_last_h320,torch.Tensor)
              and original_last_h320.shape==(1,HIDDEN)
              and bool(torch.isfinite(original_last_h320).all()),
              "actual original43M final h320 with no original gradient")
        h=F.layer_norm(original_last_h320.detach().float(),(HIDDEN,))
        result=(h@self.down)@self.up
        check(result.shape==(1,HIDDEN)
              and bool(torch.isfinite(result).all()),
              "lowrank16 predicted FUTURE native embedding state only")
        return F.normalize(result,dim=-1,eps=1e-6)

    def predict_future_candidates(self,original_native_logits,
                                  original_last_h320,original_native_embed):
        check(isinstance(original_native_logits,torch.Tensor)
              and original_native_logits.shape==(1,VOCAB)
              and torch.isfinite(original_native_logits).all()
              and isinstance(original_native_embed,torch.Tensor)
              and original_native_embed.shape==(VOCAB,HIDDEN)
              and not original_native_embed.requires_grad,
              "real original49152 BPE logits and native frozen vocab")
        # Top64 is fixed by original logits BEFORE any learned state.
        native=original_native_logits.detach().float()[0]
        ids=torch.topk(native,TOP_K,sorted=True).indices
        candidates=F.normalize(original_native_embed.detach().index_select(
            0,ids).float(),dim=-1,eps=1e-6)
        prediction=self(original_last_h320)
        cosine=(candidates@prediction[0])
        score=native[ids]+FIXED_LOGIT_COSINE_SCALE*cosine
        check(score.shape==(TOP_K,) and
              torch.isfinite(score).all() and
              ids.unique().numel()==TOP_K,
              "target blind native top64 + fixed future similarity, EOS unmasked")
        return ids,score,prediction.detach()

    def train_loss(self,original_last_h320,train_future_target,*,split):
        check(split=="TRAIN" and isinstance(train_future_target,torch.Tensor)
              and train_future_target.shape==(1,HIDDEN)
              and not train_future_target.requires_grad
              and torch.isfinite(train_future_target).all(),
              "only Z21 new TRAIN future native embedding supervision")
        prediction=self(original_last_h320)
        return (1-F.cosine_similarity(prediction,
                                      train_future_target.detach(),dim=-1)).mean()


def deterministic_derangement_indices(n=28):
    check(n==28 and PERMUTED_TRAIN_SEED==119121,
          "fixed 28 TRAIN shuffle only, never DEV")
    g=torch.Generator(device="cpu").manual_seed(PERMUTED_TRAIN_SEED)
    # Derange all TRAIN indices by fixed seeded rotation of permutation.
    order=torch.randperm(n,generator=g).tolist()
    d={order[i]:order[(i+1)%n] for i in range(n)}
    check(len(d)==n and all(i!=d[i] for i in range(n))
          and set(d.values())==set(range(n)),
          "fixed human TRAIN shuffled negative control strict derangement")
    return d


def train_matched_aligned_vs_permuted(train_only,original_native_embed):
    check(isinstance(train_only,(list,tuple)) and len(train_only)==28
          and len({r["source_id"] for r in train_only})==28
          and all(r["split"]=="TRAIN"
                  and r["source_id"].startswith("S119Q2I8Z21_")
                  and type(r["own_first_correct"]) is bool
                  and r["hidden"].shape==(1,HIDDEN)
                  and r["target_future_embedding"].shape==(1,HIDDEN)
                  and r["original_native_logits"].shape==(1,VOCAB)
                  for r in train_only),
          "fixed new Z21 native human 28 TRAIN only, no previous DEV")
    check(original_native_embed.shape==(VOCAB,HIDDEN)
          and not original_native_embed.requires_grad,
          "original native lookup entirely frozen during shadow training")
    derange=deterministic_derangement_indices()
    eligible=[i for i,r in enumerate(train_only) if r["own_first_correct"]]
    check(len(eligible)>=2,
          "at least two original OWN-first correct real 43M source TRAIN rows")
    models={};summary={}
    for name in STATES:
        model=TinyFutureState()
        opt=torch.optim.AdamW(model.parameters(),lr=TRAIN_LR,weight_decay=TRAIN_WD)
        for epoch in range(TRAIN_EPOCHS):
            model.train()
            for i in eligible:
                # The negative control gets SAME number of updates with
                # randomly shuffled human TRAIN-only FUTURE targets.
                j=i if name==STATES[0] else derange[i]
                record=train_only[i]
                target=train_only[j]["target_future_embedding"]
                opt.zero_grad(set_to_none=True)
                loss=model.train_loss(record["hidden"],target,split="TRAIN")
                check(torch.isfinite(loss) and loss.requires_grad,
                      "only TRAIN 10240 shadow params updated")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
        model.eval()
        check(sum(x.numel() for x in model.parameters())==10240
              and model.output_gate==0
              and not model.candidate_accepted
              and all(torch.isfinite(x).all() for x in model.parameters()),
              "new tiny future-state params finite no original updates")
        models[name]=model
        summary[name]={
            "new_Z21_TRAIN_source_total":28,
            "actual_original_OWN_first_correct_TRAIN_eligible":len(eligible),
            "TRAIN_updates":len(eligible)*TRAIN_EPOCHS,
            "fixed_epochs":TRAIN_EPOCHS,
            "ADAMW_lr":TRAIN_LR,
            "ADAMW_weight_decay":TRAIN_WD,
            "trainable_shadow_state_params":10240,
            "fixed_seeded_permuted_control":name==STATES[1],
            "permuted_uses_only_new_TRAIN_future_targets":True,
            "original43M_gradient_updates":0,
            "candidate_accepted":False,"output_gate":0,
        }
    return models,summary
