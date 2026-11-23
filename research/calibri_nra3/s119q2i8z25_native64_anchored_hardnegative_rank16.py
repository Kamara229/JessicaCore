"""Z25 native hard-negative TRAIN-only contrastive predictor, fixed output_gate0.

Specific fix to earlier failed Z22 Predictive State: don't train mean
of next3 native token embeddings (cannot distinguish gold candidate).
Supervise the correct ORIGINAL native BPE2 against 63 target-blind
original model hard negative token candidates. No gold injection.
Bound contextual residual ±3 and regularize KL to original distribution
and squared deviation, to avoid prior score-collapse from unconstrained
shadow. Negative control: deliberately WRONG shifted candidate index
on same eligible human TRAIN source. Same initial weights, optimizer,
update steps. Distinguishes real correct-candidate learning from simply
learning decoder token statistics. Eligible only correct model OWN first
and original gold second already among 64 target-blind candidates.

No more original S142 43M full forwards per candidate. No change to
original model, 12GT/Core/NRA3/NPA1/native head nor production.
NEW DEV human answer never enters loss, no data leakage.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z25_84historicaltrain_new14dev_hardnegative_source import (
    TOP_K,RANK,RESIDUAL_BOUND,TRAIN_EPOCHS,TRAIN_ADAMW_LR,
    TRAIN_ADAMW_WD,TRAIN_KL_ANCHOR,TRAIN_RESIDUAL_L2,
    NEGATIVE_TARGET_OFFSET,
)

HIDDEN=320
VOCAB=49152
EOS=2
INIT_SEED=119125
MODES=("HUMAN_BPE2_HARDNEG","SHIFTED_WRONG_TRAIN_CANDIDATE_CONTROL")

def check(ok,reason):
    if not ok:raise ValueError("Z25 strict frozen original hardnegative native64: "+reason)

class Native64AnchoredHardNegative(nn.Module):
    def __init__(self):
        super().__init__()
        g=torch.Generator(device="cpu").manual_seed(INIT_SEED)
        self.left=nn.Parameter(torch.randn((HIDDEN,RANK),generator=g)*.02)
        self.right=nn.Parameter(torch.zeros((RANK,HIDDEN)))
        self.output_gate=0
        self.candidate_accepted=False
        check(sum(p.numel() for p in self.parameters())==2*HIDDEN*RANK,
              "compact 10240 actual neural shadow parameters only")

    def forward(self,native_logits,h320,orig_native_emb):
        check(self.output_gate==0 and self.candidate_accepted is False
              and isinstance(native_logits,torch.Tensor)
              and native_logits.shape==(1,VOCAB)
              and isinstance(h320,torch.Tensor) and h320.shape==(1,HIDDEN)
              and isinstance(orig_native_emb,torch.Tensor)
              and orig_native_emb.shape==(VOCAB,HIDDEN)
              and not orig_native_emb.requires_grad
              and bool(torch.isfinite(native_logits).all())
              and bool(torch.isfinite(h320).all())
              and TOP_K==64 and RESIDUAL_BOUND==3.0,
              "actual original native49152 logits, same paid h320, frozen original embeddings")
        native=native_logits.detach().float()[0]
        ids=torch.topk(native,TOP_K,sorted=True).indices
        candidate_e=F.normalize(orig_native_emb.detach().index_select(
            0,ids).float(),dim=-1,eps=1e-6)
        h=F.layer_norm(h320.detach().float(),(HIDDEN,))
        query=(h@self.left)@self.right
        raw=candidate_e@query[0]
        residual=RESIDUAL_BOUND*torch.tanh(raw)
        scores=native.index_select(0,ids)+residual
        check(ids.shape==(TOP_K,) and scores.shape==(TOP_K,)
              and torch.isfinite(scores).all()
              and bool((residual.abs()<=RESIDUAL_BOUND+1e-5).all())
              and ids.unique().numel()==TOP_K,
              "native shortlist target-free, bounded logits residual, no EOS mask")
        return ids,scores,residual

    def train_only_loss(self,native_logits,h320,orig_native_emb,
                        human_second,*,split,own_first_correct,control=False):
        check(split=="TRAIN" and type(own_first_correct) is bool
              and type(human_second) is int
              and 0<=human_second<VOCAB and type(control) is bool,
              "TRAIN only human next native BPE candidate, never spent DEV")
        ids,score,residual=self(native_logits,h320,orig_native_emb)
        pos=(ids==human_second).nonzero(as_tuple=False).flatten()
        eligible=bool(own_first_correct and len(pos)==1)
        if not eligible:
            return score.sum()*0.,{
                "eligible":False,
                "gold_inside_original_top64":bool(len(pos)==1),
                "correct_own_first":own_first_correct,
                "original_top1_correct":False,
            }
        gold_index=int(pos.item())
        label=(gold_index+NEGATIVE_TARGET_OFFSET)%TOP_K if control else gold_index
        # KL(original||reranked): anchored native model, with penalty on
        # in-distribution deviation and priority for originally correct
        # cases, so do not arbitrarily flip correct greedy.
        ref=native_logits.detach().float()[0,ids]
        ce=F.cross_entropy(score.unsqueeze(0),
                           torch.tensor([label],dtype=torch.long))
        kl=F.kl_div(F.log_softmax(score,dim=0),
                    F.softmax(ref,dim=0),reduction="sum")
        reg=(residual.square()).mean()
        original_correct=gold_index==0
        # Strong penalty for losing a correct original greedy to bad shift.
        weight=2.0 if original_correct and not control else 1.0
        loss=weight*ce+TRAIN_KL_ANCHOR*kl+TRAIN_RESIDUAL_L2*reg
        check(torch.isfinite(loss) and loss.requires_grad,
              "real differentiable candidate-specific native human next BPE")
        return loss,{
            "eligible":True,
            "gold_inside_original_top64":True,
            "correct_own_first":True,
            "original_top1_correct":original_correct,
            "train_negative_control_target_shifted":control,
        }

def train_matched_control_on_historical_train(records,orig_native_emb):
    check(isinstance(records,(tuple,list)) and len(records)==84
          and len({r["source_id"] for r in records})==84
          and all(r["split"]=="TRAIN"
                  and r["source_id"].startswith("S119Q2I8Z25_TRONLY_")
                  and type(r["target_BPE2"]) is int
                  and 0<=r["target_BPE2"]<VOCAB
                  and type(r["own_first_correct"]) is bool
                  and r["native_logits"].shape==(1,VOCAB)
                  and r["h320"].shape==(1,HIDDEN)
                  and not r["h320"].requires_grad
                  and not r["native_logits"].requires_grad
                  for r in records),
          "exact 84 previously labelled TRAIN-only original author sources, NO spent DEV")
    check(orig_native_emb.shape==(VOCAB,HIDDEN)
          and not orig_native_emb.requires_grad,
          "frozen original S142 TrainEmbedding native49152 table")
    all_ids=[torch.topk(r["native_logits"][0],TOP_K).indices for r in records]
    eligible=[bool(r["own_first_correct"] and
                   (ids==r["target_BPE2"]).any())
              for r,ids in zip(records,all_ids)]
    check(sum(eligible)>=15,
          "fail closed: insufficient true original43M gold-second candidate TRAIN coverage")
    correct_ref=sum(bool(use and int(ids[0])==r["target_BPE2"])
                   for r,ids,use in zip(records,all_ids,eligible))
    models={}
    summary={}
    for mode in MODES:
        ranker=Native64AnchoredHardNegative()
        optimizer=torch.optim.AdamW(ranker.parameters(),lr=TRAIN_ADAMW_LR,
                                    weight_decay=TRAIN_ADAMW_WD)
        is_control=mode==MODES[1]
        for epoch in range(TRAIN_EPOCHS):
            ranker.train()
            for rec,use in zip(records,eligible):
                if not use:continue
                optimizer.zero_grad(set_to_none=True)
                loss,diagnostic=ranker.train_only_loss(
                    rec["native_logits"],rec["h320"],orig_native_emb,
                    rec["target_BPE2"],split="TRAIN",
                    own_first_correct=rec["own_first_correct"],
                    control=is_control)
                check(diagnostic["eligible"],
                      "never inject human BPE2 into missing original top64")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ranker.parameters(),1.0)
                optimizer.step()
        ranker.eval()
        check(all(torch.isfinite(p).all() for p in ranker.parameters())
              and sum(p.numel() for p in ranker.parameters())==10240
              and ranker.output_gate==0
              and ranker.candidate_accepted is False,
              "only shadow 10240 param updated; no promotion")
        models[mode]=ranker
        summary[mode]={
            "previously_legal_human_TRAIN_only_sources":84,
            "true_gold_eligible_targetblind_native_top64":sum(eligible),
            "source_native_greedy_correct_within_eligible":correct_ref,
            "shadow_neural_params":10240,
            "fixed_training_epochs":TRAIN_EPOCHS,
            "shadow_optimizer_updates":sum(eligible)*TRAIN_EPOCHS,
            "fixed_optimizer_lr":TRAIN_ADAMW_LR,
            "fixed_optimizer_weight_decay":TRAIN_ADAMW_WD,
            "frozen_logit_KL_anchor":TRAIN_KL_ANCHOR,
            "bounded_residual_logit_magnitude":RESIDUAL_BOUND,
            "wrong_target_negcontrol_shift":NEGATIVE_TARGET_OFFSET if is_control else None,
            "original43M_gradient_steps":0,
            "DEV_gradient_steps":0,
            "candidate_accepted":False,"output_gate":0,
        }
    return models,summary
