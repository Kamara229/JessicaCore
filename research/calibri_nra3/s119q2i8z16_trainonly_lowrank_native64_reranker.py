"""S119q2i8z16: compact TRAIN-only learned top64 native BPE2 reranker CONTRACT.

No new full original43M passes per candidate: capture its existing
last-hidden(320) from the already-made model next-token forward,
and gather the ORIGINAL FROZEN token embedding rows for top64 candidates.
Train one tiny 320x2x320 bilinear residual adapter (1280 parameters)
ONLY on new Z16 human TRAIN with actually correct OWN first BPE
and gold BPE2 ALREADY in the preselected target-blind top64. Missed
gold BPE2 are not fabricated/inserted. Do not train on Z16 DEV.
No cached activation from DEV enters gradient, no teacher forcing at DEV,
no new NRA3/NPA1 regions, no S142/GT/native head weights modified.
No promotion: output_gate=0, independent research shadow ONLY.
TOP64 inference still uses original 43M true native49152 logits.
A fixed training loop may later evaluate against full-native greedy,
but this contract+unit tests are NOT a physically trained candidate.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F

VOCAB=49152
HIDDEN=320
RANK=2
K=64
EOS=2
RESIDUAL_GAIN=.25
INITIAL_SEED=119163
TRAIN_EPOCHS=25
TRAIN_ADAMW_LR=.03
TRAIN_ADAMW_WD=.10

def check(v,reason):
    if not v:
        raise ValueError("Z16 trainonly cheap candidate rerank contract: "+reason)


@contextmanager
def capture_existing_full43m_hidden(model):
    """View the true original model's already-computed LM head input.

    Must surround one actual original43M native readout. Capture only
    the detached final 320 hidden after Core/NRA3/NPA1/GT12. Neither
    hooks, states nor original weights persist after the probe.
    """
    check(hasattr(model,"lm_head") and isinstance(model.lm_head,nn.Module),
          "original frozen native lm_head must exist")
    old=set(model.lm_head._forward_pre_hooks)
    witnessed=[]
    def capture(module,args):
        check(len(witnessed)==0 and len(args)>=1
              and isinstance(args[0],torch.Tensor)
              and args[0].ndim==3 and args[0].shape[0]==1
              and args[0].shape[-1]==HIDDEN
              and bool(torch.isfinite(args[0]).all()),
              "one original43M native lm_head input [1,seq,320] only")
        witnessed.append(args[0][:,-1,:].detach().float().clone())
    handle=model.lm_head.register_forward_pre_hook(capture)
    try:
        yield witnessed
        check(len(witnessed)==1
              and witnessed[0].shape==(1,HIDDEN),
              "exactly one physical 43M hidden recorded")
    finally:
        handle.remove()
        check(set(model.lm_head._forward_pre_hooks)==old,
              "LM head hidden capture MUST rollback hook exactly")


class NativeBPE2CandidateReranker(nn.Module):
    """Existing-context & frozen token embedding, small trainable readout.

    Initialization produces EXACT same scores and ranking as original
    43M native logits. Candidate selection is top64 BEFORE label. This
    reranker is not a new neural area and is never live or auto-accepted.
    """
    def __init__(self):
        super().__init__()
        g=torch.Generator(device="cpu").manual_seed(INITIAL_SEED)
        self.left=nn.Parameter(torch.randn((HIDDEN,RANK),generator=g)*.02)
        self.right=nn.Parameter(torch.zeros((RANK,HIDDEN)))
        self.output_gate=0
        self.candidate_accepted=False

    def forward(self,logits,hidden,frozen_native_embedding):
        check(self.output_gate==0 and self.candidate_accepted is False
              and isinstance(logits,torch.Tensor)
              and tuple(logits.shape)==(1,VOCAB)
              and isinstance(hidden,torch.Tensor)
              and tuple(hidden.shape)==(1,HIDDEN)
              and isinstance(frozen_native_embedding,torch.Tensor)
              and tuple(frozen_native_embedding.shape)==(VOCAB,HIDDEN)
              and bool(torch.isfinite(logits).all())
              and bool(torch.isfinite(hidden).all()),
              "original43M full-native49152 logits + actual h320 + frozen vocab embedding")
        # Absolutely NO gradient to original model/other areas.
        native=logits.detach().float()[0]
        selected=torch.topk(native,K,sorted=True).indices
        candidates=frozen_native_embedding.detach().index_select(0,selected).float()
        check(torch.isfinite(candidates).all(),"finite existing S142 native token embeddings")
        query=F.layer_norm(hidden.detach().float(),(HIDDEN,))[0]
        e=F.normalize(candidates,dim=-1,eps=1e-6)
        residual=(e@((query@self.left)@self.right))*RESIDUAL_GAIN
        out=native[selected]+residual
        check(out.shape==(K,) and bool(torch.isfinite(out).all())
              and selected.unique().numel()==K,
              "target-blind top64 ORIGINAL native logits + cheap lowrank rerank only")
        return selected,out

    def trainonly_loss(self,logits,hidden,frozen_native_embedding,
                       target_native_BPE2,*,split,own_first_correct):
        check(split=="TRAIN" and type(own_first_correct) is bool
              and type(target_native_BPE2) is int
              and 0<=target_native_BPE2<VOCAB,
              "only Z16 new human TRAIN gold BPE2 after correct OWN first")
        ids,score=self.forward(logits,hidden,frozen_native_embedding)
        present=(ids==target_native_BPE2).nonzero(as_tuple=False).flatten()
        eligible=bool(own_first_correct and present.numel()==1)
        if eligible:
            loss=F.cross_entropy(score.unsqueeze(0),present.long())
        else:
            # Neither hallucinate gold into candidate set nor mine
            # any gradients from incorrect own-first contexts.
            loss=score.sum()*0.
        return loss,{
            "source_split":"TRAIN",
            "first_own_gold_match":own_first_correct,
            "human_second_gold_in_original_native_top64":bool(present.numel()==1),
            "eligible_for_gradient":eligible,
            "top64_contains_EOS":bool((ids==EOS).any()),
        }


def exact_fixed_trainonly_reranker(train_records, vocab_embeddings):
    """Epochs/budget/optimizer locked before DEV, never touch Z16 DEV.

    Records are original43M *cached detached TRAIN-only* tensors,
    not a new training corpus for the pretrained model. No updates of
    the original S142/GT/Core/NRA3/NPA1/native full head are allowed.
    """
    check(isinstance(train_records,(list,tuple)) and len(train_records)==28
          and len({r["source_id"] for r in train_records})==28
          and all(r["split"]=="TRAIN"
                  and str(r["source_id"]).startswith("S119Q2I8Z16_")
                  for r in train_records),
          "28 NEW prereg TRAIN sources only (no spent DEV or GOLD)")
    ranker=NativeBPE2CandidateReranker()
    original_embed=vocab_embeddings.detach().clone()
    check(original_embed.shape==(VOCAB,HIDDEN)
          and not original_embed.requires_grad,
          "original frozen native vocabulary embeddings")
    optimizer=torch.optim.AdamW(ranker.parameters(),
                               lr=TRAIN_ADAMW_LR,weight_decay=TRAIN_ADAMW_WD)
    accepted=[]
    for r in train_records:
        check(type(r["target_native_BPE2"]) is int
              and type(r["own_first_correct"]) is bool,
              "exact TRAIN-only human BPE2 with actual own first status")
        # Source-only eligibility check, no loss on DEV or own first fail.
        eligible=ranker.trainonly_loss(
            r["logits"],r["hidden"],original_embed,r["target_native_BPE2"],
            split="TRAIN",own_first_correct=r["own_first_correct"])[1][
                "eligible_for_gradient"]
        accepted.append(eligible)
    check(sum(accepted)>=2,
          "fail closed: insufficient original native gold-BPE2 top64 coverage in TRAIN")
    for epoch in range(TRAIN_EPOCHS):
        for r,eligible in zip(train_records,accepted):
            if not eligible:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss,_=ranker.trainonly_loss(
                r["logits"],r["hidden"],original_embed,
                r["target_native_BPE2"],split="TRAIN",
                own_first_correct=r["own_first_correct"])
            check(bool(torch.isfinite(loss)) and loss.requires_grad,
                  "learned residual is differentiable only on TRAIN source")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(),1.)
            optimizer.step()
    check(sum(x.numel() for x in ranker.parameters())==2*HIDDEN*RANK
          and all(bool(torch.isfinite(p).all()) for p in ranker.parameters())
          and ranker.output_gate==0 and ranker.candidate_accepted is False,
          "1280-param neural candidate readout only, not accepted")
    return ranker,{
        "new_Z16_TRAIN_records":28,
        "TRAIN_only_eligible_original_TOP64_gold_second":sum(accepted),
        "TRAIN_epochs_fixed":TRAIN_EPOCHS,
        "fixed_ADAMW_lr":TRAIN_ADAMW_LR,
        "fixed_ADAMW_weight_decay":TRAIN_ADAMW_WD,
        "reranker_trainable_params":2*HIDDEN*RANK,
        "original_model_gradient_steps":0,
        "candidate_accepted":False,"output_gate":0,
    }
