"""Z40 only separate trainable position-2/3/4 native top64 shadow (source-only).

Policy: train on legal TRAIN teacher prefixes only when target already
appears in original S142's *target-blind* native top64; evaluate only
self-generated own prefixes. Candidate entropy floor is enforced at
inference without gold access. Original 43M/GT12/Core/NRA3/NPA1 frozen.
"""
from __future__ import annotations
import math
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as cfg

NATIVE_VOCAB=49152
SEED=119140
MODES=("HUMAN_GOLD_BPE234", "WRONG_OFFSET17_BPE234")
LOGIT_KL_CEILING=.12

def require(value, why):
    if not value:
        raise ValueError("Z40 research native BPE234 fail closed: "+why)


class CausalBPE234CandidateComparator(nn.Module):
    def __init__(self):
        super().__init__()
        rng=torch.Generator().manual_seed(SEED)
        self.input_project=nn.Parameter(torch.randn(cfg.INPUT_DIM,cfg.RANK,generator=rng)*.013)
        self.output_project=nn.Parameter(torch.zeros(cfg.RANK,cfg.HIDDEN_DIM))
        self.position_context=nn.Parameter(torch.zeros(len(cfg.POSITIONS),cfg.HIDDEN_DIM))
        self.output_gate=0
        self.candidate_accepted=False
        require(sum(p.numel() for p in self.parameters())==cfg.SHADOW_PARAMS==31680,
                "only 31680 separate shadow parameters")

    def forward(self,original,h320,prior_emb,native_emb,*,position):
        require(position in cfg.POSITIONS and original.shape==(1,NATIVE_VOCAB)
                and h320.shape==(1,cfg.HIDDEN_DIM)
                and prior_emb.shape==(1,cfg.HIDDEN_DIM)
                and native_emb.shape==(NATIVE_VOCAB,cfg.HIDDEN_DIM)
                and all(not v.requires_grad for v in (original,h320,prior_emb,native_emb))
                and torch.isfinite(original).all() and torch.isfinite(h320).all()
                and torch.isfinite(prior_emb).all() and torch.isfinite(native_emb).all(),
                "frozen original 49k logits and same-pass h320, own/gold preceding native embedding")
        logits=original.float()[0]
        ids=torch.topk(logits,cfg.TOP_K,sorted=True).indices
        native_candidates=F.normalize(native_emb.detach().index_select(0,ids).float(),
                                      dim=-1,eps=1e-6)
        x=torch.cat((F.layer_norm(h320.detach().float(),(cfg.HIDDEN_DIM,)),
                     F.layer_norm(prior_emb.detach().float(),(cfg.HIDDEN_DIM,))),dim=-1)
        context=(x@self.input_project)@self.output_project + self.position_context[position-2]
        raw=native_candidates@context[0]
        centered=raw-raw.mean()
        rms=centered.square().mean().clamp_min(cfg.CENTERED_RMS_FLOOR**2).sqrt()
        squashed=torch.tanh(centered/rms/cfg.CENTERED_DIVISOR)
        zero_mean=squashed-squashed.mean()
        residual=cfg.LOGIT_RESIDUAL_BOUND*zero_mean/zero_mean.abs().max().clamp_min(1.)
        updated=logits.index_select(0,ids)+residual
        require(updated.shape==(cfg.TOP_K,) and torch.isfinite(updated).all() and
                float(residual.abs().max().detach())<=cfg.LOGIT_RESIDUAL_BOUND+1e-5,
                "finite bound and target-blind top64 neural candidate logits")
        return ids,updated,residual

    @staticmethod
    def entropy_of(logits):
        probs=F.softmax(logits.float(),dim=0)
        return -(probs*F.log_softmax(logits.float(),dim=0)).sum()

    @staticmethod
    @torch.no_grad()
    def guarded_scores(original_top64,modified_top64):
        """Select most corrective fixed-scale interpolation satisfying
        per-prompt entropy floor and KL trust region. No human label.
        Works on fixed ORIGINAL candidate set; no EOS masking.
        """
        require(original_top64.shape==modified_top64.shape==(cfg.TOP_K,),
                "both distributions same physical original top64")
        baseline=original_top64.detach().float()
        corrected=modified_top64.detach().float()
        baseline_entropy=CausalBPE234CandidateComparator.entropy_of(baseline)
        tolerance=cfg.ENTROPY_FLOOR_TOLERANCE_BITS*math.log(2.)
        for alpha in (1.,.75,.5,.375,.25,.125,.0625,.03125,0.):
            candidate=baseline+(corrected-baseline)*alpha
            entropy=CausalBPE234CandidateComparator.entropy_of(candidate)
            kl=F.kl_div(F.log_softmax(candidate,dim=0),
                        F.softmax(baseline,dim=0),reduction="sum")
            if bool(entropy>=baseline_entropy-tolerance-1e-7 and
                    kl<=LOGIT_KL_CEILING+1e-7):
                return candidate,alpha,float(entropy-baseline_entropy)/math.log(2.)
        raise AssertionError("original baseline alpha0 must always pass")

    def train_loss(self,record,embedding,*,wrong_label):
        require(record["split"]=="TRAIN" and record["gold_prefix_only"] is True
                and type(wrong_label) is bool and record["position"] in cfg.POSITIONS
                and record["original_logits"].shape==(1,NATIVE_VOCAB)
                and type(record["gold_token"]) is int,
                "gold may appear only in legal 84 TRAIN teacher-prefix, NEVER DEV")
        ids,score,residual=self(record["original_logits"],record["h320"],
                                record["prior_emb"],embedding,position=record["position"])
        idx=(ids==record["gold_token"]).nonzero(as_tuple=False).flatten()
        if idx.numel()==0:
            return None
        require(idx.numel()==1,"one native gold candidate in original top64")
        target=(int(idx.item())+cfg.WRONG_OFFSET17_CONTROL)%cfg.TOP_K if wrong_label else int(idx.item())
        baseline=record["original_logits"].float()[0].index_select(0,ids).detach()
        ce=F.cross_entropy(score.view(1,-1),torch.tensor([target]))
        kl=F.kl_div(F.log_softmax(score,dim=0),
                    F.softmax(baseline,dim=0),reduction="sum")
        entropy_dropped=F.relu(self.entropy_of(baseline)-self.entropy_of(score)
                      -cfg.ENTROPY_FLOOR_TOLERANCE_BITS*math.log(2.))
        margin=F.relu(score[0]-score[target]+cfg.HARD_NEGATIVE_MARGIN)
        penalty=0.02*residual.square().mean()
        weighted=cfg.PER_POSITION_LOSS_WEIGHT[record["position"]]
        loss=weighted*(ce+cfg.KL_TO_ORIGINAL_TOP64*kl+
                       cfg.ENTROPY_COLLAPSE_PENALTY*entropy_dropped+
                       cfg.HARD_NEGATIVE_WEIGHT*margin+penalty)
        require(loss.requires_grad and torch.isfinite(loss),"finite train-only objective")
        return loss


def fit_controls(records,embeddings):
    """Exactly 84*3 TRAIN original frozen teacher-prefix states, no DEV."""
    require(len(records)==252 and embeddings.shape==(NATIVE_VOCAB,cfg.HIDDEN_DIM)
            and not embeddings.requires_grad,
            "84 legitimate TRAIN x BPE2/3/4, original native embedding frozen")
    source_ids={r["source_id"] for r in records}
    require(len(source_ids)==84 and all(r["source_id"].startswith("S119Q2I8Z40_LEGAL_")
            and r["split"]=="TRAIN" and r["gold_prefix_only"] is True
            and r["position"] in cfg.POSITIONS for r in records),
            "TRAIN only 84 distinct and 3 positions each")
    count=Counter(r["position"] for r in records)
    require(count==Counter({p:84 for p in cfg.POSITIONS}), "complete 84*3 source positions")
    eligible=Counter()
    for r in records:
        require(all(not r[k].requires_grad for k in
                    ("original_logits","h320","prior_emb")),
                "no backwards into any original 43M model state")
        if (torch.topk(r["original_logits"][0],cfg.TOP_K).indices==r["gold_token"]).any():
            eligible[r["position"]]+=1
    require(all(eligible[p]>=cfg.MIN_ELIGIBLE_TRAIN_PER_POSITION for p in cfg.POSITIONS),
            "precommitted minimum target-blind native top64 coverages")
    models={}
    audits={}
    for mode in MODES:
        wrong=(mode==MODES[1])
        net=CausalBPE234CandidateComparator()
        optimizer=torch.optim.AdamW(net.parameters(),lr=cfg.LEARNING_RATE,
                                   weight_decay=cfg.WEIGHT_DECAY)
        steps=0
        for _ in range(cfg.TRAIN_EPOCHS):
            for rec in records:
                if not (torch.topk(rec["original_logits"][0],cfg.TOP_K).indices==
                        rec["gold_token"]).any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss=net.train_loss(rec,embeddings,wrong_label=wrong)
                require(loss is not None,"candidate eligible for training")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step()
                steps+=1
        net.eval()
        require(net.output_gate==0 and net.candidate_accepted is False
                and all(torch.isfinite(v).all() for v in net.parameters()),
                "fitted shadow finite, reject deployment")
        models[mode]=net
        audits[mode]={"eligible_legal_TRAIN_native_top64_by_BPE_position":dict(eligible),
                      "epochs":cfg.TRAIN_EPOCHS,"optimizer_steps":steps,
                      "updated_shadow_params_only":cfg.SHADOW_PARAMS,
                      "wrong_offset17_training":wrong,
                      "DEV_optimizer_steps":0,
                      "original43M_GT12_and_Z10_optimizer_steps":0,
                      "candidate_accepted":False,"output_gate":0}
    return models,audits
