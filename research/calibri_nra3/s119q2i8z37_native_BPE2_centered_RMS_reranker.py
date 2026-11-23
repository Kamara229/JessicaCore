"""Z37 new native second BPE comparator, same frozen S142 original model.

Actual Z36 read-only forensic on previously SPENT Z34 dev found
8/13 first-correct human BPE2 already in original native top64,
mean original logits greedy minus gold ≈0.86–0.87 (<±1.5
candidate correction max pairwise3), but previously fitted rank32
human and WRONG control produced exact ZERO relative gold-vs-greedy
residual, despite tanh ABSOLUTE amplitude 1.5. This identifies
degenerate candidate common-mode/saturation rather than inadequate
original greedy-gap scale.

Preserve exact same 30720param rank32 h320+OWN token embedding,
64 native original targetblind candidates, 84 PREVIOUSLY LEGAL
TRAIN, same train epochs and hyperparams and WRONG-offset17
control. ONLY changed neural readout: raw 64 candidate dot-products
are mean-centered, RMS normalized floor .1, divided2 and tanh;
tanh scores then mean-centered and bounded ±1.5; gradients remain
relative to candidate slate. No original model weights changes,
no DEV gradients. GATE0 research-only, no production.
"""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z37_new14dev_candidate_centered_rms_source import (
    TOP_K,RANK,HIDDEN_DIM,INPUT_DIM,TRAIN_EPOCHS,
    LEARNING_RATE,WEIGHT_DECAY,LOGIT_RESIDUAL_BOUND,
    ORIGINAL_TOP64_KL_WEIGHT,HARD_NEGATIVE_MARGIN,
    CORRECT_ORIGINAL_GREEDY_WEIGHT,WRONG_OFFSET17_CONTROL,
    ACTIVATE_SECOND_BPE_ONLY,TRAIN_ELIGIBLE_MIN,
    CENTERED_CANDIDATE_RMS_FLOOR,CENTERED_PRE_TANH_DIVISOR,
)
NATIVE_VOCAB=49152
INIT_SEED=119134
MODES=("HUMAN_GOLD_BPE2","WRONG_OFFSET17_LABEL_CONTROL")
SHADOW_PARAMS=INPUT_DIM*RANK+RANK*HIDDEN_DIM
def check(ok,why):
    if not ok:raise ValueError("Z34 strict second original-native-BPE only gated rank32: "+why)

class Position2NativeCandidateComparator(nn.Module):
    def __init__(self):
        super().__init__()
        rng=torch.Generator().manual_seed(INIT_SEED)
        self.input_project=nn.Parameter(torch.randn(INPUT_DIM,RANK,generator=rng)*.013)
        self.output_project=nn.Parameter(torch.zeros(RANK,HIDDEN_DIM))
        self.output_gate=0
        self.candidate_accepted=False
        check(sum(p.numel() for p in self.parameters())==SHADOW_PARAMS==30720
              and INPUT_DIM==640 and RANK==32 and HIDDEN_DIM==320,
              "only 30720 new research params; original native table completely frozen")

    def forward(self,original49152,h320,prior_own_native_embedding320,
                frozen_native49152_embeddings,*,native_bpe_position):
        check(native_bpe_position==2 and ACTIVATE_SECOND_BPE_ONLY
              and original49152.shape==(1,NATIVE_VOCAB)
              and h320.shape==(1,HIDDEN_DIM)
              and prior_own_native_embedding320.shape==(1,HIDDEN_DIM)
              and frozen_native49152_embeddings.shape==(NATIVE_VOCAB,HIDDEN_DIM)
              and not original49152.requires_grad
              and not h320.requires_grad
              and not prior_own_native_embedding320.requires_grad
              and not frozen_native49152_embeddings.requires_grad
              and torch.isfinite(original49152).all()
              and torch.isfinite(h320).all()
              and LOGIT_RESIDUAL_BOUND==1.5 and TOP_K==64,
              "BPE2 ON-policy exact original native logits same paid h320 + frozen last OWN native embedding")
        native=original49152.float()[0]
        ids=torch.topk(native,TOP_K,sorted=True).indices
        relevant=frozen_native49152_embeddings.detach().index_select(0,ids).float()
        candidate_emb=F.normalize(relevant,dim=-1,eps=1e-6)
        x=torch.cat([F.layer_norm(h320.detach().float(),(HIDDEN_DIM,)),
                     F.layer_norm(prior_own_native_embedding320.detach().float(),
                                  (HIDDEN_DIM,))],dim=-1)
        query=(x@self.input_project)@self.output_project
        # Root cause (actual frozen original S142 43M Z36): previous
        # bounded rank32 wrote ±1.5 ABSOLUTE correction, but zero
        # GOLD-vs-GREEDY RELATIVE correction on spent native BPE2.
        # Never feed a large global common-mode candidate dot product
        # to tanh: it saturates the ENTIRE top64 and preserves greedy
        # original ordering regardless of learned labels.
        raw_candidate_logits=candidate_emb@query[0]
        raw_contrast=raw_candidate_logits-raw_candidate_logits.mean()
        # Clamp variance BEFORE square root: sqrt(0) has infinite
        # reverse derivative even when later clamped (0 * Inf -> NaN).
        # Identical forward max(RMS,.10), but gradient at zero init
        # stays finite, and centered candidate contrast is trainable.
        slate_rms=raw_contrast.square().mean().clamp_min(
            CENTERED_CANDIDATE_RMS_FLOOR**2).sqrt()
        unsaturated=(raw_contrast/slate_rms)/CENTERED_PRE_TANH_DIVISOR
        bounded_tanh=torch.tanh(unsaturated)
        centered_bounded=bounded_tanh-bounded_tanh.mean()
        # Differentiable bounded and ZERO-mean relative native score:
        # unlike absolute global sigmoid saturation, common-mode
        # score cannot consume the full ±1.5 corrective budget.
        scale=centered_bounded.abs().max().clamp_min(1.0)
        residual=LOGIT_RESIDUAL_BOUND*centered_bounded/scale
        score=native.index_select(0,ids)+residual
        check(score.shape==(TOP_K,)
              and ids.unique().numel()==TOP_K
              and bool((residual.abs()<=LOGIT_RESIDUAL_BOUND+1e-6).all())
              and torch.isfinite(score).all(),
              "reversible original-native top64 candidate-only BPE2 scoring no gold/EOS masking")
        return ids,score,residual

    def train_only_loss(self,original49152,h320,prev_emb,native_emb,
                        *,gold_second,own_first_correct,split,wrong_control):
        check(split=="TRAIN" and type(own_first_correct) is bool
              and type(wrong_control) is bool
              and type(gold_second) is int and 0<=gold_second<NATIVE_VOCAB,
              "only human gold in LEGAL TRAIN, no spent DEV/GOLD/EVAL")
        ids,score,residual=self(
            original49152,h320,prev_emb,native_emb,native_bpe_position=2)
        pos=(ids==gold_second).nonzero(as_tuple=False).flatten()
        eligible=bool(own_first_correct and pos.numel()==1)
        if not eligible:
            return score.sum()*0.,{"eligible":False,
                 "gold_second_original_top64":bool(pos.numel()==1),
                 "first_own_correct":own_first_correct}
        correct_index=int(pos.item())
        target=(correct_index+WRONG_OFFSET17_CONTROL)%TOP_K if wrong_control else correct_index
        baseline=original49152.detach().float()[0,ids]
        ce=F.cross_entropy(score.view(1,-1),torch.tensor([target],dtype=torch.long))
        baseline_kl=F.kl_div(F.log_softmax(score,dim=0),
                            F.softmax(baseline,dim=0),reduction="sum")
        hard=F.relu(score[0]-score[target]+HARD_NEGATIVE_MARGIN)
        penalty=(residual.square()).mean()
        weight=CORRECT_ORIGINAL_GREEDY_WEIGHT if correct_index==0 and not wrong_control else 1.
        loss=(weight*ce+ORIGINAL_TOP64_KL_WEIGHT*baseline_kl
              +hard+0.02*penalty)
        check(torch.isfinite(loss) and loss.requires_grad,
              "source-only native next BPE candidate margin, no train on absent gold")
        return loss,{"eligible":True,
                     "gold_second_original_top64":True,
                     "first_own_correct":True,
                     "original_second_top1_correct":correct_index==0,
                     "wrong_target_training_control":wrong_control}

def fit_two_trainonly_shadow_controls(records,native_emb):
    check(len(records)==84
          and len({r["source_id"] for r in records})==84
          and all(r["split"]=="TRAIN"
                  and r["source_id"].startswith("S119Q2I8Z37_LEGAL_")
                  and type(r["own_first_correct"]) is bool
                  and type(r["gold_second"]) is int
                  and r["native_logits"].shape==(1,NATIVE_VOCAB)
                  and r["h320"].shape==(1,HIDDEN_DIM)
                  and r["prev_own_embed"].shape==(1,HIDDEN_DIM)
                  and all(not r[k].requires_grad
                          for k in ("native_logits","h320","prev_own_embed"))
                  for r in records)
          and native_emb.shape==(NATIVE_VOCAB,HIDDEN_DIM)
          and not native_emb.requires_grad,
          "84 legal TRAIN-only original human prompts with actual OWN first native BPE, no spent DEV")
    eligible=[bool(r["own_first_correct"] and (
         torch.topk(r["native_logits"][0],TOP_K).indices==r["gold_second"]).any())
         for r in records]
    check(sum(eligible)>=TRAIN_ELIGIBLE_MIN==20,
          "fail closed: enough true original43M correct own first and BPE2 present in original targetblind top64")
    models={}
    audit={}
    for mode in MODES:
        control=mode==MODES[1]
        shadow=Position2NativeCandidateComparator()
        opt=torch.optim.AdamW(shadow.parameters(),lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
        for _epoch in range(TRAIN_EPOCHS):
            for rec,can_train in zip(records,eligible):
                if not can_train:continue
                opt.zero_grad(set_to_none=True)
                loss,meta=shadow.train_only_loss(
                    rec["native_logits"],rec["h320"],rec["prev_own_embed"],
                    native_emb,gold_second=rec["gold_second"],
                    own_first_correct=rec["own_first_correct"],
                    split="TRAIN",wrong_control=control)
                check(meta["eligible"],"original native source-only gold BPE2 eligible")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(shadow.parameters(),1.0)
                opt.step()
        shadow.eval()
        check(sum(p.numel() for p in shadow.parameters())==SHADOW_PARAMS
              and all(torch.isfinite(p).all() for p in shadow.parameters())
              and shadow.output_gate==0 and shadow.candidate_accepted is False,
              "only tiny BPE2-only neural shadow updated, no original S142")
        models[mode]=shadow
        audit[mode]={"legal_TRAIN84":84,
                     "gold_inside_original_NATIVE_top64_and_OWN_first_correct":
                         sum(eligible),
                     "first_gold_native_BPE2_match_already_original_GREEDY":
                         sum(bool(good and int(torch.argmax(r["native_logits"]))==
                                  r["gold_second"]) for r,good in zip(records,eligible)),
                     "shadow_rank32_params":SHADOW_PARAMS,
                     "train_epochs":TRAIN_EPOCHS,
                     "optimizer_steps":sum(eligible)*TRAIN_EPOCHS,
                     "wrong_offset17_label_control":control,
                     "zero_model_S142_optimizer_steps":True,
                     "zero_previous_SPENT_DEV_gradient":True,
                     "only_NATIVE_SECOND_BPE_activation":True,
                     "candidate_accepted":False,"output_gate":0}
    return models,audit
