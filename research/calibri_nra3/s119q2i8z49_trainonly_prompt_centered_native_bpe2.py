"""Z49 isolated TRAIN-only hypothesis: subtract global prompt context mean.

Z45/Z47 on-policy HUMAN BPE2 30,720-parameter rank32 adapter provides
evidence of better first2 recognition but collapses second-token ID
diversity. Per-prompt entropy + KL were insufficient to preserve the
population empirical entropy. One SMALL blinded-to-DEV architecture
change: compute the mean of actual S142 native320 BPE2 hidden + actual
preceding OWN BPE1 native320 embedding over the 72 LEGAL TRAIN
prompts in each leave-one-kind-out fold; subtract this mean from the
layer-normalized 640 context BEFORE the unchanged rank32 bilinear head.

This removes global prompt-invariant token bias as a *hypothesis*, NOT
a result. All model and training hyperparameters, target-blind native
top64, bounded centered-RMS tanh residual, and per-prompt KL/entropy
guard are identical to Z45. Compare HUMAN labels vs WRONG17 control;
test on 12 previous legal TRAIN sources from the held-out KIND only.

The trained original S142 43M, GT12, Core/NRA3/NPA1 and Z10 are
frozen. No DEV, J7, EVAL128, GOLD, Jessica chat, deployment, approval,
original optimizer steps or accepted shadow weights. output_gate=0.
"""
from __future__ import annotations
import math
from collections import Counter
import torch
from torch.nn import functional as F
from research.calibri_nra3 import s119q2i8z45_trainonly_onpolicy_bpe2_domain_holdout as z45
from research.calibri_nra3.s119q2i8z37_native_BPE2_centered_RMS_reranker import (
    Position2NativeCandidateComparator,
    HIDDEN_DIM, INPUT_DIM, NATIVE_VOCAB, TOP_K,
    LOGIT_RESIDUAL_BOUND, CENTERED_CANDIDATE_RMS_FLOOR,
    CENTERED_PRE_TANH_DIVISOR,SHADOW_PARAMS,
)
from research.calibri_nra3.s119q2i8z40_causal_BPE234_entropy_guard import (
    CausalBPE234CandidateComparator
)

STAGE="S119Q2I8Z49_ORIGINAL43M_LEGAL_TRAIN84_ONLY_CENTERED_OWN_BPE2_LODO7"
FOLDS=tuple(z45.DOMAINS)
EPOCHS=z45.EPOCHS
LR=z45.LR
WEIGHT_DECAY=z45.WEIGHT_DECAY
SEED=z45.SEED
MIN_ELIGIBLE=31
MIN_CORRECT_FIRST2=6
MIN_HUMAN_MINUS_WRONG_FIRST2=3
MIN_DISTINCT_FRACTION=.75
MIN_EMPIRICAL_ENTROPY_DELTA_BITS=-.25
ORIGINAL_UNCHANGED=True
SHADOW_ACCEPTED=False
OUTPUT_GATE=0
NATIVE_EMBEDDING_FROZEN=True
NO_DEV=True

def check(ok,why):
    if not ok:raise ValueError("Z49 frozen original onpolicy context-centered TRAIN-only: "+why)

def paired_context(records):
    """Unlabeled originally-computed OWN h320 and previous native embedding.

    No gold, no teacher forced prefix or dev in this statistic. Each
    fold's 12 heldout examples are excluded from mean computation.
    """
    check(len(records)==72 and all(r["split"]=="TRAIN" for r in records),
          "only 72 previous legitimate TRAIN sources, no heldout rows")
    with torch.no_grad():
        features=[
            torch.cat([F.layer_norm(r["h320"].detach().float(),(HIDDEN_DIM,)),
                       F.layer_norm(r["prev_own_embed"].detach().float(),
                                    (HIDDEN_DIM,))],dim=-1)
            for r in records
        ]
        stacked=torch.cat(features,dim=0)
        check(stacked.shape==(72,INPUT_DIM) and torch.isfinite(stacked).all(),
              "true original frozen 640 dimensional own-prefix contexts")
        out=stacked.mean(dim=0,keepdim=True)
        check(out.shape==(1,INPUT_DIM) and torch.isfinite(out).all(),
              "one TRAIN-only mean for both correct and WRONG training controls")
        return out

class ContextCenteredNativeBPE2Comparator(Position2NativeCandidateComparator):
    def __init__(self,training_center):
        super().__init__()
        check(training_center.shape==(1,INPUT_DIM) and
              not training_center.requires_grad and
              torch.isfinite(training_center).all(),
              "unlabeled LEGAL TRAIN72 center computed before training")
        self.register_buffer("train_context_mean",training_center.detach().clone())
        self.output_gate=0
        self.candidate_accepted=False

    def forward(self,original49152,h320,prior_own_native_embedding320,
                frozen_native49152_embeddings,*,native_bpe_position):
        check(native_bpe_position==2 and
              original49152.shape==(1,NATIVE_VOCAB) and
              h320.shape==(1,HIDDEN_DIM) and
              prior_own_native_embedding320.shape==(1,HIDDEN_DIM) and
              frozen_native49152_embeddings.shape==(NATIVE_VOCAB,HIDDEN_DIM) and
              all(not x.requires_grad and torch.isfinite(x).all() for x in (
                  original49152,h320,prior_own_native_embedding320,
                  frozen_native49152_embeddings)),
              "one original 43M fullnative49152 pass & frozen 320D OWN BPE1")
        original=original49152.detach().float()[0]
        ids=torch.topk(original,TOP_K,sorted=True).indices
        emb=F.normalize(frozen_native49152_embeddings.detach().float().
                        index_select(0,ids),dim=-1,eps=1e-6)
        x=torch.cat([
            F.layer_norm(h320.detach().float(),(HIDDEN_DIM,)),
            F.layer_norm(prior_own_native_embedding320.detach().float(),
                         (HIDDEN_DIM,))
        ],dim=-1)
        check(x.shape==(1,INPUT_DIM),"own 640 feature pair")
        # UNIQUE Z49 modification versus Z45: center by unlabeled TRAIN
        # source-specific mean; never center by held-out or DEV features.
        x=x-self.train_context_mean
        query=(x@self.input_project)@self.output_project
        raw=emb@query[0]
        centered=raw-raw.mean()
        rms=centered.square().mean().clamp_min(
            CENTERED_CANDIDATE_RMS_FLOOR**2).sqrt()
        squashed=torch.tanh((centered/rms)/CENTERED_PRE_TANH_DIVISOR)
        z=squashed-squashed.mean()
        residual=LOGIT_RESIDUAL_BOUND*z/z.abs().max().clamp_min(1.0)
        result=original.index_select(0,ids)+residual
        check(result.shape==(TOP_K,) and
              bool(torch.isfinite(result).all()) and
              float(residual.abs().max().detach())<=LOGIT_RESIDUAL_BOUND+1e-5,
              "old Z45 targetblind native top64 and bounded centered RMS tanh")
        return ids,result,residual

def fit_one_fold(records,embedding,*,heldout,wrong):
    check(heldout in FOLDS and type(wrong) is bool and len(records)==84,
          "fixed 7-kind LODO and matched human vs WRONG17")
    fit=[r for r in records if r["kind"]!=heldout]
    check(len(fit)==72 and
          len({r["source_id"] for r in fit})==72 and
          all(r["kind"]!=heldout for r in fit),
          "72 old legal TRAIN ONLY, one kind completely unseen by shadow")
    center=paired_context(fit)
    eligible=[r for r in fit if z45.eligible(r)]
    check(MIN_ELIGIBLE<=len(eligible)<=49,
          "same 49 overall legal source and min31 6-domain fold")
    torch.manual_seed(SEED)
    net=ContextCenteredNativeBPE2Comparator(center)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    steps=0
    for _ in range(EPOCHS):
        for row in eligible:
            opt.zero_grad(set_to_none=True)
            loss,meta=net.train_only_loss(
                row["native_logits"],row["h320"],row["prev_own_embed"],
                embedding,gold_second=row["gold_second"],
                own_first_correct=True,split="TRAIN",wrong_control=wrong)
            check(meta["eligible"] and loss.requires_grad and
                  bool(torch.isfinite(loss)),
                  "on-policy original correct BPE1+gold in targetblind original64")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            steps+=1
    net.eval()
    check(sum(p.numel() for p in net.parameters())==SHADOW_PARAMS==30720 and
          net.train_context_mean.requires_grad is False and
          torch.equal(net.train_context_mean,center) and
          net.output_gate==0 and net.candidate_accepted is False and
          all(torch.isfinite(p).all() for p in net.parameters()) and
          steps==EPOCHS*len(eligible),
          "no gradient to center or original S142; separate tiny shadow only")
    return net,{
        "fit_previous_legal72":72,
        "heldout_previous_legal12":12,
        "eligible_training_samples":len(eligible),
        "fixed_shadow_epochs":EPOCHS,
        "shadow_optimizer_steps":steps,
        "wrong17_training_labels":wrong,
        "new_train_center_unlabeled_only":True,
        "heldout_kind_never_in_center_or_optimizer":True,
        "original43M_GT12_Z10_optimizer_steps":0,
        "candidate_accepted":False,"output_gate":0,
    }

def score_previous_legal84(records,embedding):
    z45.validate(records,embedding)
    results={}
    for label,wrong in (("CENTER_HUMAN",False),("CENTER_WRONG17",True)):
        folds=[]
        predictions=[]
        for kind in FOLDS:
            ranker,training=fit_one_fold(records,embedding,heldout=kind,
                                          wrong=wrong)
            heldout=z45.evaluate_heldout(ranker,records,embedding,heldout=kind)
            check(len(heldout)==12, "12 previous LEGAL TRAIN for each heldout kind")
            predictions+=heldout
            folds.append({
                "kind":kind,
                "train":training,
                "first2_correct":sum(x["correct_first2_guarded"] for x in heldout),
                "second_distinct_native_IDs":len({x["chosen_BPE2_id"] for x in heldout}),
            })
        check(len(predictions)==84 and
              len({x["source_id"] for x in predictions})==84,
              "full seven folds each prior LEGAL84 once, original backbone frozen")
        original=[x["original_BPE2_id"] for x in predictions]
        adapted=[x["chosen_BPE2_id"] for x in predictions]
        stats={
            "old_LEGAL_TRAIN84":84,
            "correct_first_native":sum(x["first_correct"] for x in predictions),
            "original_correct_first2":sum(x["correct_first2_original"] for x in predictions),
            "centered_correct_first2":sum(x["correct_first2_guarded"] for x in predictions),
            "centered_first2_raw":sum(x["correct_first2_raw"] for x in predictions),
            "original_native_second_distinct_IDs":len(set(original)),
            "centered_native_second_distinct_IDs":len(set(adapted)),
            "original_crossprompt_empirical_entropy_bits":z45.entropy(original),
            "centered_crossprompt_empirical_entropy_bits":z45.entropy(adapted),
            "guard_nonzero":sum(x["guard_alpha"]>0 for x in predictions),
            "guarded_changed_original_second":sum(x["shadow_changed_original_BPE2"]
                                                   for x in predictions),
        }
        check(stats["old_LEGAL_TRAIN84"]==84 and
              stats["correct_first_native"]==77 and
              stats["original_correct_first2"]==1 and
              stats["original_native_second_distinct_IDs"]==14 and
              stats["original_crossprompt_empirical_entropy_bits"]==3.270078,
              "source-pinned 43M Z43/Z45 original TRAIN baseline parity")
        results[label]={"folds":folds,"metrics":stats}
    human=results["CENTER_HUMAN"]["metrics"]
    wrong=results["CENTER_WRONG17"]["metrics"]
    criteria={
        "centered_human_first2_at_least_6_84":
            human["centered_correct_first2"]>=MIN_CORRECT_FIRST2,
        "centered_human_first2_wrong17_plus3":
            human["centered_correct_first2"]-wrong["centered_correct_first2"]>=
                MIN_HUMAN_MINUS_WRONG_FIRST2,
        "centered_human_second_distinct_retain_at_least_75percent_original":
            human["centered_native_second_distinct_IDs"]>=
                math.ceil(human["original_native_second_distinct_IDs"]*
                          MIN_DISTINCT_FRACTION),
        "centered_human_crossprompt_entropy_drop_no_more_than_0_25_bits":
            human["centered_crossprompt_empirical_entropy_bits"]>=
            human["original_crossprompt_empirical_entropy_bits"]+
                MIN_EMPIRICAL_ENTROPY_DELTA_BITS-1e-8,
    }
    return {
        "stage":STAGE,
        "fitted_centered_prompt_640_source_only_previous_legal_72_each_fold":True,
        "train_only_old_LEGAL84_out_of_fold_not_independent":True,
        "base_original43M_shadow_and_Z10_A_unmodified":True,
        "original43M_added_decoder_forward_during_shadow_training":0,
        "old_Z47_SPENT_DEV28_not_loaded_or_scored":True,
        "trainable_params_unchanged_from_Z45":30720,
        "precommitted_internal_gates":criteria,
        "all_precommitted_internal_gates_pass":all(criteria.values()),
        "comparison":results,
        "shadow_fitted_checkpoints_not_saved":True,
        "no_new_DEV_or_J7_EVAL128_GOLD_CPU_bpp_certification":True,
        "candidate_accepted":False,"output_gate":0,
    }
