"""Z50 TRAIN-only guarded hard top-1 marginal contrast.
Same Z49 shadow, CE training, folds, loss weights and candidate slate.
Replace soft marginal forward values with 95% hard argmax + 5% soft,
with a temperature .25 straight-through surrogate for the shadow.
Use the actual inference entropy/KL guard scale as a stopped gradient.
No new or spent DEV, no pretrained updates, no promotion.
"""
from __future__ import annotations
from collections import Counter
from math import log, ceil
import torch
from torch.nn import functional as F
from research.calibri_nra3 import s119q2i8z45_trainonly_onpolicy_bpe2_domain_holdout as z45
from research.calibri_nra3.s119q2i8z37_native_BPE2_centered_RMS_reranker import (
    Position2NativeCandidateComparator,SHADOW_PARAMS,
)
from research.calibri_nra3.s119q2i8z40_causal_BPE234_entropy_guard import (
    CausalBPE234CandidateComparator,
)

STAGE="S119Q2I8Z50_REAL43M_TRAIN84_BPE2_GUARDED_HARD_TOP1_MARGINAL_7DOMAIN_FOLDS"
EPOCHS=8
SEED=119145
LR=.004
WEIGHT_DECAY=.04
TOP_K=64
NATIVE_VOCAB=49152
MARGINAL_KL_WEIGHT=.8
MARGINAL_ENTROPY_DEFICIT_WEIGHT=.8
MARGINAL_ENTROPY_TOLERANCE_BITS=.10
EPS=1e-9
HARD_MASS=.95
SURROGATE_TEMPERATURE=.25
DOMAINS=z45.DOMAINS
MODES=z45.MODES
MIN_FIRST2_HUMAN=6
MIN_FIRST2_GAIN_VS_WRONG=3
MIN_DISTINCT_FRACTION=.75
MIN_CROSSPROMPT_ENTROPY_CHANGE_BITS=-.25

def require(value,why):
    if not value:raise ValueError("Z50 TRAIN-only original marginal contrast fail-closed: "+why)

def native_marginal_penalty(original_probs,corrected_probs):
    """Marginal KL and entropy loss strictly over ORIGINAL top64 per TRAIN.

    Native token candidate slates stay target blind. Probabilities are
    differentiable only through shadow corrected logits, not original
    weights, and no gold token enters this loss. Every input pair is
    a (original-native-top64 ID list, reference probabilities, updated
    probabilities) from previously LEGAL TRAIN; never DEV.
    """
    require(len(original_probs)==len(corrected_probs)>=2,
            "two or more prior TRAIN-only original candidate sets")
    baseline=torch.zeros(NATIVE_VOCAB)
    modified=torch.zeros(NATIVE_VOCAB)
    for (ids,p),(ids2,q) in zip(original_probs,corrected_probs):
        require(ids.shape==ids2.shape==p.shape==q.shape==(TOP_K,) and
                torch.equal(ids,ids2) and ids.unique().numel()==TOP_K and
                int(ids.min())>=0 and int(ids.max())<NATIVE_VOCAB and
                not p.requires_grad and q.requires_grad and
                bool(torch.isfinite(p).all()) and bool(torch.isfinite(q).all()) and
                abs(float(p.detach().sum())-1.)<1e-5 and
                abs(float(q.detach().sum())-1.)<1e-5,
                "targetblind original top64 probabilities, frozen original and learnable shadow")
        baseline=baseline.scatter_add(0,ids,p)
        modified=modified.scatter_add(0,ids,q)
    baseline=baseline/len(original_probs)
    modified=modified/len(original_probs)
    ensure_base=baseline.clamp_min(EPS)
    ensure_mod=modified.clamp_min(EPS)
    kl=(modified*(ensure_mod.log()-ensure_base.log())).sum()
    base_entropy=-(baseline*ensure_base.log()).sum()
    modified_entropy=-(modified*ensure_mod.log()).sum()
    entropy_shortfall=F.relu(base_entropy-modified_entropy-
                             MARGINAL_ENTROPY_TOLERANCE_BITS*log(2.))
    loss=MARGINAL_KL_WEIGHT*kl+MARGINAL_ENTROPY_DEFICIT_WEIGHT*entropy_shortfall
    require(bool(torch.isfinite(loss)) and loss.requires_grad and
            float(loss.detach())>=-1e-6,"differentiable positive marginal KL/entropy loss")
    return loss,{
        "original_train_marginal_entropy_bits":round(float(base_entropy.detach())/log(2.),6),
        "shadow_train_marginal_entropy_bits":round(float(modified_entropy.detach())/log(2.),6),
        "original_train_marginal_KL_to_shadow":round(float(kl.detach()),6),
        "train_marginal_penalty":round(float(loss.detach()),6),
    }

def hard_marginal_probs(scores):
    """Hard-forward, smooth-backward selection; no stochastic decoding.
    The 5% detached soft component avoids a wholly singular reference.
    Gradient is a surrogate, not a derivative of argmax.
    """
    require(scores.shape==(TOP_K,) and bool(torch.isfinite(scores).all()),
            "finite original top64 selection scores")
    soft=F.softmax(scores,dim=0)
    surrogate=F.softmax(scores/SURROGATE_TEMPERATURE,dim=0)
    hard=F.one_hot(scores.detach().argmax(),num_classes=TOP_K).to(scores.dtype)
    forward=HARD_MASS*hard+(1.-HARD_MASS)*soft.detach()
    return forward+(surrogate-surrogate.detach())

def one_group_marginal_loss(net,legal,embedding):
    original_probs=[]
    changed_probs=[]
    for rec in legal:
        ids,scores,_=net(rec["native_logits"],rec["h320"],
                         rec["prev_own_embed"],embedding,native_bpe_position=2)
        baseline=rec["native_logits"][0].index_select(0,ids).detach().float()
        # Train the distribution of actual guarded selections, not merely
        # per-prompt soft logits. Guard itself is target-blind and detached.
        guarded,alpha,_=CausalBPE234CandidateComparator.guarded_scores(baseline,scores)
        effective=baseline+(scores-baseline)*alpha
        require(torch.allclose(effective.detach(),guarded,atol=1e-6,rtol=1e-6),
                "straight-through score must match inference guard")
        ref=hard_marginal_probs(baseline).detach()
        q=hard_marginal_probs(effective)
        original_probs.append((ids,ref))
        changed_probs.append((ids,q))
    return native_marginal_penalty(original_probs,changed_probs)

def fit_heldout_domain(records,embedding,*,heldout,wrong):
    require(heldout in DOMAINS and type(wrong) is bool,
            "exact one seven-domain holdout and matched WRONG17 control")
    train=[r for r in records if r["kind"]!=heldout]
    require(len(train)==72 and all(r["kind"]!=heldout for r in train),
            "72 old legal TRAIN, never update shadow on heldout 12")
    legal=[r for r in train if z45.eligible(r)]
    require(31<=len(legal)<=49,
            "minimum eligible genuine OWN-first and targetblind original-native top64")
    torch.manual_seed(SEED)
    net=Position2NativeCandidateComparator()
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    ce_updates=0
    marginal_updates=0
    last_metrics={}
    for _ in range(EPOCHS):
        # Matched to Z45: each eligible old TRAIN example receives one
        # genuine on-policy supervised loss per epoch; same WRONG17 control.
        for rec in legal:
            opt.zero_grad(set_to_none=True)
            loss,meta=net.train_only_loss(
                rec["native_logits"],rec["h320"],rec["prev_own_embed"],embedding,
                gold_second=rec["gold_second"],own_first_correct=True,
                split="TRAIN",wrong_control=wrong)
            require(meta["eligible"] and loss.requires_grad and
                    bool(torch.isfinite(loss)),
                    "train-only exact old native target after correct OWN BPE1")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            ce_updates+=1
        opt.zero_grad(set_to_none=True)
        preserve,last_metrics=one_group_marginal_loss(net,legal,embedding)
        preserve.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        marginal_updates+=1
    net.eval()
    require(ce_updates==EPOCHS*len(legal) and marginal_updates==EPOCHS and
            sum(p.numel() for p in net.parameters())==SHADOW_PARAMS==30720 and
            net.output_gate==0 and net.candidate_accepted is False and
            all(bool(torch.isfinite(p).all()) for p in net.parameters()),
            "only 30,720 shadow params updated; no original/GT12/Jessica modifications")
    return net,{
        "old_legal_TRAIN72":72,
        "shadow_fit_legal_old_TRAIN_ownprefix_native_BPE2":len(legal),
        "same_label_control_wrong_offset17":wrong,
        "supervised_OLD_TRAIN_only_updates":ce_updates,
        "old_TRAIN_marginal_preserve_updates":marginal_updates,
        "last_train_marginal_diagnostics":last_metrics,
        "original43M_and_GT12_optimizer_steps":0,
        "heldout_kind_shadow_optimizer_steps":0,
        "DEV_optimizer_steps":0,
        "candidate_accepted":False,"output_gate":0
    }

def cross_domain(records,embedding):
    z45.validate(records,embedding)  # Pins Z43 77 correct first, 49 first2 candidate and 1 greedy.
    result={}
    for mode in MODES:
        wrong=mode==MODES[1]
        folds=[]
        predictions=[]
        for kind in DOMAINS:
            ranker,fit=fit_heldout_domain(
                records,embedding,heldout=kind,wrong=wrong)
            holdout=z45.evaluate_heldout(ranker,records,embedding,heldout=kind)
            require(len(holdout)==12 and all(r["kind"]==kind for r in holdout),
                    "old LEGAL 12 of same domain never included in tiny shadow optimizer")
            folds.append({"heldout_kind":kind,"train_meta":fit,
                          "N":12,
                          "first2_human_or_wrong":sum(r["correct_first2_guarded"]
                                                      for r in holdout)})
            predictions.extend(holdout)
        require(len(predictions)==84 and
                len({r["source_id"] for r in predictions})==84,
                "each previous legal TRAIN sample held out exactly once")
        orig=[r["original_BPE2_id"] for r in predictions]
        shadow=[r["chosen_BPE2_id"] for r in predictions]
        first2=sum(r["correct_first2_guarded"] for r in predictions)
        correct_base=sum(r["correct_first2_original"] for r in predictions)
        first=sum(r["first_correct"] for r in predictions)
        reach=sum(r["first_correct"] and r["target_in_original_top64"] for r in predictions)
        require(correct_base==1 and first==77 and reach==49 and
                len(set(orig))==14,
                "Z43 reference exact own-first true native original S142 TRAIN84")
        metrics={
            "old_train84_heldout_shadow_first2_correct":first2,
            "original_native_greedy_first2_correct":correct_base,
            "original_native_greedy_BPE2_distinct_IDs":len(set(orig)),
            "original_native_greedy_BPE2_entropy_bits":z45.entropy(orig),
            "marginal_shadow_BPE2_distinct_IDs":len(set(shadow)),
            "marginal_shadow_BPE2_entropy_bits":z45.entropy(shadow),
            "shadow_changed_original_BPE2":sum(a!=b for a,b in zip(orig,shadow)),
            "shadow_guard_nonzero":sum(r["guard_alpha"]>0 for r in predictions),
            "old_TRAIN_gold_BPE2_in_native_TOP64_after_correct_own_first":reach,
        }
        result[mode]={"folds":folds,"metrics":metrics,"per_prior_legal_TRAIN":predictions}
    true=result[MODES[0]]["metrics"]
    wrong=result[MODES[1]]["metrics"]
    orig_entropy=true["original_native_greedy_BPE2_entropy_bits"]
    internal_gates={
        "human_correct_first2_at_least_6_old_TRAIN":
            true["old_train84_heldout_shadow_first2_correct"]>=MIN_FIRST2_HUMAN,
        "human_first2_gt_wrong_by_at_least_3":
            true["old_train84_heldout_shadow_first2_correct"]-
                wrong["old_train84_heldout_shadow_first2_correct"]>=
                MIN_FIRST2_GAIN_VS_WRONG,
        "human_distinct_second_at_least_75percent_original":
            true["marginal_shadow_BPE2_distinct_IDs"]>=
                ceil(true["original_native_greedy_BPE2_distinct_IDs"]*
                     MIN_DISTINCT_FRACTION),
        "human_empirical_crossprompt_entropy_no_worse_than_original_minus0_25":
            true["marginal_shadow_BPE2_entropy_bits"]>=
                orig_entropy+MIN_CROSSPROMPT_ENTROPY_CHANGE_BITS-1e-6,
    }
    return {
        "stage":STAGE,
        "shadow_trainonly_marginal_regularization_fixed":{
            "old_legal_TRAIN84":84,"seven_heldout_kinds":7,
            "eligible_native_BPE2_original_top64_after_correct_first":49,
            "rank32_shadow_params":SHADOW_PARAMS,
            "CE_epochs":EPOCHS,"CE_lr":LR,"CE_weight_decay":WEIGHT_DECAY,
            "extra_train_only_marginal_step_per_epoch":1,
            "native_original_slate_topK":TOP_K,
            "hard_argmax_forward_mass":HARD_MASS,
            "straight_through_surrogate_temperature":SURROGATE_TEMPERATURE,
            "actual_inference_guard_scale_stopped_gradient":True,
            "marginal_KL_weight":MARGINAL_KL_WEIGHT,
            "marginal_entropy_deficit_weight":MARGINAL_ENTROPY_DEFICIT_WEIGHT,
            "marginal_entropy_tolerance_bits":MARGINAL_ENTROPY_TOLERANCE_BITS,
            "targetblind_no_gold_candidate_injection":True,
            "no_DEV_training_and_no_spent_Z46_DEV28_evaluation":True,
            "correct_first_BPE_only_native_BPE2_rerank":True,
        },
        "comparison_HUMAN_vs_wrong17_crossdomain_oldTRAIN84":result,
        "precommitted_training_only_diagnostic_gates":internal_gates,
        "all_training_only_diagnostic_gates_pass":all(internal_gates.values()),
        "original_model_43M_native_forward_calls_from_cached_train_states":0,
        "original43M_GT12_Core_NRA3_NPA1_Z10_optimizer_steps":0,
        "old_SPENT_Z46_DEV28_never_fitted_or_rescored":True,
        "no_sealed_J7_EVAL128_GOLD_bpp_CB59_CPU_certification":True,
        "not_independent_external_generalization":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"TRAIN_ONLY_GUARDED_HARD_TOP1_MARGINAL_DIAGNOSTIC_NO_PROMOTION",
    }
