"""S119q2i8r: physically train ONLY existing 49152-native rank4 readout.

STRICT 34 TRAIN items only: 28 still-unused author E70 multiword, six old
TRAIN rehearsal. Both same-initial-state A/B frozen original full43M,
12 Green Tunnel gates, NRA3, NPA1, former 53,751 S115, projector.
Exactly 34 equal Adam head-only updates; matched A full gold-native CE
at all BPE positions (incl genuine EOS), B identical CE PLUS frozen
TRAIN-only learnable differentiable zero-margin EOS-vs-gold ranking loss.
NO manual inference mask, no fixed EOS suppression, no new neural area,
no early stopping on any DEV, no teacher-on-DEV, no production.
Both run actual rank4 learned head gain=1, motivated by *prior* EOS
mechanistic ablation #951/#953; predeclared gain/loss no posthoc tuning.
"""
from __future__ import annotations
from collections import Counter
import torch
from torch.nn import functional as F

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    real_joint_native,
)
from research.calibri_nra3.s119q2i8o_native_head_gain_physical_ablation import (
    scoped_exact_native_gain,
)

STAGE="CALIBRI_S119Q2I8R_34_TRAIN_NATIVE_HEAD_RANK4_EOS_MARGIN_VS_CE_REAL43M"
VOCAB=49152
GOLD_EOS=2
N_NEW=28
N_OLD=6
EXPECTED_TRAIN=N_NEW+N_OLD
ROUNDS=1
GAIN=1.0
MARGIN_LAMBDA=.25
HEAD_UPDOWN_LR=.00035
HEAD_BIAS_LR=.00002
CLIP=.5
TRAINABLE_HEAD=320*4+49152*4+49152


def eos_margin_loss(logits,gold_native,eos_native,*,is_last_gold_BPE):
    """TRAIN-only gold target; no manual EOS suppression in inference.

    At non-EOS human target we learn how to prefer correct gold over EOS;
    at genuine terminal human EOS we learn how to prefer EOS over the
    strongest other native token. All 49152 logits are physical.
    """
    check(logits.shape==(1,VOCAB)
          and isinstance(gold_native,int) and 0<=gold_native<VOCAB
          and type(eos_native) is int and eos_native==GOLD_EOS
          and isinstance(is_last_gold_BPE,bool)
          and (is_last_gold_BPE==(gold_native==GOLD_EOS)),
          "only real native full49k TRAIN human targets, EOS last exactly")
    eos=logits[0,eos_native]
    if is_last_gold_BPE:
        other=torch.cat((logits[0,:eos_native],
                         logits[0,eos_native+1:]))
        competitor=torch.max(other)
        val=F.softplus(competitor-eos)
    else:
        val=F.softplus(eos-logits[0,gold_native])
    check(bool(torch.isfinite(val)) and val.ndim==0,
          "train-only finite EOS native ranking objective")
    return val


def head_only_train(model,sources,bridge,full,expansion,child,train,tok,
                    *,objective):
    check(objective in ("FULL_NATIVE_CE","FULL_NATIVE_CE_PLUS_EOS_MARGIN")
          and len(train)==EXPECTED_TRAIN
          and Counter(x["split"] for x in train)=={"TRAIN":EXPECTED_TRAIN}
          and sum(x["id"].startswith("S119Q2I8E_") for x in train)==N_NEW
          and sum(not x["id"].startswith("S119Q2I8E_") for x in train)==N_OLD
          and tok.vocab_size==VOCAB and tok.eos_token_id==GOLD_EOS
          and ROUNDS==1 and GAIN==1. and MARGIN_LAMBDA==.25,
          "exact 28 never-trained TRAIN + six old rehearsal no DEV leakage")
    child.requires_grad_(False)
    full.requires_grad_(False)
    head=child.language_head
    params=[head.down.weight,head.up.weight,head.output_bias]
    check(sum(p.numel() for p in params)==TRAINABLE_HEAD==247040
          and head.up.weight.shape==(VOCAB,4)
          and head.down.weight.shape==(4,320)
          and head.output_bias.shape==(VOCAB,)
          and head.output_gate==0 and child.output_gate==0
          and head.candidate_accepted is False
          and child.candidate_accepted is False,
          "existing rank4 physical readout exclusively; zero new params")
    for p in params:p.requires_grad_(True)
    # The existing onepass early-layer hook strictly requires the
    # injected adapter delta to be differentiable. Allow an old
    # 6x6 projector mix tensor as a *gradient-only* autograd anchor:
    # NO optimizer ever references it. Verify exact immutable weights.
    anchor=child.child.mix_logits
    check(anchor.numel()==36 and all(id(anchor)!=id(p) for p in params),
          "only existing old six-slot causal projector anchor")
    anchor.requires_grad_(True)
    check(sum(p.numel() for p in child.parameters() if p.requires_grad)==
          TRAINABLE_HEAD+anchor.numel()
          and all(not p.requires_grad for p in full.parameters()),
          "only 247040 HEAD weights optimized; old36 mix gradient-only no update")
    opt=torch.optim.Adam([
        {"params":[head.down.weight,head.up.weight],"lr":HEAD_UPDOWN_LR},
        {"params":[head.output_bias],"lr":HEAD_BIAS_LR},
    ])
    init={k:v.detach().clone() for k,v in head.state_dict().items()}
    immutable={k:v.detach().cpu().clone() for k,v in child.state_dict().items()
               if not k.startswith("language_head.")}
    whole_frozen_full={k:v.detach().cpu().clone() for k,v in full.state_dict().items()}
    calls=steps=margin_calls=0
    first_ce=[];last_eos_ce=[];all_ce=[]
    first_margin=[];last_margin=[]
    for row in train:
        native_gold=row["answer_native_with_eos"]
        check(2<=len(native_gold)<=40 and native_gold[-1]==GOLD_EOS
              and all(x!=GOLD_EOS for x in native_gold[:-1]),
              "author correct BPE full answer before genuine terminal EOS")
        opt.zero_grad(set_to_none=True)
        anchor.grad=None
        with scoped_exact_native_gain(head,GAIN) as witness:
            for t,target in enumerate(native_gold):
                target=int(target)
                native_prefix=row["prefix"]+native_gold[:t]
                check(native_prefix[:len(row["prefix"])]==row["prefix"]
                      and native_prefix[len(row["prefix"]):]==native_gold[:t]
                      and len(native_prefix)<160,
                      "only TRUE previous author TRAIN gold BPE enters causal original 43M")
                logits=real_joint_native(
                    model,sources,bridge,full,expansion,
                    child,native_prefix,row["prompt_features"],contextual=True)
                check(logits.shape==(1,VOCAB)
                      and full.last_forward_coverage==tuple(range(12))
                      and not full._active and full._carry is None,
                      "real pretrained original43M GT12 Core NRA3 NPA1 native49k forward")
                loss=F.cross_entropy(logits,
                         torch.tensor([target],dtype=torch.long))
                weighted=loss/len(native_gold)
                if objective=="FULL_NATIVE_CE_PLUS_EOS_MARGIN":
                    margin=eos_margin_loss(
                        logits,target,GOLD_EOS,
                        is_last_gold_BPE=t==len(native_gold)-1)
                    weighted=weighted+MARGIN_LAMBDA*margin/len(native_gold)
                    margin_calls+=1
                    (last_margin if t==len(native_gold)-1 else first_margin).append(
                        float(margin.detach()))
                check(bool(torch.isfinite(weighted)),
                      "full49k native differentiable train loss finite")
                weighted.backward()
                val=float(loss.detach())
                all_ce.append(val)
                if t==0:first_ce.append(val)
                if t==len(native_gold)-1:last_eos_ce.append(val)
                calls+=1
                del logits,loss,weighted
        check(witness[0]==len(native_gold),
              "exact original full 49k native head hook at all trained BPE positions")
        torch.nn.utils.clip_grad_norm_(params,CLIP)
        opt.step();steps+=1
        anchor.grad=None
        check(all(bool(torch.isfinite(p).all()) for p in params),
              "real rank4 neural head Adam finite")
    check(steps==EXPECTED_TRAIN and len(first_ce)==len(last_eos_ce)==EXPECTED_TRAIN
          and margin_calls==(calls if objective=="FULL_NATIVE_CE_PLUS_EOS_MARGIN" else 0)
          and calls>EXPECTED_TRAIN*3,
          "actual 34 identical update budgets, no untracked gradients")
    changes={k:float(torch.linalg.vector_norm(v.detach()-init[k]).item())
             for k,v in head.state_dict().items()}
    check(all(v>0 for v in changes.values()),
          "all three genuine head matrices/bias physically learn")
    check(all(torch.equal(v.detach().cpu(),immutable[k])
              for k,v in child.state_dict().items()
              if not k.startswith("language_head."))
          and all(torch.equal(v.detach().cpu(),whole_frozen_full[k])
                  for k,v in full.state_dict().items()),
          "all previous 43M, GT12, core/NRA3/NPA1/protected 53k weights unchanged")
    child.eval()
    head.requires_grad_(False)
    # The physical onepass causal hook REQUIRES an autograd-attached
    # early-layer delta also for honest OWN-token evaluation. Keep the
    # old 36-value mixer as gradient-enabled ONLY until that read-only
    # evaluation; it is explicitly excluded from ALL optimizer groups,
    # never updated and its immutable value is verified below.
    check(anchor.requires_grad and all(not p.requires_grad for p in params),
          "head updates finished; only frozen-value old mixer remains a grad anchor")
    check(torch.equal(anchor.detach().cpu(),immutable["child.mix_logits"]),
          "old 36-param 6slot projection anchor used for autograd ONLY, never updated")
    return {
        "stage":STAGE,
        "arm":objective,
        "physical_43M_full_native49152_teacher_forced_TRAIN_forwards":calls,
        "actual_matched_native_rank4_head_only_Adam_updates":steps,
        "trainable_parameters_original_native_head_ONLY":TRAINABLE_HEAD,
        "old_sixslot_mix_logits_36param_gradient_anchor_NOT_in_optimizer":True,
        "all_head_down_up_bias_changed_l2":changes,
        "mean_full_native_human_gold_CE":sum(all_ce)/len(all_ce),
        "mean_true_first_gold_native_CE":sum(first_ce)/len(first_ce),
        "mean_true_terminal_EOS_CE":sum(last_eos_ce)/len(last_eos_ce),
        "extra_eos_relative_softplus_on_all_TRAIN_gold_BPE":
            MARGIN_LAMBDA if objective=="FULL_NATIVE_CE_PLUS_EOS_MARGIN" else 0.,
        "real_additional_train_only_margin_loss_calls":margin_calls,
        "train_only_last_gold_EOS_supervised":True,
        "no_fixed_EOS_mask_or_logit_hack":True,
        "native_output_gain_locked_to_1":True,
        "learning_all_original49k_gold_BPE_EOS":True,
        "NRA3_NPA1_original43M_GT12_unchanged":True,
        "original_616J7_and_EVAL128_not_accessed":True,
        "candidate_accepted":False,"output_gate":0,
    }
