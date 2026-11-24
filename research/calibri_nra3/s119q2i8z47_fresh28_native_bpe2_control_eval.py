"""Z47 sealed-in-advance Z46 NEW28 internal DEV, train fit only old legal84.

No original model gradients, no DEV optimizer, no teacher forcing, no
target insertion into original native top64; guard is independent of DEV
gold. All modes share original BPE1/BPE2 logits; BPE3 computed on each
mode's TRUE naturally selected BPE2 self-generated prefix. Research gate0.
"""
from __future__ import annotations
import math
from collections import Counter
import torch
from research.calibri_nra3 import s119q2i8z46_fresh28_onpolicy_bpe2_preregister as z46
from research.calibri_nra3 import s119q2i8z45_trainonly_onpolicy_bpe2_domain_holdout as z45
from research.calibri_nra3.s119q2i8z37_native_BPE2_centered_RMS_reranker import (
    Position2NativeCandidateComparator,SHADOW_PARAMS,
)
from research.calibri_nra3.s119q2i8z40_causal_BPE234_entropy_guard import (
    CausalBPE234CandidateComparator,
)

MODES=("ORIGINAL_NATIVE_GREEDY","HUMAN_OWNFIRST_TRAIN_ONLY",
       "WRONG17_OWNFIRST_TRAIN_CONTROL")
def require(cond,msg):
    if not cond:raise ValueError("Z47 frozen Z46 DEV native2-3 eval: "+msg)

def fit_all_old84(records,embedding,*,wrong):
    z45.validate(records,embedding)
    require(type(wrong) is bool and len(records)==84,
            "fixed prior LEGAL84 onpolicy HUMAN or wrong17 control")
    train=[r for r in records if z45.eligible(r)]
    require(len(train)==49,"source-only genuine 49 eligible original TOP64")
    torch.manual_seed(z46.SEED)
    net=Position2NativeCandidateComparator()
    opt=torch.optim.AdamW(net.parameters(),lr=z46.TRAIN_LR,
                          weight_decay=z46.TRAIN_WEIGHT_DECAY)
    steps=0
    for _ in range(z46.TRAIN_EPOCHS):
        for record in train:
            opt.zero_grad(set_to_none=True)
            loss,meta=net.train_only_loss(
                record["native_logits"],record["h320"],record["prev_own_embed"],
                embedding,gold_second=record["gold_second"],
                own_first_correct=True,split="TRAIN",wrong_control=wrong)
            require(meta["eligible"] and loss.requires_grad and
                    bool(torch.isfinite(loss)),
                    "only eligible prior TRAIN and real original on-policy correct BPE1")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            steps+=1
    net.eval()
    require(steps==49*z46.TRAIN_EPOCHS==392 and
            SHADOW_PARAMS==z46.PARAMS==30720 and
            all(bool(torch.isfinite(p).all()) for p in net.parameters()) and
            net.output_gate==0 and net.candidate_accepted is False,
            "no original43M/GT12 optimizer, only 30,720-param shadow")
    return net,{"prior_legal_TRAIN84":84,
                "original_correct_first_BPE_TRAIN":77,
                "original_top64_gold_BPE2_after_correct_first":49,
                "shadow_optimizer_steps":steps,
                "wrong_offset17_target_label_control":wrong,
                "original_core_GT12_NRA3_NPA1_Z10_optimizer_steps":0,
                "DEV_optimizer_steps":0,
                "candidate_accepted":False,"output_gate":0}

@torch.no_grad()
def pick_second(original_logits,h320,first,embedding,ranker):
    """All input comes from the SAME original S142 real own-prefix readout."""
    require(original_logits.shape==(1,49152) and h320.shape==(1,320)
            and isinstance(first,int) and 0<=first<49152 and
            embedding.shape==(49152,320),
            "original native49152 logits, existing hidden320, true own previous native")
    greedy=int(original_logits[0].argmax().item())
    if ranker is None:
        return greedy,{"alpha":0.,"entropy_delta_bits":0.,
                       "changed":False,"candidate_original_top64":True}
    ids,score,residual=ranker(original_logits.detach(),h320.detach(),
                             embedding[first].view(1,-1).detach(),embedding,
                             native_bpe_position=2)
    orig=original_logits[0].index_select(0,ids)
    guarded,alpha,delta=CausalBPE234CandidateComparator.guarded_scores(orig,score)
    chosen=int(ids[guarded.argmax()].item())
    require(bool((ids==chosen).any()) and bool(torch.isfinite(guarded).all())
            and float(residual.abs().max())<=1.50001
            and alpha in (1.,.75,.5,.375,.25,.125,.0625,.03125,0.),
            "target blind original candidate only, bounded score and KL+entropy guard")
    return chosen,{"alpha":round(float(alpha),6),
                   "entropy_delta_bits":round(float(delta),6),
                   "changed":chosen!=greedy,
                   "candidate_original_top64":True}

def scores(rowset):
    """Score only AFTER three full native tokens were autonomously generated."""
    require(len(rowset)==z46.DEV_COUNT and
            len({r["source_id"] for r in rowset})==z46.DEV_COUNT,
            "exact 28 never-before-scored Z46 new DEV examples")
    base={}
    for mode in MODES:
        records=[x["results"][mode] for x in rowset]
        second_ids=[r["native_first3_ids"][1] for r in records
                    if len(r["native_first3_ids"])>=2]
        require(len(second_ids)==z46.DEV_COUNT,
                "all three arms generate their own native BPE2")
        base[mode]={
          "n":len(records),
          "correct_first_native":sum(r["correct_first"] for r in records),
          "correct_first2_native":sum(r["correct_first2"] for r in records),
          "correct_first3_native":sum(r["correct_first3"] for r in records),
          "premature_EOS_first3":sum(r["premature_EOS_first3"] for r in records),
          "native_second_distinct_IDs":len(set(second_ids)),
          "native_second_crossprompt_empirical_entropy_bits":z45.entropy(second_ids),
          "gold_second_in_original_top64_given_correct_first":sum(
              x["first_correct"] and x["original_second_gold_in_original_TOP64"]
              for x in rowset),
          "per_prompt_second_control_applied":sum(r["second"].get("changed",False)
              for r in records),
          "per_prompt_second_guard_nonzero":sum(r["second"].get("alpha",0.)>0
              for r in records),
        }
    original=base[MODES[0]]
    human=base[MODES[1]]
    wrong=base[MODES[2]]
    require(original["correct_first_native"]==human["correct_first_native"]==
            wrong["correct_first_native"],"BPE1 must be unchanged for all arms")
    d=human["native_second_distinct_IDs"]>=math.ceil(
        original["native_second_distinct_IDs"]*
        z46.MIN_DISTINCT_SECOND_FRACTION_VS_BASELINE)
    e=(human["native_second_crossprompt_empirical_entropy_bits"]-
       original["native_second_crossprompt_empirical_entropy_bits"])>=(
           z46.MIN_CROSSPROMPT_ENTROPY_DELTA_BITS-1e-8)
    metrics={
      "human_first2_at_least_4_of_28":
        human["correct_first2_native"]>=z46.MIN_HUMAN_DEV_FIRST2,
      "human_first2_gt_original_by_at_least_3":
        human["correct_first2_native"]-original["correct_first2_native"]>=
            z46.MIN_HUMAN_FIRST2_OVER_EACH_CONTROL,
      "human_first2_gt_wrong_by_at_least_3":
        human["correct_first2_native"]-wrong["correct_first2_native"]>=
            z46.MIN_HUMAN_FIRST2_OVER_EACH_CONTROL,
      "human_first3_at_least_one":
        human["correct_first3_native"]>=z46.MIN_HUMAN_DEV_FIRST3,
      "human_second_distinct_at_least_75percent_original":d,
      "human_crossprompt_entropy_drop_at_most_0_25_bits":e,
      "human_no_premature_EOS_first3":
        human["premature_EOS_first3"]<=z46.MAX_PREMATURE_EOS_FIRST3,
    }
    return {"modes":base,
            "precommitted_diagnostic_thresholds":metrics,
            "all_precommitted_nonpromoting_thresholds_pass":all(metrics.values()),
            "candidate_accepted":False,"output_gate":0,
            "NEW_DEV28_becomes_SPENT_once_scored":True,
            "no_first_BPE_or_third_BPE_reranking":True,
            "no_DEV_candidate_gold_injection_or_teacher_forcing":True,
            "original_real43M_unchanged":True}
