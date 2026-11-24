"""Z51 fixed confidence abstention on old TRAIN84.
Exactly Z45 training, no new parameters or optimizer changes.
After target-blind entropy/KL guard, only replace original greedy if
the winning candidate exceeds every other guarded score by .25 logits.
Otherwise return original greedy. No gold or domain IDs in the decision.
Seven domain folds, HUMAN/WRONG17, fixed original native top64.
No new/spent DEV, no promotion.
"""
from __future__ import annotations
from collections import Counter
from math import log2
import torch
from research.calibri_nra3.s119q2i8z37_native_BPE2_centered_RMS_reranker import (
    Position2NativeCandidateComparator,SHADOW_PARAMS,
    WRONG_OFFSET17_CONTROL,
)
from research.calibri_nra3.s119q2i8z40_causal_BPE234_entropy_guard import (
    CausalBPE234CandidateComparator,
)
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS

SEED=119145
EPOCHS=8
LR=.004
WEIGHT_DECAY=.04
NATIVE_K=64
NATIVE_VOCAB=49152
HIDDEN=320
ABSTENTION_MARGIN=.25
MODES=("HUMAN_OWNFIRST_TRAIN_ONLY","WRONG17_OWNFIRST_TRAIN_CONTROL")
DOMAINS=tuple(KINDS)
DISTINCT_BPE2_MIN_RETAIN_ORIGINAL=.75
REQUIRE_HUMAN_BPE2_GT_WRONG_AND_ORIGINAL=False  # report, never auto-accept
SOURCE_SHA="a66833e8f4e83e4ba080dc3db0b3032a3315bcfc2fe7cee473e7e9ea7569023e"

def check(cond,why):
    if not cond:raise ValueError("Z51 legal84 on-policy frozen BPE2 fail-closed: "+why)

def eligible(record):
    orig=record["native_logits"]
    return bool(record["own_first_correct"] and
        (torch.topk(orig[0],NATIVE_K,sorted=True).indices==
            record["gold_second"]).any())

def validate(records,embeddings):
    check(len(records)==84 and len({r["source_id"] for r in records})==84
          and embeddings.shape==(NATIVE_VOCAB,HIDDEN)
          and not embeddings.requires_grad and
          Counter(r["kind"] for r in records)==Counter({d:12 for d in DOMAINS})
          and all(r["split"]=="TRAIN" and
                  r["source_id"].startswith("S119Q2I8Z40_LEGAL_")
                  and r["native_logits"].shape==(1,NATIVE_VOCAB)
                  and r["h320"].shape==(1,HIDDEN)
                  and r["prev_own_embed"].shape==(1,HIDDEN)
                  and type(r["gold_second"]) is int
                  and type(r["own_first_correct"]) is bool
                  and 0<=r["gold_second"]<NATIVE_VOCAB
                  and all(not r[key].requires_grad for key in
                      ("native_logits","h320","prev_own_embed"))
                  for r in records),
          "84 legal prev TRAIN, 7 kinds x12, original native49352 frozen features")
    check(sum(eligible(r) for r in records)==49
          and sum(r["own_first_correct"] for r in records)==77,
          "strict Z43 physically observed previous TRAIN-only 49 top64 and 77 first")
    return True

def entropy(ids):
    if not ids:return 0.
    c=Counter(ids);n=len(ids)
    return round(-sum(v/n*log2(v/n) for v in c.values()),6)

def fit_without_domain(records,embedding,*,heldout,wrong):
    check(heldout in DOMAINS and type(wrong) is bool,
          "one precommitted domain and one exact-label or wrong17 control")
    train=[r for r in records if r["kind"]!=heldout]
    check(len(train)==72 and len({r["source_id"] for r in train})==72
          and all(r["kind"]!=heldout for r in train),
          "exact 6 domains = 72 OLD legal TRAIN, no fold leakage")
    accepted=[r for r in train if eligible(r)]
    check(31<=len(accepted)<=49,
          "minimum physical BPE2 eligible in 6-domain fit")
    torch.manual_seed(SEED)
    ranker=Position2NativeCandidateComparator()
    opt=torch.optim.AdamW(ranker.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    iterations=0
    for _ in range(EPOCHS):
        for rec in accepted:
            opt.zero_grad(set_to_none=True)
            loss,meta=ranker.train_only_loss(
                rec["native_logits"],rec["h320"],rec["prev_own_embed"],
                embedding,gold_second=rec["gold_second"],
                own_first_correct=True,split="TRAIN",wrong_control=wrong)
            check(meta["eligible"] and bool(torch.isfinite(loss)) and
                  loss.requires_grad,"true next BPE gradient on preceding OWN correct only")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(),1.)
            opt.step()
            iterations+=1
    ranker.eval()
    check(sum(p.numel() for p in ranker.parameters())==SHADOW_PARAMS==30720
          and ranker.output_gate==0 and ranker.candidate_accepted is False
          and all(torch.isfinite(p).all() for p in ranker.parameters())
          and iterations==EPOCHS*len(accepted),
          "30,720 shadow params only updated, never original43M")
    return ranker,{"train_previous_legal72":72,
                   "heldout_previous_legal12":12,
                   "eligible_train_for_gradient":len(accepted),
                   "epochs":EPOCHS,"optimizer_steps":iterations,
                   "wrong_label_offset17":wrong,
                   "original_43M_optimizer_steps":0,
                   "heldout_domain_optimizer_steps":0,
                   "original_GreenTunnel_optimizer_steps":0,
                   "candidate_accepted":False,"output_gate":0}

def selective_index(baseline,guarded):
    """Deterministic target-blind decision, never force a new token."""
    check(baseline.shape==guarded.shape==(NATIVE_K,) and
          bool(torch.isfinite(baseline).all()) and
          bool(torch.isfinite(guarded).all()),"finite same native top64")
    orig=int(baseline.argmax())
    winner=int(guarded.argmax())
    best=torch.topk(guarded,2).values
    margin=float(best[0]-best[1])
    chosen=winner if winner==orig or margin>=ABSTENTION_MARGIN else orig
    return chosen,margin,chosen!=winner

def evaluate_heldout(ranker,records,embedding,*,heldout):
    subset=[r for r in records if r["kind"]==heldout]
    check(len(subset)==12 and all(r["kind"]==heldout for r in subset),
          "exactly previously LEGAL withheld domain12, no new DEV")
    result=[]
    with torch.no_grad():
        for rec in subset:
            ids,scores,res=ranker(
                rec["native_logits"],rec["h320"],rec["prev_own_embed"],
                embedding,native_bpe_position=2)
            baseline=rec["native_logits"][0].index_select(0,ids).float()
            chosen_guard,alpha,delta=CausalBPE234CandidateComparator.guarded_scores(
                baseline,scores)
            original=rec["native_logits"][0]
            first=int(original.argmax().item())
            chosen_idx,margin,abstained=selective_index(baseline,chosen_guard)
            chosen=int(ids[chosen_idx].item())
            unselective=int(ids[chosen_guard.argmax()].item())
            raw=int(ids[scores.argmax()].item())
            target=rec["gold_second"]
            check(ids[0].item()==first and
                  (ids==chosen).any() and (ids==raw).any()
                  and 0<=alpha<=1. and abs(float(res.abs().max()))<=1.500001,
                  "original native top64 original greedy and bounded guarded human-blind")
            result.append({
              "source_id":rec["source_id"],"kind":heldout,
              "first_correct":rec["own_first_correct"],
              "original_BPE2_id":first,
              "chosen_BPE2_id":chosen,
              "unselective_guarded_BPE2_id":unselective,
              "unselective_correct_first2":rec["own_first_correct"] and unselective==target,
              "abstained":abstained,"guarded_top1_margin":margin,
              "raw_shadow_BPE2_id":raw,
              "correct_first2_original":rec["own_first_correct"] and first==target,
              "correct_first2_guarded":rec["own_first_correct"] and chosen==target,
              "correct_first2_raw":rec["own_first_correct"] and raw==target,
              "target_in_original_top64":bool((ids==target).any()),
              "guard_alpha":float(alpha),"guard_entropy_delta_bits":round(float(delta),6),
              "shadow_changed_original_BPE2":chosen!=first,
              "candidate_accepted":False,"output_gate":0,
            })
    return result

def cross_domain(records,embedding):
    validate(records,embedding)
    results={}
    for mode in MODES:
        wrong=mode==MODES[1]
        folds=[]
        all_rows=[]
        for domain in DOMAINS:
            model,meta=fit_without_domain(records,embedding,heldout=domain,wrong=wrong)
            heldout=evaluate_heldout(model,records,embedding,heldout=domain)
            all_rows+=heldout
            folds.append({"kind":domain,"fit":meta,
                          "n":len(heldout),
                          "correct_first2_guarded":sum(r["correct_first2_guarded"] for r in heldout),
                          "original_first2_correct":sum(r["correct_first2_original"] for r in heldout)})
        check(len(all_rows)==84 and len({r["source_id"] for r in all_rows})==84
              and sum(f["n"] for f in folds)==84,
              "each previous legal TRAIN predicted by ranker that NEVER fit own domain")
        ids=[r["chosen_BPE2_id"] for r in all_rows]
        original_ids=[r["original_BPE2_id"] for r in all_rows]
        main={
            "correct_first2_original":sum(r["correct_first2_original"] for r in all_rows),
            "correct_first2_guarded":sum(r["correct_first2_guarded"] for r in all_rows),
            "correct_first2_raw_shadow":sum(r["correct_first2_raw"] for r in all_rows),
            "unselective_correct_first2":sum(r["unselective_correct_first2"] for r in all_rows),
            "unselective_entropy_bits":entropy([r["unselective_guarded_BPE2_id"] for r in all_rows]),
            "abstained_count":sum(r["abstained"] for r in all_rows),
            "original_native_second_distinct_IDs":len(set(original_ids)),
            "heldout_second_distinct_IDs":len(set(ids)),
            "original_native_second_empirical_entropy_bits":entropy(original_ids),
            "heldout_second_empirical_entropy_bits":entropy(ids),
            "fraction_change_BPE2":round(sum(r["shadow_changed_original_BPE2"]
                                            for r in all_rows)/84,6),
            "guard_nonzero_count":sum(r["guard_alpha"]>0 for r in all_rows),
            "guard_mean_alpha":round(sum(r["guard_alpha"] for r in all_rows)/84,6),
            "first_correct":sum(r["first_correct"] for r in all_rows),
            "correct_first_and_gold_top64":sum(r["first_correct"] and
                   r["target_in_original_top64"] for r in all_rows),
        }
        check(main["correct_first2_original"]==1
              and main["original_native_second_distinct_IDs"]==14
              and main["first_correct"]==77
              and main["correct_first_and_gold_top64"]==49,
              "physical Z43 native legal TRAIN baseline must match independently")
        results[mode]={"folds":folds,"metrics":main,"per_source":all_rows}
    human=results[MODES[0]]["metrics"]
    wrong=results[MODES[1]]["metrics"]
    diversity=human["heldout_second_distinct_IDs"]>=int(human["original_native_second_distinct_IDs"]*DISTINCT_BPE2_MIN_RETAIN_ORIGINAL+0.999)
    # Deliberately no acceptance—even positive internal cross-domain results
    # require one independently authored NEW DEV prereg & physical test later.
    return {
        "stage":"S119Q2I8Z51_REAL43M_ONPOLICY_LEGAL84_BPE2_DOMAIN_OUT_FOLDS",
        "original_43M_native_forwards_from_frozen_cache_only":True,
        "original_S142_43M_and_GT12_child_optimizer_steps":0,
        "fixed_guarded_top1_margin":ABSTENTION_MARGIN,
        "shadow_train_EPOCHS":EPOCHS,"folds_by_kind":len(DOMAINS),
        "heldout_domain_is_previously_legal_TRAIN_NOT_independent":True,
        "no_SPENT_DEV_examples_or_label_optimizer":True,
        "no_GOLD_EVAL128_J7_CPU_bpp_promotion":True,
        "original_BPE2_greedy_first2_correct_on_previous_train":1,
        "human_vs_wrong_out_of_fold":results,
        "internal_informational_gate":{
            "human_correct_first2_gt_original":
                human["correct_first2_guarded"]>human["correct_first2_original"],
            "human_correct_first2_gt_wrong":
                human["correct_first2_guarded"]>wrong["correct_first2_guarded"],
            "human_distinct_second_75percent_original":diversity,
            "human_first2_at_least_6":human["correct_first2_guarded"]>=6,
            "human_beats_wrong17_by_3":human["correct_first2_guarded"]-wrong["correct_first2_guarded"]>=3,
            "human_entropy_within_quarter_bit":human["heldout_second_empirical_entropy_bits"]>=human["original_native_second_empirical_entropy_bits"]-.25-1e-6,
        },
        "candidate_accepted":False,"output_gate":0,
        "decision":"CROSS_DOMAIN_PREVIOUS_TRAIN_ONLY_DIAGNOSTIC_NEEDS_PRECOMMITTED_NEW_DEV_BEFORE_ANY_PROMOTION",
    }
