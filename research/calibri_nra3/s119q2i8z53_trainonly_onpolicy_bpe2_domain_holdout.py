"""Z53 EOPD-inspired bounded pilot: actual 135M teacher on own BPE1.
Same Z51 rank32, CE, selective guard, seven old TRAIN domain folds.
Four matched arms CE, WRONG17, CE+reverse KL, CE+entropy-aware KL.
Full native distribution; student logits outside original top64 fixed.
Teacher entropy >=2 nats activates extra forward KL. No DEV or promotion.
This is a fixed-prefix BPE2 pilot, not full sequence on-policy EOPD.
"""
from __future__ import annotations
from collections import Counter
from math import log2
import torch
from torch.nn import functional as F
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
MODES=("HUMAN_OWNFIRST_TRAIN_ONLY","WRONG17_OWNFIRST_TRAIN_CONTROL",
       "HUMAN_REVERSE_KL","HUMAN_ENTROPY_AWARE_KL")
KD_WEIGHT=.25
TEACHER_ENTROPY_THRESHOLD_NATS=2.
DOMAINS=tuple(KINDS)
DISTINCT_BPE2_MIN_RETAIN_ORIGINAL=.75
REQUIRE_HUMAN_BPE2_GT_WRONG_AND_ORIGINAL=False  # report, never auto-accept
SOURCE_SHA="a66833e8f4e83e4ba080dc3db0b3032a3315bcfc2fe7cee473e7e9ea7569023e"

def check(cond,why):
    if not cond:raise ValueError("Z53 legal84 on-policy frozen BPE2 fail-closed: "+why)

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

def teacher_loss(original,ids,scores,teacher_logits,*,adaptive):
    check(original.shape==teacher_logits.shape==(1,NATIVE_VOCAB) and
          ids.shape==scores.shape==(NATIVE_K,) and
          ids.unique().numel()==NATIVE_K and
          torch.equal(ids,torch.topk(original[0],NATIVE_K,sorted=True).indices) and
          not original.requires_grad and not teacher_logits.requires_grad and
          bool(torch.isfinite(teacher_logits).all()) and type(adaptive) is bool,
          "full native frozen teacher and targetblind candidate support")
    student=original.detach().float()[0].scatter(0,ids,scores)
    logq=F.log_softmax(student,dim=0)
    logp=F.log_softmax(teacher_logits.detach().float()[0],dim=0)
    p=logp.exp()
    q=logq.exp()
    h=-(p*logp).sum()
    reverse=(q*(logq-logp)).sum()
    forward=(p*(logp-logq)).sum()
    active=adaptive and float(h)>=TEACHER_ENTROPY_THRESHOLD_NATS
    loss=reverse+forward if active else reverse
    check(bool(torch.isfinite(loss)) and float(loss.detach())>=-1e-5,
          "finite full native teacher divergences")
    return loss,active

def fit_without_domain(records,embedding,*,heldout,wrong,mode):
    check(heldout in DOMAINS and type(wrong) is bool and mode in MODES and wrong==(mode==MODES[1]),
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
    teacher_updates=0
    high_entropy_updates=0
    for _ in range(EPOCHS):
        for rec in accepted:
            opt.zero_grad(set_to_none=True)
            loss,meta=ranker.train_only_loss(
                rec["native_logits"],rec["h320"],rec["prev_own_embed"],
                embedding,gold_second=rec["gold_second"],
                own_first_correct=True,split="TRAIN",wrong_control=wrong)
            check(meta["eligible"] and bool(torch.isfinite(loss)) and
                  loss.requires_grad,"true next BPE gradient on preceding OWN correct only")
            if mode in MODES[2:]:
                ids,scores,_=ranker(rec["native_logits"],rec["h320"],
                    rec["prev_own_embed"],embedding,native_bpe_position=2)
                kd,active=teacher_loss(rec["native_logits"],ids,scores,
                    rec["teacher_logits"],adaptive=mode==MODES[3])
                loss=loss+KD_WEIGHT*kd
                teacher_updates+=1
                high_entropy_updates+=int(active)
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
                   "mode":mode,"teacher_KL_updates":teacher_updates,
                   "high_entropy_forward_KL_updates":high_entropy_updates,
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
    check(all(r["teacher_logits"].shape==(1,NATIVE_VOCAB) and
              not r["teacher_logits"].requires_grad and
              bool(torch.isfinite(r["teacher_logits"]).all()) for r in records),
          "84 genuine frozen teacher own-prefix states")
    teacher_audit=[]
    for r in records:
        logp=F.log_softmax(r["teacher_logits"][0],dim=0)
        p=logp.exp()
        ids=torch.topk(r["native_logits"][0],NATIVE_K).indices
        teacher_audit.append({
            "source_id":r["source_id"],
            "correct_first2":r["own_first_correct"] and int(p.argmax())==r["gold_second"],
            "entropy_nats":float(-(p*logp).sum()),
            "mass_in_student_top64":float(p[ids].sum()),
            "teacher_gold_probability":float(p[r["gold_second"]]),
        })
    results={}
    for mode in MODES:
        wrong=mode==MODES[1]
        folds=[]
        all_rows=[]
        for domain in DOMAINS:
            model,meta=fit_without_domain(records,embedding,heldout=domain,wrong=wrong,mode=mode)
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
    adaptive=results[MODES[3]]["metrics"]
    reverse=results[MODES[2]]["metrics"]
    diversity=human["heldout_second_distinct_IDs"]>=int(human["original_native_second_distinct_IDs"]*DISTINCT_BPE2_MIN_RETAIN_ORIGINAL+0.999)
    # Deliberately no acceptance—even positive internal cross-domain results
    # require one independently authored NEW DEV prereg & physical test later.
    return {
        "stage":"S119Q2I8Z53_REAL43M_ONPOLICY_LEGAL84_BPE2_DOMAIN_OUT_FOLDS",
        "original_43M_native_forwards_from_frozen_cache_only":True,
        "original_S142_43M_and_GT12_child_optimizer_steps":0,
        "fixed_guarded_top1_margin":ABSTENTION_MARGIN,
        "teacher_KD_weight":KD_WEIGHT,
        "teacher_entropy_threshold_nats":TEACHER_ENTROPY_THRESHOLD_NATS,
        "teacher_audit_old_TRAIN84":teacher_audit,
        "teacher_correct_first2_on_student_prefix":sum(r["correct_first2"] for r in teacher_audit),
        "teacher_high_entropy_count":sum(r["entropy_nats"]>=TEACHER_ENTROPY_THRESHOLD_NATS for r in teacher_audit),
        "teacher_mean_mass_in_student_top64":sum(r["mass_in_student_top64"] for r in teacher_audit)/84,
        "shadow_train_EPOCHS":EPOCHS,"folds_by_kind":len(DOMAINS),
        "heldout_domain_is_previously_legal_TRAIN_NOT_independent":True,
        "no_SPENT_DEV_examples_or_label_optimizer":True,
        "no_GOLD_EVAL128_J7_CPU_bpp_promotion":True,
        "original_BPE2_greedy_first2_correct_on_previous_train":1,
        "human_vs_wrong_out_of_fold":results,
        "adaptive_diagnostic_gates":{
            "first2_at_least_6":adaptive["correct_first2_guarded"]>=6,
            "beats_wrong_by_3":adaptive["correct_first2_guarded"]-wrong["correct_first2_guarded"]>=3,
            "distinct_at_least_75pct":adaptive["heldout_second_distinct_IDs"]>=11,
            "entropy_within_quarter_bit":adaptive["heldout_second_empirical_entropy_bits"]>=3.020078-1e-6,
            "no_first2_regression_vs_CE":adaptive["correct_first2_guarded"]>=human["correct_first2_guarded"],
            "no_entropy_regression_vs_CE":adaptive["heldout_second_empirical_entropy_bits"]>=human["heldout_second_empirical_entropy_bits"]-1e-6,
            "entropy_beats_reverse_KL":adaptive["heldout_second_empirical_entropy_bits"]>reverse["heldout_second_empirical_entropy_bits"],
        },
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
