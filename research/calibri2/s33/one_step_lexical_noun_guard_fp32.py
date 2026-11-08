"""CALIBRI S33: protected FP32 36M second-NOUN learning, TRAIN-only.

S30 unrestricted FP32 learned some NEW TRAIN answers (0 -> 4/16) but lost
old correct TRAIN answers (6 -> 4/8). S31 24-step guard rolled back all
updates. S32 2-step PCGrad guard preserved old 6 (new 0/16, old 7/8) and
accepted 8 actual FP32 steps, but its new-noun ability did not improve.

S33 precommits one isolated, fixed 128-step *one-step* guarded research
experiment. Trains distinct noun answers sharing the same color with an
explicit lexical noun-vs-foil token margin and EOS margin. Each new full
dense-FP32 update projects against an old TRAIN gradient, then short
backtracking chooses the first candidate that improves the SAME TRAIN
task and preserves all six initially correct historical TRAIN answers
with exact greedy full answer+EOS. Candidate selection uses TRAIN only.

Not the physically paid 0.7bpp model! The dense FP32 network is a
capability control, not proof of compact accuracy, not FP32 pretrained
teacher, and not generalization beyond TRAIN. Independent S29 QA stays
sealed and unseen, S26 QA already inspected and never read for tuning.
All work remains on an isolated GitHub research branch. No LIVE Jessica.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s32.microguard_pcgrad_fp32 import old_train_probe,guarded_old_ids
from v186.pairs import build_pairs
from v174.run import save

SOURCE_SHA=s30.SOURCE_SHA
S29_QA_FROZEN_SHA=s30.S29_FROZEN_SHA
TOTAL_STEPS=128
LR_BACKTRACK=(0.00005,0.00002,0.000008)
OLD_GRADIENT_BLEND=0.20
NOUN_CE_WEIGHT=12.
NOUN_FOIL_MARGIN=3.
NOUN_FOIL_MARGIN_WEIGHT=.60
NOUN_OVER_EOS_MARGIN=2.
NOUN_EOS_MARGIN_WEIGHT=.30
SCHEDULE=("new","new","old","new")
SEED=3030


def new_same_color_noun_margin(model,pair,tok):
    """Only S30 TRAIN color+noun examples and true second-token targets."""
    if pair.get("kind")!="new":raise AssertionError("New lexical pair required")
    rows=[]
    for side in ("left","right"):
        r=pair[side]
        color=r["answer"].split()[0]
        prefix=tok.encode(color,add_special_tokens=False)
        if len(prefix)!=1 or len(r["targets"])!=3:
            raise AssertionError("Cannot isolate second noun+EOS")
        if int(r["targets"][0])!=prefix[0]:
            raise AssertionError("Wrong first color answer")
        rows.append(r)
    l,r=rows
    if int(l["targets"][0])!=int(r["targets"][0]):
        raise AssertionError("New contrast must have SAME first color")
    if int(l["targets"][1])==int(r["targets"][1]):
        raise AssertionError("Opposing noun first tokens indistinguishable")
    losses=[]
    for side,foil in (("left",r),("right",l)):
        row=pair[side]
        logits=model(input_ids=row["ids"][None],use_cache=False).logits[0][row["positions"]]
        target=row["targets"]
        ce=F.cross_entropy(logits,target,reduction="none")
        if len(ce)!=3:raise AssertionError("Expected color+noun+EOS")
        weighted=(ce[0]+NOUN_CE_WEIGHT*ce[1]+2.*ce[2])/(3.+NOUN_CE_WEIGHT)
        gold_id=int(target[1]);foil_id=int(foil["targets"][1])
        noun=logits[1]
        foil_margin=F.relu(NOUN_FOIL_MARGIN - (noun[gold_id]-noun[foil_id]))
        eos_margin=F.relu(NOUN_OVER_EOS_MARGIN - (noun[gold_id]-noun[int(tok.eos_token_id)]))
        losses.append(weighted + NOUN_FOIL_MARGIN_WEIGHT*foil_margin +
                      NOUN_EOS_MARGIN_WEIGHT*eos_margin)
    return .5*(losses[0]+losses[1])


def primary_loss(model,kind,pair,tok):
    if kind=="new":return new_same_color_noun_margin(model,pair,tok)
    if kind=="old":return s30.train_loss(model,pair,False)
    raise AssertionError("Unknown fixed TRAIN curriculum")


def one_full_dense_fp32_update(model,opt,params,tok,kind,pair,guard):
    """Single projected gradient then AdamW step, with no QA access."""
    opt.zero_grad(set_to_none=True)
    objective=primary_loss(model,kind,pair,tok)
    if not torch.isfinite(objective):raise AssertionError("Nonfinite TRAIN primary")
    before=float(objective.detach())
    objective.backward()
    g=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
       for p in params]
    opt.zero_grad(set_to_none=True)
    old=s30.train_loss(model,guard,False)
    if not torch.isfinite(old):raise AssertionError("Nonfinite historical TRAIN anchor")
    old_before=float(old.detach())
    old.backward()
    h=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
       for p in params]
    dot=sum((x.double()*y.double()).sum() for x,y in zip(g,h))
    norm=sum((y.double()*y.double()).sum() for y in h)
    if not norm>0 or not torch.isfinite(norm):
        raise AssertionError("Zero or nonfinite old TRAIN replay gradient")
    projected=bool(dot<0)
    c=float(dot/norm) if projected else 0.
    for p,v,older in zip(params,g,h):
        p.grad=(v-c*older+OLD_GRADIENT_BLEND*older).detach()
    grad_norm=float(torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True))
    if not grad_norm>0:raise AssertionError("No FP32 learning gradient")
    opt.step()
    return {
        "TRAIN_task_loss_before":before,
        "TRAIN_old_anchor_loss_before":old_before,
        "gradient_conflict":projected,
        "PCGrad_coefficient":c,
        "whole_FP32_gradient_norm":grad_norm
    }


@torch.inference_mode()
def noun_position_training_probe(model,tok,pairs):
    """TRAIN-only teacher-forced diagnostic, never a full-answer success."""
    rows=[]
    for pair in pairs[::4]:
        for side in ("left","right"):
            r=pair[side]
            logits=model(input_ids=r["ids"][None],use_cache=False).logits[0][r["positions"]]
            target=int(r["targets"][1])
            noun=logits[1]
            rank=int((noun>noun[target]).sum())+1
            top=int(noun.argmax())
            rows.append({
                "id":f'{pair["pair_id"]}_{side}',
                "gold_noun_rank_after_forced_gold_color":rank,
                "gold_noun_top1_after_forced_gold_color":top==target,
                "top1_token":repr(tok.decode([top]))
            })
    ranks=sorted(x["gold_noun_rank_after_forced_gold_color"] for x in rows)
    return {
        "n":len(rows),"median_gold_noun_rank":ranks[len(rows)//2],
        "gold_noun_top1":sum(x["gold_noun_top1_after_forced_gold_color"] for x in rows),
        "rows":rows,"note":"TRAIN teacher-forcing is NOT a correct generated answer"
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw=a.source.read_bytes()
    if len(raw)!=SOURCE_BYTES or hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise AssertionError("Wrong immutable 36M S26 source")
    cb=CB59(raw)
    if cb.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Not original 36M model")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    assert len(new)==32 and len(old)==24
    model=s30.dense_from_paid(cb)
    parity=s30.double_check_original_logits(cb,model,tok)
    initial_sha=s30.snapshot_sha(model)
    before=s30.train_probe(model,tok,new,old)
    first_old=old_train_probe(model,tok,old)
    protected=guarded_old_ids(first_old)
    if len(protected)!=6:
        raise AssertionError("6/8 old original exact TRAIN replies not reproducible")
    initial_rank=noun_position_training_probe(model,tok,new)
    opt=torch.optim.AdamW(model.parameters(),lr=LR_BACKTRACK[0],weight_decay=0.)
    params=list(model.parameters())
    seen=Counter()
    history=[]
    accepted=0
    rejected=0
    attempted_candidates=0
    conflict_count=0
    for step in range(TOTAL_STEPS):
        kind=SCHEDULE[step%len(SCHEDULE)]
        pos=seen[kind];seen[kind]+=1
        pair=(new[pos%len(new)] if kind=="new" else old[pos%len(old)])
        anchor=old[(step*7)%len(old)]
        with torch.inference_mode():
            old_loss=float(primary_loss(model,kind,pair,tok))
        pre_weights=copy.deepcopy(model.state_dict())
        pre_optim=copy.deepcopy(opt.state_dict())
        chosen=None
        trial=[]
        for lr in LR_BACKTRACK:
            # Every candidate starts from EXACT same last accepted state
            # and AdamW moments. Never select on a heldout result.
            model.load_state_dict(pre_weights)
            opt.load_state_dict(pre_optim)
            for group in opt.param_groups:group["lr"]=lr
            info=one_full_dense_fp32_update(model,opt,params,tok,kind,pair,anchor)
            conflict_count+=int(info["gradient_conflict"])
            attempted_candidates+=1
            with torch.inference_mode():
                new_loss=float(primary_loss(model,kind,pair,tok))
            improved=(new_loss+1e-7<old_loss)
            retained=[]
            if improved:
                candidate_old=old_train_probe(model,tok,old)
                retained=sorted(guarded_old_ids(candidate_old))
            passed=improved and protected.issubset(set(retained))
            trial.append({"LR":lr,"TRAIN_primary_after":new_loss,
                          "TRAIN_primary_improved":improved,
                          "old_train_guard_passed":passed,
                          "protected_answers_remaining":len(protected.intersection(retained)),
                          "rejected_old_answer_ids":sorted(protected-set(retained)) if improved else None,
                          **info})
            if passed:
                chosen=lr
                accepted+=1
                break
        if chosen is None:
            model.load_state_dict(pre_weights)
            opt.load_state_dict(pre_optim)
            rejected+=1
        history.append({
            "attempted_step":step+1,"kind":kind,
            "TRAIN_pair":pair["pair_id"],
            "TRAIN_guard_pair":anchor["pair_id"],
            "accepted_LR":chosen,
            "all_3_possible_TRAIN_trials":trial,
            "rolled_back_weights_and_AdamW":chosen is None
        })
        if (step+1)%16==0 or step==0:
            print(json.dumps({
                "event":"S33_TRAIN_LEXICAL_PCGRAD_MICROSTEP_PROGRESS",
                "step":step+1,"accepted":accepted,"rejected":rejected,
                "latest_kind":kind,"latest_accepted_LR":chosen
            },ensure_ascii=False),flush=True)
        del pre_weights,pre_optim
    after=s30.train_probe(model,tok,new,old)
    last_old=old_train_probe(model,tok,old)
    retained_final=guarded_old_ids(last_old)
    if not protected.issubset(retained_final):
        raise AssertionError("S33 lost one of 6 originally correct old TRAIN answers")
    final_rank=noun_position_training_probe(model,tok,new)
    final_sha=s30.snapshot_sha(model)
    report={
        "stage":"S33_MATCHED_36M_FP32_TRAIN_NOUN_MARGIN_ONE_STEP_GUARD",
        "immutable_compact_S26_seed_sha256":SOURCE_SHA,
        "starting_model_is_same_36M_architecture_and_decompressed_weights":True,
        "starting_paid_FP32_logit_max_abs_diff":parity,
        "dense_FP32_payload_not_0_7_bpp":True,
        "S29_heldout_QA_sha_frozen_but_not_opened":S29_QA_FROZEN_SHA,
        "S26_previous_QA_not_loaded":True,
        "no_heldout_QA_used_in_training_or_checkpoint_selection":True,
        "strict_complete_generated_EOS_only_counts_as_right":True,
        "TRAIN_only_outcome_cannot_prove_unseen_accuracy":True,
        "attempted_precommitted_steps":TOTAL_STEPS,
        "all_candidate_train_updates_proposed":attempted_candidates,
        "accepted_full_FP32_microsteps":accepted,
        "rejected_full_FP32_microsteps":rejected,
        "PCGrad_conflicts_across_candidates":conflict_count,
        "begin_unique_FP32_params_sha256":initial_sha,
        "end_unique_FP32_params_sha256":final_sha,
        "all_original_6_old_TRAIN_exact_answer_ids_kept":True,
        "original_six_old_TRAIN_answer_ids":sorted(protected),
        "TRAIN_initial_full_answer_probes":before,
        "TRAIN_final_full_answer_probes":after,
        "TRAIN_old_original_eight_initial":first_old,
        "TRAIN_old_original_eight_final":last_old,
        "TRAIN_initial_forced_noun_rank_DIAGNOSTIC":initial_rank,
        "TRAIN_final_forced_noun_rank_DIAGNOSTIC":final_rank,
        "all_STEP_computation_and_rollback_evidence":history
    }
    save(a.out/"ACTUAL_S33_36M_DENSE_FP32_ONE_STEP_NOUN_MARGIN_TRAIN.json",report)
    print(json.dumps({
        "stage":report["stage"],
        "accepted":accepted,"reverted":rejected,
        "TRAIN_new_before":before["per_kind"]["new_noun_TRAIN"],
        "TRAIN_new_after":after["per_kind"]["new_noun_TRAIN"],
        "TRAIN_old_before":before["per_kind"]["historical_TRAIN"],
        "TRAIN_old_after":after["per_kind"]["historical_TRAIN"],
        "TRAIN_noun_rank_median_before":initial_rank["median_gold_noun_rank"],
        "TRAIN_noun_rank_median_after":final_rank["median_gold_noun_rank"],
        "weights_changed":final_sha!=initial_sha,
        "S29_heldout_never_read":True
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
