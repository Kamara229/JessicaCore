"""CALIBRI2 S14: 36M multi-guard paid-index Green Tunnel training trial.

Motivation is MEASURED S13: S12 kept one rotating OLD TRAIN anchor
nonworsening on each of 24/24 steps, but on 112 broader TRAIN answers four
previously correct responses were lost while four were gained (71->71).
Thus one transactional guard is not full retention. S14 precommits ONE
192-step 36M TRAIN-only attempt using five old TRAIN anchor classes:
  - color (red/blue), numeric Yes/No, story Yes/No,
  - 2-word copy and arithmetic.
Every proposed update must improve a real new TRAIN task and keep all
five ACTUAL old TRAIN losses within tolerance, *after quantizing candidate
centroids to their paid FP16 physical representation*. No teacher model,
QA/VALID/TEST or old frozen question is used for optimizer/checkpoint
selection. All 85 matrices train. A 5-bit code index update, when proposed,
is accepted only under the identical five-anchor transaction.

This protocol is NOT a proof of generalization, universal memory, or
FP32 parity. One fixed final physical checkpoint, no best-of-QA selection.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.codec import replace_discrete
from v185.optim import apply_codes, propose_codes
from v186.pairs import build_pairs
from calibri2.s3.diverse_task_replay import make_new_pairs as s3_pairs
from calibri2.s6.train_29m import new_pairs as s4_pairs
from calibri2.s8.broad_replay import (
    SOURCE_SHA, SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC, PAID_OPERATORS,
    freeze_language_corpus, weighted_pair_loss
)
from calibri2.s11.green_tunnel_protected import new_train_pairs, take_paid_gradient
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from v174.run import save

SEED=2114
STEPS=192
CLIP_NORM=1.0
GUARD_TOL=1.0e-5
TASK_MIN_GAIN=1.0e-7
REPLAY_INJECTION=.20
INDEX_PERIOD=16
INDEX_MAX_SWAPS=512
RATES=(.12,.06,.03,.015,.0075,.00375)
TRAIN_TASKS=("color","color","numeric","copy","color","copy","sentence",
             "color","numeric","copy","language","color")
GUARD_NAMES=("color","number","story_yesno","copy","arithmetic")


def setup():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fixed_guard_banks(tok):
    a=build_pairs(tok,"TRAIN")
    b=s3_pairs(tok)
    c=s4_pairs(tok)
    if (len(a),len(b),len(c))!=(24,64,56):
        raise AssertionError("Old TRAIN data changed")
    pools={
      "color":a[18:24]+b[:24],
      "number":a[:6]+b[24:40],
      "story_yesno":a[6:12],
      "copy":b[40:56]+[p for p in c if "copy" in p["domain"]],
      "arithmetic":a[12:18],
    }
    if set(pools)!=set(GUARD_NAMES) or any(not pools[x] for x in GUARD_NAMES):
        raise AssertionError("Five TRAIN-only saved skill anchors absent")
    return pools


def mean_joint_projection(primary, guards):
    """Constrain 5 actual gradients, not one; no new inference sidecar."""
    update=[p.clone().double() for p in primary]
    anchors=[[a.clone().double() for a in task] for task in guards]
    length=len(anchors)
    for k in range(len(update)):
        update[k]+=REPLAY_INJECTION*sum(gg[k] for gg in anchors)/length
    projected=0
    for _ in range(12):
        changes=0
        for axis in anchors:
            dot=sum(torch.sum(g*h) for g,h in zip(update,axis))
            norm=sum(torch.sum(h*h) for h in axis)
            if norm<=0 or not torch.isfinite(norm):
                raise AssertionError("Frozen TRAIN loss has invalid zero anchor")
            if dot<0:
                rate=dot/norm
                update=[g-rate*h for g,h in zip(update,axis)]
                projected+=1
                changes+=1
        if not changes:
            break
    signed=[float(sum(torch.sum(g*h) for g,h in zip(update,axis)))
            for axis in anchors]
    # Hard numeric guard: if not even first-order feasible, no destructive step.
    if any(v < -1.e-8 for v in signed):
        return None,{"projection_rounds":projected,
                     "signed_guard_after":signed,"first_order_feasible":False}
    norm=float(torch.sqrt(sum(torch.sum(v*v) for v in update)))
    if norm<=1e-15 or not np.isfinite(norm):
        return None,{"projection_rounds":projected,
                     "signed_guard_after":signed,"first_order_feasible":False}
    update=[g/max(CLIP_NORM,norm) for g in update]
    return [g.float() for g in update],{
        "projection_rounds":projected,
        "signed_guard_after":signed,"first_order_feasible":True,
        "before_clipping_L2":norm
    }


def value(model,sources,pair,token_ids=None):
    clear(sources)
    with torch.inference_mode():
        if pair is not None:
            result=float(weighted_pair_loss(model,pair))
        else:
            logits=model(input_ids=token_ids[None],use_cache=False).logits[0]
            result=float(F.cross_entropy(logits[:-1],token_ids[1:]))
    clear(sources)
    if not np.isfinite(result):
        raise AssertionError("TRAIN criterion unexpectedly nonfinite")
    return result


def guard_values(model,sources,anchor_pairs):
    return [value(model,sources,p) for p in anchor_pairs]


def unchanged_or_better(new_scores,old_scores):
    if len(new_scores)!=len(old_scores) or len(old_scores)!=len(GUARD_NAMES):
        raise AssertionError("All 5 old TRAIN guards must be checked")
    return all(new<=old+GUARD_TOL for new,old in zip(new_scores,old_scores))


def quantize_paid_centroids_after_candidate(params,orig,update,rate):
    with torch.no_grad():
        for p,old,direction in zip(params,orig,update):
            # In-memory payload is exactly the future physically paid FP16 values.
            p.copy_((old -rate*direction).clamp(-16,16).half().float())


def precommitted_train_selection(step,new,corpus,counters):
    kind=TRAIN_TASKS[step%len(TRAIN_TASKS)]
    slot=counters[kind]
    counters[kind]+=1
    if kind=="language":
        offset=(slot%(len(corpus)//48))*48
        return kind,None,corpus[offset:offset+48]
    return kind,new[kind][slot%len(new[kind])],None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    setup()
    packed=args.source.read_bytes()
    if hashlib.sha256(packed).hexdigest()!=SOURCE_SHA or len(packed)!=SOURCE_BYTES:
        raise AssertionError("S14 uses exact S7 36M source, not QA-chosen S8/S12")
    archive=CB59(packed)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Incorrect unique source parameter count")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    bank=fixed_guard_banks(tok)
    training=new_train_pairs(tok)
    corpus,corpus_meta=freeze_language_corpus(tok)
    net,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    keys=sorted(sources)
    if len(keys)!=PAID_OPERATORS:
        raise AssertionError("S14 85 paid codebooks missing")
    params=[sources[k].centroids for k in keys]
    counts=Counter()
    trace=[]
    rate_stats=Counter()
    rejected=0
    accepted_index=0
    guard_limit_fail=0
    for step in range(STEPS):
        kind,train_pair,ids=precommitted_train_selection(
            step,training,corpus,counts)
        protect=[bank[name][(step//len(TRAIN_TASKS)+i*3)%len(bank[name])]
                 for i,name in enumerate(GUARD_NAMES)]
        # The model is unmodified while all gradient directions are sampled.
        _,main_gradient,weight_gradient=take_paid_gradient(
            net,sources,keys,pair=train_pair,tokens=ids,
            target_name=keys[(step//INDEX_PERIOD)%len(keys)]
                        if (step+1)%INDEX_PERIOD==0 else None)
        protected_gradients=[]
        for guard in protect:
            _,g,_=take_paid_gradient(net,sources,keys,pair=guard)
            protected_gradients.append(g)
        combined,projection=mean_joint_projection(main_gradient,protected_gradients)
        old_task=value(net,sources,train_pair,ids)
        old_guard=guard_values(net,sources,protect)
        old_centroids=[p.detach().clone() for p in params]
        accepted_rate=None
        trial_log=[]
        if combined is not None:
            for rate in RATES:
                quantize_paid_centroids_after_candidate(
                    params,old_centroids,combined,rate)
                new_task=value(net,sources,train_pair,ids)
                new_guard=guard_values(net,sources,protect)
                guards_pass=unchanged_or_better(new_guard,old_guard)
                gain_pass=new_task+TASK_MIN_GAIN<old_task
                trial_log.append({
                    "step_rate":rate,"new_task_loss":new_task,
                    "guard_loss_new":new_guard,
                    "guard_check_pass":guards_pass,
                    "new_TASK_gain_pass":gain_pass
                })
                if gain_pass and guards_pass:
                    accepted_rate=rate
                    rate_stats[str(rate)]+=1
                    break
                if not guards_pass:
                    guard_limit_fail+=1
        if accepted_rate is None:
            with torch.no_grad():
                for p,old in zip(params,old_centroids):
                    p.copy_(old)
            rejected+=1
            clear(sources)
        else:
            # The exact accepted physical FP16 centroids are already installed.
            if not unchanged_or_better(guard_values(net,sources,protect),old_guard):
                raise AssertionError("Committed actual old guard was harmed")
        swaps=0
        swaps_rejected=False
        if (step+1)%INDEX_PERIOD==0 and accepted_rate is not None:
            source=sources[keys[(step//INDEX_PERIOD)%len(keys)]]
            baseline_task=value(net,sources,train_pair,ids)
            baseline_guard=guard_values(net,sources,protect)
            with torch.inference_mode():
                restored_2d=source().detach().cpu().numpy().copy()
            proposal=propose_codes(source,weight_gradient,restored_2d)
            if proposal["proposed_groups"]>INDEX_MAX_SWAPS:
                raise AssertionError("Exceeded paid 5-bit group proposal budget")
            if proposal["proposed_groups"]:
                with torch.no_grad():
                    apply_codes(source,proposal,True)
                clear(sources)
                next_task=value(net,sources,train_pair,ids)
                next_guard=guard_values(net,sources,protect)
                if next_task+TASK_MIN_GAIN<baseline_task and unchanged_or_better(next_guard,baseline_guard):
                    swaps=int(proposal["proposed_groups"])
                    accepted_index+=swaps
                else:
                    swaps_rejected=True
                    with torch.no_grad():
                        apply_codes(source,proposal,False)
                    clear(sources)
        trace.append({
            "step":step+1,"kind":kind,
            "new_training_pair":None if train_pair is None else train_pair["pair_id"],
            "protecting_five_TRAIN_pairs":
                {name:x["pair_id"] for name,x in zip(GUARD_NAMES,protect)},
            "task_loss_before":old_task,"five_guard_loss_before":old_guard,
            "precommitted_trials":trial_log,
            "accepted_lr":accepted_rate,
            "updated_paid_centroids":accepted_rate is not None,
            "accepted_paid_5bit_group_indices":swaps,
            "rejected_5bit_code_proposal":swaps_rejected,
            **projection
        })
        if (step+1)%24==0:
            print(json.dumps({
                "event":"S14_MULTIGUARD_REAL_PAID_TRAIN_PROGRESS",
                "step":step+1,
                "accepted_centroid_steps":(step+1)-rejected,
                "rejected_steps":rejected,
                "accepted_discrete_5bit_codes":accepted_index,
                "active_TASK":kind,
                "5_guard_checked_every_candidate":True,
                "candidate_rates":dict(rate_stats),
                "rejected_backtrack_on_guard":guard_limit_fail,
            }),flush=True)
    serialized,phys=replace_discrete(archive,sources,strict=False)
    final=args.out/"CALIBRI2_S14_36M_192_MULTIGUARD_PAID.cb59"
    final.write_bytes(serialized)
    sha=hashlib.sha256(serialized).hexdigest()
    if len(serialized)!=SOURCE_BYTES or phys["complete_bpp"]>.700:
        raise AssertionError("S14 complete paid source violated bpp cap")
    if (STEPS-rejected)>0 and sha==SOURCE_SHA:
        raise AssertionError("Accepted centroid steps were not saved into paid physical archive")
    # Check exact FP16 centroid arrays, 5bit codes and model reload.
    reload=CB59(final.read_bytes())
    for name in keys:
        native=reload.handles[name]
        paid=sources[name]
        actual=paid.centroids.detach().cpu().numpy().astype(np.float16)
        if not np.array_equal(native.centroids.astype(np.float16),actual):
            raise AssertionError("S14 physical FP16 codebook changed after pack/reload")
        expected=paid.codes.detach().cpu().numpy().astype(np.uint8)
        from direct_codebook_v59 import unpack5
        real=unpack5(native.packed.tobytes(),len(expected))
        if not np.array_equal(real,expected):
            raise AssertionError("S14 packed physical 5-bit indices changed")
    report={
      "stage":"S14_36M_FIVE_OLD_TASK_GUARDS_REAL_FP16_TRANSACTION",
      "original_S7_sha256":SOURCE_SHA,
      "sealed_S14_sha256":sha,"steps":STEPS,
      "unique_params":SOURCE_PARAMS,"complete_physical_bytes":len(serialized),
      "complete_bpp":phys["complete_bpp"],
      "changed_existing_5bit_indices":phys["actual_changed_5bit_group_indices"],
      "changed_paid_FP16_codebook_cells":phys["actually_changed_FP16_codebook_cells"],
      "actual_training_steps_accepted":STEPS-rejected,
      "actual_training_steps_rejected":rejected,
      "paid_indices_accepted":accepted_index,
      "candidate_failures_on_old_guards":guard_limit_fail,
      "old_TRAIN_guard_categories":GUARD_NAMES,
      "five_actual_old_TRAIN_guard_forward_passes_per_candidate":True,
      "centroids_quantized_to_exact_paid_FP16_before_each_trial":True,
      "no_optimizer_momentum_to_violate_guards":True,
      "TRAIN_only_input_provenance":corpus_meta,
      "training_cycle_counts":dict(counts),
      "no_QA_DEV_VALID_TEST_ever_read_in_training":True,
      "no_pretrained_FP32_teacher_weights":True,
      "no_fp32_weight_sidecar":True,
      "one_precommitted_final_checkpoint":True,
      "model_generalization_not_yet_established":True,
      "all_192_step_actual_trial_evidence":trace,
    }
    save(args.out/"ACTUAL_CALIBRI2_S14_MULTIGUARD_TRAIN.json",report)
    print(json.dumps({
      "event":"S14_MULTI_SKILL_GUARDED_SEALED",
      "sha256":sha,"physical_bpp":phys["complete_bpp"],
      "accepted_centroid_steps":STEPS-rejected,
      "rejected_centroid_steps":rejected,
      "accepted_index_swaps":accepted_index,
      "changed_codebook_cells":phys["actually_changed_FP16_codebook_cells"],
      "no_unpaid_weights":True
    }),flush=True)


if __name__=="__main__":
    main()
