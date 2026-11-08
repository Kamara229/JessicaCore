"""CALIBRI2 S12 transactional protection: small TRAIN-only engineering pilot.

S10 showed local gradient conflict; S11 tests gradient projection but Adam's
stateful preconditioning and momentum can still increase previous TRAIN loss
despite a non-negative instantaneous guard dot product. This preflight uses
an *actual packed 36M* paid codebook model and checks both losses on each real
candidate weight update, accepting only when new-task loss decreases and a
precommitted old TRAIN guard loss does not increase.

No QA/DEV/VALID/TEST, no pretrained FP32 teacher, no 5-bit size increases,
no sidecar. This is a feasibility test, not an S11 result or quality claim.
Fixed 24-step schedule and guard choices. Never use it to pick a checkpoint
based on the frozen independent benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.codec import replace_discrete
from v186.pairs import build_pairs
from calibri2.s3.diverse_task_replay import make_new_pairs as s3_pairs
from calibri2.s8.broad_replay import SOURCE_SHA, SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC, PAID_OPERATORS
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s11.green_tunnel_protected import new_train_pairs, take_paid_gradient, loss_of
from v174.run import save

STEPS = 24
SEED = 2112
PROJECTION_GUARD_COEFF = .35
TRIAL_RATES = (0.05, 0.01, 0.002, 0.0004)
GUARD_TOLERANCE = 1e-5
TASK_IMPROVE = 1e-7
TRAIN_TASK_SCHEDULE = ("color","numeric","copy","color","copy","color")


def setup():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def construct_update(primary, guard):
    dot = sum(torch.sum(a.double()*b.double()) for a,b in zip(primary,guard))
    guard_norm = sum(torch.sum(b.double().square()) for b in guard)
    if not torch.isfinite(guard_norm) or guard_norm<=0:
        raise AssertionError("Guard has no valid TRAIN direction")
    projection = float(dot/guard_norm) if dot<0 else 0.0
    grad = [(a-projection*b+PROJECTION_GUARD_COEFF*b).detach()
            for a,b in zip(primary,guard)]
    norm = float(torch.sqrt(sum(torch.sum(v.float().square()) for v in grad)))
    if not np.isfinite(norm) or norm<=1e-16:
        raise AssertionError("Protected updates have no finite norm")
    scale = max(1.0,norm)
    grad = [x/scale for x in grad]
    signed_after = float(sum(torch.sum(x.double()*b.double())
                 for x,b in zip(grad,guard)))
    if signed_after < -1e-9:
        raise AssertionError("Projected TRAIN guard direction still opposes guard")
    return grad, {"negative_primary_guard":bool(dot<0),
                  "dot_before":float(dot),"dot_after":signed_after,
                  "step_preclip_l2":norm}


@torch.no_grad()
def evaluate_loss(student,sources,pair):
    clear(sources)
    value=float(loss_of(student,pair,None).detach())
    clear(sources)
    if not np.isfinite(value):
        raise AssertionError("Nonfinite evaluated TRAIN loss")
    return value


def step_with_exact_guard(model,sources,keys,task_pair,guard_pair):
    task_before, task_g, _ = take_paid_gradient(model,sources,keys,pair=task_pair)
    guard_before, guard_g, _ = take_paid_gradient(model,sources,keys,pair=guard_pair)
    update, evidence=construct_update(task_g,guard_g)
    params=[sources[k].centroids for k in keys]
    with torch.no_grad():
        old=[p.detach().clone() for p in params]
    accepted = None
    evaluated=[]
    for rate in TRIAL_RATES:
        with torch.no_grad():
            for p,initial,delta in zip(params,old,update):
                p.copy_(initial-rate*delta)
                p.clamp_(-16,16)
        after_new=evaluate_loss(model,sources,task_pair)
        after_guard=evaluate_loss(model,sources,guard_pair)
        okay=(after_new+TASK_IMPROVE<task_before
              and after_guard <= guard_before+GUARD_TOLERANCE)
        evaluated.append({"rate":rate,"new_task_loss":after_new,
                          "old_guard_loss":after_guard,"accepted":okay})
        if okay:
            accepted=rate
            break
    if accepted is None:
        with torch.no_grad():
            for p,initial in zip(params,old):
                p.copy_(initial)
        clear(sources)
    else:
        actual_guard=evaluate_loss(model,sources,guard_pair)
        if actual_guard>guard_before+GUARD_TOLERANCE:
            raise AssertionError("Actual accepted paid update breaks old guard")
    return {
      "task_pair_id":task_pair["pair_id"],
      "guard_pair_id":guard_pair["pair_id"],
      "TASK_loss_before":task_before,
      "GUARD_loss_before":guard_before,
      "selected_rate":accepted,
      "accepted_actual_update":accepted is not None,
      "trials":evaluated,
      **evidence
    }


def main():
    a=argparse.ArgumentParser()
    a.add_argument("source",type=Path)
    a.add_argument("out",type=Path)
    args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    setup()
    raw=args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA or len(raw)!=SOURCE_BYTES:
        raise AssertionError("S12 preflight requires exact S7 36M source")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Unexpected original weight architecture")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    known=build_pairs(tok,"TRAIN")
    s3=s3_pairs(tok)
    new=new_train_pairs(tok)
    # Guard examples are old TRAIN only, never the S11 independent QA.
    # Old color, numeric and yes-no binding preserved in rotation.
    guards=known[:8]+known[18:24]+s3[:8]
    if len(guards)!=22:
        raise AssertionError("Frozen old TRAIN guard schedule drift")
    net,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    keys=sorted(sources)
    if len(keys)!=PAID_OPERATORS:
        raise AssertionError("Missing one of 85 physically paid codebooks")
    changes=[]
    for step in range(STEPS):
        task=TRAIN_TASK_SCHEDULE[step%len(TRAIN_TASK_SCHEDULE)]
        pair=new[task][(step//len(TRAIN_TASK_SCHEDULE))%len(new[task])]
        guard=guards[(step*7)%len(guards)]
        result=step_with_exact_guard(net,sources,keys,pair,guard)
        result["step"]=step+1
        result["task"]=task
        changes.append(result)
        print(json.dumps({"event":"S12_TRANSACTIONAL_PROTECTED_PRECHECK",
            "step":step+1,"task":task,
            "accepted":result["accepted_actual_update"],
            "rate":result["selected_rate"],
            "guard_neg_dot":result["negative_primary_guard"],
            "accepted_so_far":sum(v["accepted_actual_update"] for v in changes)
        }),flush=True)
    packed,phys=replace_discrete(archive,sources,strict=False)
    model_path=args.out/"CALIBRI2_S12_36M_24_ACTUAL_GUARDED_TRAIN.cb59"
    model_path.write_bytes(packed)
    sha=hashlib.sha256(packed).hexdigest()
    count=sum(v["accepted_actual_update"] for v in changes)
    if len(packed)!=SOURCE_BYTES or phys["complete_bpp"]>.700:
        raise AssertionError("S12 physical 36M model violates original bpp")
    if phys["actual_changed_5bit_group_indices"]!=0:
        raise AssertionError("This pilot intentionally did not update paid indices")
    if count and sha==SOURCE_SHA:
        raise AssertionError("Accepted updates never persisted into physical CB59")
    reloaded,reloaded_sources=create_small_paid_model(
        CB59(model_path.read_bytes()),build_config(MODEL_SPEC))
    # Same exact TRAIN-only loss after physical serialization.
    check=guards[0]
    original_loss=evaluate_loss(net,sources,check)
    recovered_loss=evaluate_loss(reloaded,reloaded_sources,check)
    if abs(original_loss-recovered_loss)>5e-4:
        raise AssertionError("Full paid model changed actual old TRAIN loss after reload")
    report={
      "stage":"S12_REAL_36M_EXACT_GUARD_UPDATE_PREFLIGHT",
      "source_S7_SHA":SOURCE_SHA,"sealed_24step_SHA256":sha,
      "physical_unique_params":SOURCE_PARAMS,"physical_bytes":len(packed),
      "complete_bpp":phys["complete_bpp"],
      "paid_85_centroid_codebooks_only":True,
      "actually_changed_paid_codebook_cells":phys["actually_changed_FP16_codebook_cells"],
      "5bit_index_swaps_this_pilot":0,
      "steps_attempted":STEPS,"accepted_updates":count,
      "rejected_updates":STEPS-count,
      "actual_guard_check_not_just_first_order_gradient":True,
      "exact_guard_tolerance":GUARD_TOLERANCE,
      "accepted_one_step_old_TRAIN_guard_loss_nonincreasing":True,
      "post_reload_TRAIN_loss_difference":recovered_loss-original_loss,
      "all_24_step_decisions":changes,
      "no_FP32_teacher":True,"no_QA_DEV_VALID_TEST_used":True,
      "NOT_a_generalization_or_135M_speed_proof":True,
      "LIVE_Jessica_Render_Supabase_APK_untouched":True,
    }
    save(args.out/"ACTUAL_CALIBRI2_S12_GUARD_PROOF.json",report)
    print(json.dumps({"event":"S12_TRANSACTIONAL_GUARD_PILOT_DONE",
           "accepted":count,"attempted":STEPS,
           "physical_bpp":phys["complete_bpp"],
           "archive_sha256":sha,
           "post_reload_loss_delta":report["post_reload_TRAIN_loss_difference"]}),flush=True)


if __name__=="__main__":
    main()
