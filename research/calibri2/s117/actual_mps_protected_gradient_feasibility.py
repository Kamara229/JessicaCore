"""S117 real MPS 340D gradient feasibility audit, zero trained candidates.

The S112 actual 340-parameter MPS proposal acquired +1 full+EOS but
forgot 7 correct full answers (REJECTED). Study gradient conflicts in
this SAME prompt-conditional coefficient space, especially the seven
previously correct TRAIN cases damaged by S112, using a GEM-inspired
first-order inequality projection. No optimizer step, physical model
candidate, QA/TEST read, model acceptance or LIVE/Render mutations.

First-order protected loss is ONLY a surrogate. A future actual physical
full 456 greedy S101 CPU guard (353 full, 387 first, bpp <=.7, speed)
is compulsory for every proposed learned model.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s91 import train_joint_quality_rehearsal as s91
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s95.train_and_audit_router import prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import load_frozen_numeric_weights
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import (
    cohort,load_cpu_original,all_456_cached_cpu,
)
from calibri2.s117.protected_gradient_projection import project_for_old_skills
from v176.operators import clear

N_NEW_FIRST=8
N_PROTECT_OLD=16


def round_robin_by_train_family(allkeys,cap):
    family=defaultdict(list)
    for ident in sorted(allkeys):
        family[ident.split("/",1)[0]].append(ident)
    chosen=[]
    while len(chosen)<cap and any(family.values()):
        for group in sorted(family):
            if len(chosen)==cap:break
            if family[group]:
                chosen.append(family[group].pop(0))
    if len(chosen)!=cap:
        raise AssertionError("S117 insufficient TRAIN cohort for gradient study")
    return chosen


def actual_first_ce(student,tok,row,first,head):
    """True initial S101 frozen first-controller scoring, token-0 only."""
    prompt=tok.apply_chat_template(
        [{"role":"user","content":row["question"]}],
        tokenize=False,add_generation_prompt=True)
    ids=tok.encode(prompt,add_special_tokens=False)
    inp=torch.tensor([ids],dtype=torch.long,device="mps")
    out=student(input_ids=inp,use_cache=False,output_hidden_states=True)
    h=out.hidden_states[-1][0][-1]
    logits=out.logits[0,-1]+F.linear(first(h),head)
    target=torch.tensor([row["expected"][0]],dtype=torch.long,device="mps")
    return F.cross_entropy(logits.unsqueeze(0),target)


def main():
    p=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S101_full456_report",
        "S111_parity_report","S112_report",
        "S87_report","S91_report","S93_report",
    ):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Real Apple MPS backprop is required")
    torch.manual_seed(1170117)
    torch.set_num_threads(2)
    p99=json.loads(a.S99_full456_report.read_text())
    p101=json.loads(a.S101_full456_report.read_text())
    p111=json.loads(a.S111_parity_report.read_text())
    p112=json.loads(a.S112_report.read_text())
    if (p101.get("cached_full_EOS_correct")!=353
            or p101.get("new_first_token_correct")!=387
            or p111.get("full_456_exact_parity") is not True
            or p111.get("candidate_full_EOS_correct")!=353
            or p111.get("candidate_first_token_correct")!=387
            or p112.get("stage")!="S112_REAL_MPS_CONDITIONAL_340_FP16_456_CPU_TRANSACTION"
            or p112.get("candidate_full")!=347
            or p112.get("candidate_first")!=388
            or p112.get("guard",{}).get("gained_full")!=1
            or p112.get("guard",{}).get("lost_full")!=7
            or p112.get("guard",{}).get("lost_first")!=0
            or p112.get("rollback",{}).get("committed") is not False
            or p112.get("rollback",{}).get("rollback_verified") is not True):
        raise AssertionError("Original S101/S111/S112 immutable TRAIN evidence not valid")

    for path,digest in (
        (a.S93_CB61,p99["immutable_S95_source_sha256"]),
        (a.S94_bank,p99["S94_weightbank_sha256"]),
        (a.S95_router,p99["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net,p99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes())!=digest:
            raise AssertionError("S101 original physical artifact SHA mismatch")

    _,_,_,bank,router,copy_net=load_frozen_numeric_weights(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    reports={k:freeze_report(getattr(a,k+"_report"))
             for k in ("S87","S91","S93")}
    rows,keys,routes=cohort(tok,reports,router)
    side=load_file(str(a.S93_sidecar))
    raw=s87.open_envelope(a.S93_CB61.read_bytes())
    cpu=load_cpu_original(raw,side)
    try:
        base,_=all_456_cached_cpu(
            cpu,rows,keys,routes,tok,bank,copy_net)
    finally:
        cpu[6].uninstall()
    correct={key for key in keys if base[key]==rows[key]["expected"]}
    correct_first={key for key in keys if bool(base[key]) and
                   base[key][0]==rows[key]["expected"][0]}
    if len(correct)!=353 or len(correct_first)!=387:
        raise AssertionError("S117 original S101 353/387 physical baseline drift")
    all_damaged=p112["guard"]["lost_full_identities"]
    if (len(all_damaged)!=7 or len(set(all_damaged))!=7
            or any(key not in correct or routes[key]!="S93" for key in all_damaged)):
        raise AssertionError("S112 seven protected losses were not genuine S93 TRAIN")

    wrong=[key for key in keys if routes[key]=="S93"
           and key not in correct_first]
    new_keys=round_robin_by_train_family(wrong,N_NEW_FIRST)
    old_pool=[key for key in keys if routes[key]=="S93"
              and key in correct and key not in all_damaged]
    safe_keys=sorted(all_damaged)+round_robin_by_train_family(
        old_pool,N_PROTECT_OLD)
    if len(set(safe_keys))!=7+N_PROTECT_OLD or any(
            key in correct for key in new_keys):
        raise AssertionError("S117 incorrect TRAIN protected/novel cohort split")

    # Frozen original 36M CB61 S93 student, originally trained GT3/GT9
    # donors and controllers, trainable-only prompt-conditioned 340D gates.
    student,sources=create_small_paid_model(
        CB59(raw),build_config(MODEL_SPEC))
    student.to("mps").eval().requires_grad_(False)
    tunnel=IntegratedGreenTunnel(layers=(3,9),rank=8).to("mps").eval()
    tunnel.load_state_dict({
        key.removeprefix("internal_tunnel."):val for key,val in side.items()
        if key.startswith("internal_tunnel.")
    },strict=True)
    tunnel.requires_grad_(False)
    install_verified_fastpath(tunnel)
    tunnel.install(student)
    bridge=PromptConditionedDualTunnel(tunnel).to("mps").eval().install(student)
    first,later,pointer=s91.frozen_controllers(side)
    head=student.lm_head.source().detach()
    gradients={}
    per_example_losses={}
    try:
        for key in new_keys+safe_keys:
            clear(sources)
            row=rows[key]
            feature=torch.as_tensor(
                prompt_only_vector(tok,row["question"]),dtype=torch.float32)
            protected=key in correct
            with bridge.prompt_scope("S93",feature):
                loss=s91.policy_loss(
                    student,tok,row,first,later,pointer,head,
                    protected=protected)
                first_loss=actual_first_ce(student,tok,row,first,head)
                loss=loss+(0.75 if protected else 1.75)*first_loss
            if not bool(torch.isfinite(loss).item()):
                raise AssertionError("Non-finite original S101 surrogate loss")
            (grad,)=torch.autograd.grad(
                loss,bridge.coefficients,retain_graph=False)
            if (grad is None or not bool(torch.isfinite(grad).all())):
                raise AssertionError("S117 no real MPS gradient in 340D bridge")
            gradients[key]=grad.detach().to("cpu",dtype=torch.float64).flatten()
            per_example_losses[key]=float(loss.detach())
            clear(sources)
            print(json.dumps({
                "event":"S117_REAL_MPS_340D_GRADIENT",
                "kind":"incorrect_first" if not protected else (
                    "S112_lost_protected" if key in all_damaged else
                    "other_S101_protected"),
                "family":key.split("/",1)[0],
                "nonzero_gradient_elements":int(torch.count_nonzero(grad)),
                "loss":per_example_losses[key],
            }),flush=True)
    finally:
        bridge.uninstall()
        tunnel.uninstall()

    new_grads=torch.stack([gradients[k] for k in new_keys])
    protected_grads=torch.stack([gradients[k] for k in safe_keys])
    mean_new=new_grads.mean(dim=0)
    projected=project_for_old_skills(mean_new,protected_grads)
    pair_opposition=sum(
        float(torch.dot(x,y))<0 for x in new_grads for y in protected_grads
    )
    candidate_direction=projected.direction.reshape(2,10,17)
    donor_l2=[
        float(candidate_direction[i].norm()) for i in range(2)
    ]
    layer_l2=[
        float(candidate_direction[:,i,:].norm()) for i in range(10)
    ]
    real_new_slope=-float(torch.dot(mean_new,projected.direction))
    if (not all(np.isfinite(x) for x in donor_l2+layer_l2)
            or projected.remaining_violations!=0
            or real_new_slope<-1e-7):
        raise AssertionError("Projected direction invalid/unsafe first-order")

    report={
        "stage":"S117_REAL_MPS_340D_CONDITIONAL_GT_PROTECTED_GRADIENT_FEASIBILITY",
        "unchanged_original_S101_full":353,
        "unchanged_original_S101_first":387,
        "all_456_authorized_TRAIN_reconstructed":True,
        "S112_full_model_candidate_rejected":True,
        "S112_seven_lost_old_TRAIN_covered":len(all_damaged),
        "S93_wrong_first_TRAIN_gradients":len(new_keys),
        "S93_other_old_correct_TRAIN_gradients":N_PROTECT_OLD,
        "S93_total_old_correct_constraints":len(safe_keys),
        "old_new_conflicting_pair_gradients":pair_opposition,
        "all_pair_count":len(new_keys)*len(safe_keys),
        "projected_first_order_constraints":projected.constraints,
        "initial_first_order_constraints_violated":projected.originally_violated,
        "remaining_first_order_constraints_violated":projected.remaining_violations,
        "projected_norm_fraction":projected.projected_norm_ratio,
        "new_task_projected_surrogate_loss_decrease":real_new_slope,
        "donor_projected_update_norms":donor_l2,
        "per_layer_projected_update_norms":layer_l2,
        "iterations":projected.iterations,
        "converged":projected.converged,
        "no_optimizer_steps_no_weight_updates":True,
        "no_physical_candidate_serialized":True,
        "no_held_out_QA_TEST_read_or_trained":True,
        "must_verify_exact_456_autoregressive_CPU_before_any_candidate":True,
        "must_verify_0_700_bpp_and_CPU_speed_before_acceptance":True,
        "model_accepted":False,
        "main_LIVE_Jessica_Render_Supabase_APK_untouched":True,
    }
    (a.out/"S117_ORIGINAL_S101_340D_OLD_GRADIENT_FEASIBILITY.json").write_text(
        json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({
        "event":"S117_MPS_PROTECTED_GRADIENT_DIAGNOSTIC_END",
        "opposed_pairs":pair_opposition,
        "projected_norm_fraction":projected.projected_norm_ratio,
        "protection_violations":projected.remaining_violations,
        "first_order_new_loss_decrease":real_new_slope,
        "no_new_model":True,
    }),flush=True)


if __name__=="__main__":
    main()
