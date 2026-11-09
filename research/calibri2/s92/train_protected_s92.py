"""S92: real MPS joint core+Green Tunnel constrained learning. RESEARCH ONLY.

The S91 physical prototype is a genuine compressed starting point, NOT an
accepted model. Every protected reference comes from frozen historical TRAIN.
Forgetting can still happen after first-order gradient projection and FP16
repacking: independent strict physical 456 CPU inference decides acceptance.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel, select_paid_core_codebooks
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s91 import train_joint_quality_rehearsal as s91
from calibri2.s92.gradient_safety import project_primary_gradient
from v176.operators import clear

SEED=920092
STEPS=32
CHECKPOINTS=(8,16,24,32)
LR_CORE=0.00006
LR_TUNNEL=0.00035


def audit_inputs(tok, frozen_s87, frozen_s91):
    rows, _, _ = s91.references(tok, frozen_s87)
    prev=frozen_s91["all_456_per_identity_results"]
    if len(prev)!=456 or len({x["id"] for x in prev})!=456:
        raise AssertionError("S92 complete frozen S91 456 required")
    s91_map={x["id"]:x for x in prev}
    if set(rows)!=set(s91_map):
        raise AssertionError("S92 TRAIN IDs changed")
    for key,row in rows.items():
        prior=s91_map[key]
        if (prior["question_sha256"]!=row["prior"]["question_sha256"] or
            prior["target_ids"]!=row["expected"] or
            prior["S82_ids"]!=row["prior"]["S82_ids"]):
            raise AssertionError("S91 full frozen identity drift: "+key)
        row["s91_ids"]=prior["S87_ids"]
        row["s91_full"]=prior["S87_ids"]==row["expected"]
        row["s91_first"]=bool(prior["S87_ids"]) and prior["S87_ids"][0]==row["expected"][0]
        row["union_full"]=row["prior"]["old_full"] or row["s91_full"]
        row["union_first"]=row["prior"]["old_first"] or row["s91_first"]
    old_full=sum(x["prior"]["old_full"] for x in rows.values())
    new_full=sum(x["s91_full"] for x in rows.values())
    union_full=sum(x["union_full"] for x in rows.values())
    repair=[x for x in rows.values() if x["prior"]["old_full"] and not x["s91_full"]]
    all_protected=[x for x in rows.values() if x["s91_full"]]
    first_protected=[x for x in rows.values() if x["s91_first"] and not x["s91_full"]]
    first_repair=[x for x in rows.values() if x["prior"]["old_first"] and not x["s91_first"]]
    if (old_full,new_full,union_full,len(repair),len(all_protected),len(first_repair))!=(196,257,282,25,257,1):
        raise AssertionError("S92 requires exact measured S87/S91 TRAIN negative audit")
    candidates={}
    for r in rows.values():
        if (r["family"]=="S82_new" and not r["union_full"] and
            len(r["expected"])==3):
            candidates.setdefault(r["id"].rsplit("/",1)[0],[]).append(r)
    fresh=[r for _,pair in sorted(candidates.items()) if len(pair)==2 for r in pair][:12]
    if len(fresh)!=12:
        raise AssertionError("S92 needs six still-wrong paired TRAIN contrasts")
    return rows,repair,all_protected,first_protected,first_repair,fresh


def full_policy_grad(student,sources,tok,first,later,pointer,head,row,params,protected=False):
    for p in params:p.grad=None
    clear(sources)
    val=s91.policy_loss(student,tok,row,first,later,pointer,head,protected=protected)
    if not torch.isfinite(val):
        raise AssertionError("S92 nonfinite actual full-policy loss")
    val.backward()
    result=[]
    for p in params:
        if p.grad is None:
            raise AssertionError("S92 broken live gradient connection")
        result.append(p.grad.detach().float().clone())
    clear(sources)
    return result,float(val.detach().cpu())


def mini_union_guard(raw,side,tok,cohort):
    state=s87.build_cpu(raw,side,True)
    model,sources,first,later,pointer,head,internal=state
    install_verified_fastpath(internal)
    lost_full=[];lost_first=[];gained=[];counts={"checked":len(cohort)}
    try:
        for r in cohort:
            from calibri2.s82 import real_mps_generic_token_context_copy as s82
            out=s82.full_cpu_generate(model,sources,first,later,head,tok,r["question"],pointer)
            full=out==r["expected"]
            first_ok=bool(out) and out[0]==r["expected"][0]
            if r["union_full"] and not full:lost_full.append(r["id"])
            if r["union_first"] and not first_ok:lost_first.append(r["id"])
            if full and not r["union_full"]:gained.append(r["id"])
    finally:
        internal.uninstall()
        del state,model
    return {**counts,"union_full_losses":lost_full,"union_first_losses":lost_first,
            "beyond_union_full_gains":gained}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("S87_original_CB61",type=Path)
    p.add_argument("S91_physical_CB61",type=Path)
    p.add_argument("S91_safetensors",type=Path)
    p.add_argument("S91_manifest",type=Path)
    p.add_argument("S91_full_456_guard",type=Path)
    p.add_argument("S87_frozen_456_guard",type=Path)
    p.add_argument("S82_controller",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S92 requires actual MPS gradients, no CPU fallback")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    s91_raw=a.S91_physical_CB61.read_bytes()
    source_raw=s87.open_envelope(s91_raw)
    side_bytes=a.S91_safetensors.read_bytes()
    side=load_file(str(a.S91_safetensors))
    meta=json.loads(a.S91_manifest.read_text())
    original=a.S87_original_CB61.read_bytes()
    frozen_s87=json.loads(a.S87_frozen_456_guard.read_text())
    frozen_s91=json.loads(a.S91_full_456_guard.read_text())
    if (meta.get("stage")!="S91_RESEARCH_ONLY_JOINT_NEW_FULL_EOS_REPLAY_MPS_TRAIN_UNACCEPTED" or
        meta.get("candidate_accepted") is not False or
        meta.get("source_CB61_physical_sha256")!=s91.hash_bytes(s91_raw) or
        meta.get("source_sha256")!=s91.hash_bytes(source_raw) or
        meta.get("prototype_sidecar_sha256")!=s91.hash_bytes(side_bytes) or
        meta.get("base_source_S87_sha256")!=s91.hash_bytes(original) or
        meta.get("old_S82_sidecar_sha256")!=s91.hash_bytes(a.S82_controller.read_bytes())):
        raise AssertionError("S92 frozen S91/S87/S82 provenance mismatch")
    if (frozen_s87.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT" or
        frozen_s91.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT"):
        raise AssertionError("S92 must use measured read-only complete CPU audit reports")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    rows,repair,protect,first_only,first_repair,fresh=audit_inputs(tok,frozen_s87,frozen_s91)
    student,sources=create_small_paid_model(CB59(source_raw),build_config(s87.MODEL_SPEC))
    student.to("mps").eval().requires_grad_(False)
    inner=IntegratedGreenTunnel(layers=(3,9),rank=8).to("mps").eval()
    inner.load_state_dict({k.removeprefix("internal_tunnel."):v for k,v in side.items()
                           if k.startswith("internal_tunnel.")},strict=True)
    for block in inner.blocks.values():block.gate.requires_grad_(False)
    install_verified_fastpath(inner)
    inner.install(student)
    core=select_paid_core_codebooks(sources,(3,9))
    core_params=[v.centroids for v in core.values()]
    tunnel_params=[x for x in inner.parameters() if x.requires_grad]
    params=core_params+tunnel_params
    if len(core_params)!=2 or not tunnel_params:
        raise AssertionError("S92 core + tunnel both required")
    rates=[LR_CORE]*len(core_params)+[LR_TUNNEL]*len(tunnel_params)
    first,later,pointer=s91.frozen_controllers(side)
    head=student.lm_head.source().detach()
    rng=random.Random(SEED)
    saved=[]
    best=None
    best_key=None
    print(json.dumps({"event":"S92_PHYSICAL_S91_MPS_PROTECTED_GRADIENT_START",
          "baseline_S87_full":196,"baseline_S91_full":257,"protected_union_full":282,
          "S91_regressions_for_repair":len(repair),
          "S87_first_regressions":len(first_repair),
          "train_new_wrong_paired_targets":len(fresh),
          "trainable_core_dictionaries":list(core),
          "trainable_neural_tunnel_parameters":sum(x.numel() for x in tunnel_params),
          "QA_TEST_imported":False,"LIVE_untouched":True}),flush=True)
    skipped=0
    for step in range(1,STEPS+1):
        chosen=[repair[(2*step-2)%len(repair)],
                repair[(2*step-1)%len(repair)],
                fresh[(2*step-2)%len(fresh)],
                fresh[(2*step-1)%len(fresh)]]
        if step%4==0:chosen[1]=first_repair[0]
        primary=[torch.zeros_like(x,dtype=torch.float32) for x in params]
        losses=[]
        for row in chosen:
            grads,value=full_policy_grad(student,sources,tok,first,later,pointer,
                head,row,params,protected=False)
            primary=[v+g/len(chosen) for v,g in zip(primary,grads)]
            losses.append(value)
        # Protect the S91-correct answers that are genuinely correct at our
        # physical starting checkpoint. Repair S87 losses via primary gradients.
        guard_rows=[protect[(step*11)%len(protect)],
                    protect[(step*29)%len(protect)],
                    protect[(step*47)%len(protect)],
                    first_only[(step*7)%len(first_only)]]
        protection=[]
        for row in guard_rows:
            grads,_=full_policy_grad(student,sources,tok,first,later,pointer,
                head,row,params,protected=True)
            protection.append(grads)
        proposed=[g*lr for g,lr in zip(primary,rates)]
        direction,diag=project_primary_gradient(proposed,protection,
                            sweeps=96,relative_tolerance=1e-6)
        # A projected numerical halfspace is NOT a physical no-forgetting
        # certificate. All snapshot candidates are serialized and CPU guarded.
        if diag["can_apply_gradient"]:
            with torch.no_grad():
                for p,d in zip(params,direction):
                    p.add_(-d.to(dtype=p.dtype))
                for q in core_params:q.clamp_(-16,16)
        else:skipped+=1
        if step%4==0:
            print(json.dumps({"event":"S92_REAL_MPS_PROJECTED_JOINT_STEP",
                 "step":step,"loss":sum(losses)/len(losses),
                 "skipped":skipped,**diag}),flush=True)
        if step in CHECKPOINTS:
            clear(sources)
            raw,new_side,src,sidefile,mutation=s91.materialize_candidate(
                 source_raw,sources,inner,side,a.out,step)
            pick={r["id"]:r for r in
                  repair+fresh+first_repair+guard_rows+
                  [protect[i] for i in range(0,len(protect),4)]}
            mini=mini_union_guard(raw,new_side,tok,list(pick.values()))
            stat={"step":step,"mini":mini,
                  "changed_codebook_cells":mutation["actually_changed_FP16_codebook_cells"],
                  "source_sha256":s91.hash_bytes(raw)}
            saved.append(stat)
            print(json.dumps({"event":"S92_PHYSICAL_UNION_MINI_GUARD",**stat}),flush=True)
            quality=(len(mini["beyond_union_full_gains"])
                   -25*len(mini["union_full_losses"])
                   -25*len(mini["union_first_losses"]),
                   len(mini["beyond_union_full_gains"]))
            within=(len(src.read_bytes())+len(sidefile.read_bytes())+7500)*8/s91.SOURCE_PARAMS<0.7
            if within and (best_key is None or quality>best_key):
                best_key=quality
                best=(src,sidefile,raw,mutation,stat)
    inner.uninstall()
    if best is None:
        raise AssertionError("S92 no physical compressed candidate")
    src,sidefile,raw,mutation,stat=best
    final_src=a.out/"S92_UNACCEPTED_SELECTED_CB61.s61"
    final_side=a.out/"S92_UNACCEPTED_SELECTED_JOINT.safetensors"
    final_src.write_bytes(src.read_bytes())
    final_side.write_bytes(sidefile.read_bytes())
    record={
       "stage":"S92_RESEARCH_ONLY_PROJECTED_JOINT_S91_REPAIR_MPS_UNACCEPTED",
       "S87_source_sha256":s91.hash_bytes(original),
       "base_source_S91_sha256":s91.hash_bytes(s91_raw),
       "base_S91_sidecar_sha256":s91.hash_bytes(side_bytes),
       "source_CB61_physical_sha256":s91.hash_bytes(final_src.read_bytes()),
       "source_sha256":s91.hash_bytes(raw),
       "prototype_sidecar_sha256":s91.hash_bytes(final_side.read_bytes()),
       "old_S82_sidecar_sha256":s91.hash_bytes(a.S82_controller.read_bytes()),
       "baseline_S87_full":196,"baseline_S91_full":257,
       "union_of_previously_full_correct":282,
       "strict_CPU_all456_pending":True,"selected_step":stat["step"],
       "gradient_projection_not_proof_of_no_forgetting":True,
       "QA_TEST_not_accessed":True,"LIVE_untouched":True,
       "joint_core_and_tunnel_gradient_paths":True,
       "candidate_accepted":False,
       "mini_history":saved,"changed_core_codebooks":mutation
    }
    data=s91.stable_manifest(record,final_src.read_bytes(),final_side.read_bytes())
    (a.out/"S92_UNACCEPTED_MANIFEST.json").write_bytes(data)
    print(json.dumps({"event":"S92_UNACCEPTED_PHYSICAL_PROTOTYPE",
       "physical_all_inclusive_bpp":record["all_inclusive_physical_bpp"],
       "selected_step":stat["step"],"candidate_accepted":False}),flush=True)


if __name__=="__main__":
    main()
