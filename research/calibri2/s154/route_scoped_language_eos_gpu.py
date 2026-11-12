"""CALIBRI S154: one isolated real S142 43M S93-route-scoped Neural Skill training trial.

Loads the IMMUTABLE physical S142 checkpoint and the original S101/S122
TRAIN sources (no QA). Twenty-four guarded MPS steps train an S93-only language/EOS neural Skill at 4/10 against true causal original S122 labels with protected old TRAIN replay at
skill layers 4/10, **never** altering pretrained S142 parameters. Physically
serialize new skill FP16, reload on CPU, compare protected 456+128 exact
full answer+EOS and first tokens, packed bpp and paired CPU. A failed
trial is rejected and rolled back. Never promotes, merges, or deploys.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS,MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s91 import train_joint_quality_rehearsal as s91
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s95.train_and_audit_router import prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import apply_profile,load_frozen_numeric_weights
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import cohort,load_cpu_original
from calibri2.s115.temporal_rejected_s112_full456_cpu import candidate_actual_CPU
from calibri2.s115.cached_phase_gate import frozen_cpu_phase_scope
from calibri2.s122.independent_binding_train_curriculum import materialize_train_only
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import PREFIX,load_compact_q2,export_compact_q2
from calibri2.s127.q2_prefill_gate_control import prefill_only_tunnel
from calibri2.s132.progressive_core import ProgressiveRouteCore,IDS_KEY,WEIGHTS_KEY
from calibri2.s133.verify_stage2_entry import verified_s132_stage2_entry
from calibri2.s144.train_one_continued_j7 import (
    pair_causal_train_gradient,protected_s122_first_token_margin,frozen_routes, score_full_and_pairs,select_train_pairs
)
from calibri2.s145.neural_skill_blocks import LayerwiseNeuralSkillBank,NeuralSkillSpec
from calibri2.s154.route_scoped_skill_bridge import S93ScopedNeuralSkillHookBridge
from v176.operators import clear

SEED=1510151
SKILL_ID="language_eos_s93_scoped_v2"
SKILL_LAYERS=(4,10)
SKILL_RANK=8
MPS_STEPS=24
LR=0.00030
CPU_SPEED_SAMPLES=24
MAX_BPP=0.7
MIN_CPU_RATIO=0.9
SOURCE_PIN=Path("research/calibri2/s144/source_manifest.json")


def args_from_cli():
    p=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S101_full456_report",
        "S122_curriculum_report","S112_report","S126Q_physical","S126Q_report",
        "S87_report","S91_report","S93_report",
        "S132_parent_sidecar","S132_parent_report","S131_parent_sidecar",
        "S142_parent_sidecar","S142_parent_guard",
    ):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    return p.parse_args()


def check_sources(a):
    manifest=json.loads(SOURCE_PIN.read_text())
    guard=json.loads(a.S142_parent_guard.read_text())
    data=a.S142_parent_sidecar.read_bytes()
    if (manifest["checkpoint_sha256"]!=hashlib.sha256(data).hexdigest()
            or manifest["checkpoint_sha256"]!=guard["physical_FP16_sha256"]
            or manifest["checkpoint_bytes"]!=len(data)
            or manifest["source_run_id"]!=43078895081
            or guard["old_456_full"]!=362
            or guard["old_456_first"]!=389
            or guard["S122_128_full"]!=17
            or guard["committed_gain_no_forgetting"] is not True
            or guard["physical_FP16_reload_pass"] is not True
            or guard["model_accepted"] is not False
            or guard["physical_bpp"]>MAX_BPP
            or guard["CPU_speed_ratio"]<MIN_CPU_RATIO):
        raise AssertionError("S154 immutable physical S142 source verification failed")
    p101=json.loads(a.S101_full456_report.read_text())
    p122=json.loads(a.S122_curriculum_report.read_text())
    p126=json.loads(a.S126Q_report.read_text())
    if (p101["cached_full_EOS_correct"]!=353 or
            p122["new_TRAIN_questions"]!=128 or
            p126["zero_stage_eligible_for_future_training"] is not True):
        raise AssertionError("S154 immutable original S101/S122/Q2 source invalid")
    verified_s132_stage2_entry(a.S132_parent_report,a.S132_parent_sidecar,
                               a.S131_parent_sidecar)
    p99=json.loads(a.S99_full456_report.read_text())
    for path,key in (
        (a.S93_CB61,"immutable_S95_source_sha256"),
        (a.S94_bank,"S94_weightbank_sha256"),
        (a.S95_router,"S95_neural_profile_selector_sha256"),
        (a.S98_copy_net,"S98_neural_copy_selector_sha256"),
    ):
        if sha(path.read_bytes())!=p99[key]:
            raise AssertionError("Original S101 physical source SHA mismatch")
    side=load_file(str(a.S93_sidecar))
    q2=load_file(str(a.S126Q_physical))
    base={k:v for k,v in q2.items() if k.startswith(PREFIX)}
    inherited=load_file(str(a.S142_parent_sidecar))
    if (set(inherited)!=set(side)|set(base)|{IDS_KEY,WEIGHTS_KEY}
            or any(not torch.equal(inherited[k],v) for k,v in side.items())
            or any(not torch.equal(inherited[k],v) for k,v in base.items())
            or inherited[WEIGHTS_KEY].shape!=(12,320,48)
            or inherited[WEIGHTS_KEY].dtype!=torch.float16):
        raise AssertionError("S154 physical sidecar field ABI mismatch")
    return guard,side,base,inherited


def run_outputs(cpu,old,new,old_ids,new_ids,old_routes,new_routes,tok,bank,copy_net,
                bridge,full,expansion):
    model=cpu[0]
    with expansion.prefill_only(model),prefill_only_tunnel(model,full):
        historical=candidate_actual_CPU(
            cpu,old,old_ids,old_routes,tok,bank,copy_net,bridge,"prefill_only")
        novel=candidate_actual_CPU(
            cpu,new,new_ids,new_routes,tok,bank,copy_net,bridge,"prefill_only")
    return historical,novel


def retention(parent_old,candidate_old,old,parent_new,candidate_new,new):
    old_protected_full={k for k in parent_old if parent_old[k]==old[k]["expected"]}
    old_protected_first={k for k in parent_old if parent_old[k] and
                         parent_old[k][0]==old[k]["expected"][0]}
    new_protected_full={k for k in parent_new if parent_new[k]==new[k]["expected"]}
    new_protected_first={k for k in parent_new if parent_new[k] and
                         parent_new[k][0]==new[k]["expected"][0]}
    lost_old_full=sorted(k for k in old_protected_full
                         if candidate_old[k]!=old[k]["expected"])
    lost_old_first=sorted(k for k in old_protected_first if
                          not candidate_old[k] or
                          candidate_old[k][0]!=old[k]["expected"][0])
    lost_new_full=sorted(k for k in new_protected_full
                         if candidate_new[k]!=new[k]["expected"])
    lost_new_first=sorted(k for k in new_protected_first if
                          not candidate_new[k] or
                          candidate_new[k][0]!=new[k]["expected"][0])
    return {
        "lost_old_full_ids":lost_old_full,
        "lost_old_first_ids":lost_old_first,
        "lost_new_full_ids":lost_new_full,
        "lost_new_first_ids":lost_new_first,
        "old_full_before":len(old_protected_full),
        "old_full_after":sum(candidate_old[k]==old[k]["expected"] for k in old),
        "new_full_before":len(new_protected_full),
        "new_full_after":sum(candidate_new[k]==new[k]["expected"] for k in new),
        "old_first_before":len(old_protected_first),
        "new_first_before":len(new_protected_first),
        "new_pairs_both_before":score_full_and_pairs(new,parent_new)["counterfactual_pairs_both_correct"],
        "new_pairs_both_after":score_full_and_pairs(new,candidate_new)["counterfactual_pairs_both_correct"],
    }


def candidate_bytes_and_bpp(guard,physical,manifest,cpu_full,expansion,skill_bank):
    # Parent bpp was physically proven in S142. Add ALL new persisted bytes
    # without claiming a newly allocated but untrained zero parameter is a Skill.
    original_params=(SOURCE_PARAMS+cpu_full.new_parameter_count()+
                     expansion.architectural_new_parameter_count())
    if original_params<43_261_760 or skill_bank.parameter_count()!=10_882:
        raise AssertionError("S154 unexpected parent or skill parameter ABI")
    inherited_bytes=guard["physical_bpp"]*original_params/8
    new_bytes=physical.stat().st_size+manifest.stat().st_size
    return {
        "parent_architectural_params":int(original_params),
        "new_neural_skill_params":skill_bank.parameter_count(),
        "total_architectural_params":original_params+skill_bank.parameter_count(),
        "new_sidecar_bytes":physical.stat().st_size,
        "new_manifest_bytes":manifest.stat().st_size,
        "new_total_physical_bytes":new_bytes,
        "all_in_physical_bpp":8*(inherited_bytes+new_bytes)/(original_params+
                                                              skill_bank.parameter_count()),
        "parent_reference_bpp":guard["physical_bpp"],
    }


def paired_speed(cpu,old,ids,routes,tok,bank,copy_net,bridge,full,expansion,
                 skill_hooks,parent,candidate):
    """Alternate 24 real 43M cached CPU full generations, including skill routing."""
    model,sources,first,later,pointer,head,inner,centroids,old_gt=cpu
    original_times=[]
    candidate_times=[]
    current=None
    skill_hooks.uninstall()
    def single(key,with_skill):
        nonlocal current
        route=routes[key]
        if route!=current:
            apply_profile(sources,inner,route,bank,centroids,old_gt)
            current=route
        if with_skill:
            skill_hooks.install(model)
        try:
            feat=(torch.as_tensor(prompt_only_vector(tok,old[key]["question"]),
                                  dtype=torch.float32) if route=="S93" else None)
            with torch.inference_mode():
                with expansion.prefill_only(model),prefill_only_tunnel(model,full):
                    with frozen_cpu_phase_scope(model,bridge,route,feat,
                                                phase="prefill_only"):
                        start=time.perf_counter()
                        reply,_=cached_full_cpu_with_neural_copy(
                            model,sources,first,later,head,tok,
                            old[key]["question"],pointer,copy_net)
                        sec=time.perf_counter()-start
            expected=(candidate if with_skill else parent)[key]
            if reply!=expected:
                raise AssertionError("S154 paired speed cached full+EOS drift")
            (candidate_times if with_skill else original_times).append(sec)
        finally:
            if with_skill:
                skill_hooks.uninstall()
    for index,key in enumerate(ids[:CPU_SPEED_SAMPLES]):
        if index%2:
            single(key,True);single(key,False)
        else:
            single(key,False);single(key,True)
    ratio=statistics.median(original_times)/statistics.median(candidate_times)
    return {"ratio":ratio,"paired_samples":len(original_times),
            "baseline_median":statistics.median(original_times),
            "skill_median":statistics.median(candidate_times),
            "paired_same_host":True}


def main():
    a=args_from_cli()
    a.out.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S154 is real Apple MPS-only, no CPU fake training")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    guard,side,q2,inherited=check_sources(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new_rows,digest,_=materialize_train_only(tok)
    p122=json.loads(a.S122_curriculum_report.read_text())
    if digest!=p122["new_TRAIN_question_hashset_sha256"] or len(new_rows)!=128:
        raise AssertionError("S154 new TRAIN curriculum hash drift")
    new={row["id"]:row for row in new_rows}
    new_ids=sorted(new)
    _,_,_,bank,router,copy_net=load_frozen_numeric_weights(a)
    new_routes=frozen_routes(tok,router,new)
    pairs=select_train_pairs(new_rows,new_routes)
    originals={name:freeze_report(getattr(a,name+"_report"))
               for name in ("S87","S91","S93")}
    old,old_ids,old_routes=cohort(tok,originals,router)
    if len(old_ids)!=456 or len(new_ids)!=128:
        raise AssertionError("S154 456+128 cohort invalid")
    raw=s87.open_envelope(a.S93_CB61.read_bytes())
    cpu=load_cpu_original(raw,side)
    cpu_bridge=None
    cpu_full=None
    cpu_expansion=None
    cpu_hooks=None
    mps_student=None
    mps_inner=None
    mps_bridge=None
    mps_full=None
    mps_expansion=None
    mps_hooks=None
    try:
        cpu_bridge=PromptConditionedDualTunnel(cpu[6]).eval().install(cpu[0])
        cpu_bridge.coefficients.requires_grad_(False)
        cpu_full=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
        load_compact_q2(cpu_full,q2)
        cpu_full.install(cpu[0])
        cpu_expansion=(ProgressiveRouteCore(rank=896).eval()
                       .bind_original_route_bridge(cpu_bridge).install(cpu[0]))
        payload={IDS_KEY:inherited[IDS_KEY],WEIGHTS_KEY:inherited[WEIGHTS_KEY]}
        cpu_expansion.load_active_fp16(payload)
        parent_old,parent_new=run_outputs(
            cpu,old,new,old_ids,new_ids,old_routes,new_routes,tok,bank,copy_net,
            cpu_bridge,cpu_full,cpu_expansion)
        old_base=sum(parent_old[k]==old[k]["expected"] for k in old_ids)
        new_base=sum(parent_new[k]==new[k]["expected"] for k in new_ids)
        if old_base!=362 or new_base!=17:
            raise AssertionError("S154 real S142 parent CPU 456+128 mismatch")
        spec=NeuralSkillSpec(SKILL_ID,SKILL_LAYERS,SKILL_RANK)
        zero=LayerwiseNeuralSkillBank(320,12)
        zero.add_skill(spec)
        cpu_hooks=S93ScopedNeuralSkillHookBridge(zero,cpu_bridge).install(cpu[0])
        # The new rank-8 block is exact mathematical zero on BOTH S101 and
        # S122 full generation (including current decoder controllers+EOS).
        for rowset,ids,routes,parent in (
            (old,old_ids,old_routes,parent_old),
            (new,new_ids,new_routes,parent_new),
        ):
            with cpu_expansion.prefill_only(cpu[0]),prefill_only_tunnel(cpu[0],cpu_full):
                actual=candidate_actual_CPU(
                    cpu,rowset,ids[:8],routes,tok,bank,copy_net,cpu_bridge,"prefill_only")
            if any(actual[k]!=parent[k] for k in ids[:8]):
                raise AssertionError("S154 zero Neural Skill breaks exact full+EOS output")
        cpu_hooks.uninstall()

        mps_student,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
        mps_student.to("mps").eval().requires_grad_(False)
        mps_inner=IntegratedGreenTunnel(layers=(3,9),rank=8).to("mps").eval()
        mps_inner.load_state_dict(
            {k.removeprefix("internal_tunnel."):v for k,v in side.items()
             if k.startswith("internal_tunnel.")},strict=True)
        mps_inner.requires_grad_(False)
        install_verified_fastpath(mps_inner)
        mps_inner.install(mps_student)
        mps_bridge=(PromptConditionedDualTunnel(mps_inner).to("mps").eval()
                    .install(mps_student))
        mps_bridge.coefficients.requires_grad_(False)
        first,later,pointer=s91.frozen_controllers(side)
        head=mps_student.lm_head.source().detach()
        mps_full=FullLayerStateTunnel(width=320,depth=12,rank=4).to("mps").eval()
        load_compact_q2(mps_full,q2)
        mps_full.requires_grad_(False)
        mps_full.install(mps_student)
        mps_expansion=(ProgressiveRouteCore(rank=896).to("mps").eval()
                       .bind_original_route_bridge(mps_bridge).install(mps_student))
        mps_expansion.load_active_fp16(payload)
        mps_expansion.requires_grad_(False)
        skill=LayerwiseNeuralSkillBank(320,12)
        skill.add_skill(spec)
        skill.to("mps")
        params=skill.select_new_skill_for_training(SKILL_ID)
        mps_hooks=S93ScopedNeuralSkillHookBridge(skill,mps_bridge).install(mps_student)
        optimizer=torch.optim.Adam(params,lr=LR)
        initial=copy.deepcopy(skill.state_dict())
        initial_adam=copy.deepcopy(optimizer.state_dict())
        steps=[]
        if len(pairs)<8:
            raise AssertionError("S154 requires eight precommitted original S122 TRAIN pairs")
        # Original TRAIN only; select protected correct examples BEFORE updates.
        protected_old=tuple(k for k in old_ids if old_routes[k]=="S93"
                            and parent_old[k]==old[k]["expected"])[:24]
        protected_new_first=tuple(k for k in new_ids if new_routes[k]=="S93"
                            and parent_new[k] and
                            parent_new[k][0]==new[k]["expected"][0])[:16]
        if len(protected_old)<8 or len(protected_new_first)<2:
            raise AssertionError("S154 insufficient protected S93 original TRAIN replay")
        for step in range(MPS_STEPS):
            chosen=pairs[step % 8] # deterministic 8 color counterfactual pairs, never QA
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for row in chosen:
                clear(sources)
                with mps_expansion.prefill_only(mps_student):
                    one=pair_causal_train_gradient(
                        mps_student,mps_bridge,mps_full,tok,row,
                        first,later,pointer,head,protected=False)
                losses.append(one)
            # Keep actual previously-correct original S93 full+EOS behavior
            # and S122 first tokens under a differentiable causal margin.
            protected_old_id=protected_old[step % len(protected_old)]
            protected_new_id=protected_new_first[step % len(protected_new_first)]
            clear(sources)
            with mps_expansion.prefill_only(mps_student):
                old_replay=pair_causal_train_gradient(
                    mps_student,mps_bridge,mps_full,tok,old[protected_old_id],
                    first,later,pointer,head,protected=True)
            clear(sources)
            with mps_expansion.prefill_only(mps_student):
                new_first_replay=protected_s122_first_token_margin(
                    mps_student,mps_bridge,mps_full,tok,
                    new[protected_new_id],first,head)
            loss=sum(losses)/len(losses)+0.35*old_replay+0.20*new_first_replay
            loss.backward()
            first_layer=skill.layers[str(SKILL_LAYERS[0])]
            up=first_layer.blocks[SKILL_ID].up.weight
            down=first_layer.blocks[SKILL_ID].down.weight
            gate=first_layer.routers[SKILL_ID].bias
            grads={
                "up":float(up.grad.abs().sum()) if up.grad is not None else 0.0,
                "down":float(down.grad.abs().sum()) if down.grad is not None else 0.0,
                "gate":float(gate.grad.abs().sum()) if gate.grad is not None else 0.0,
            }
            if not grads["up"]>0 or (step==1 and
                    (not grads["down"]>0 or not grads["gate"]>0)):
                raise AssertionError("S154 actual MPS neural Skill learning gradients missing")
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optimizer.step()
            torch.mps.synchronize()
            steps.append({"step":step+1,"loss":float(loss.detach()),"grad_sum":grads,
                          "train_cases":[row["id"] for row in chosen],
                          "protected_old_replay":protected_old_id,
                          "protected_new_first_replay":protected_new_id})
            print(json.dumps({"event":"S154_REAL_S142_MPS_NEURAL_SKILL_STEP",**steps[-1]}),
                  flush=True)
        clear(sources)
        # Original physical S142 tensors are not even included in the
        # optimizer; prove the structural freeze at the time of update.
        if any(p.requires_grad for p in mps_student.parameters()):
            raise AssertionError("S154 original transformer was trainable")
        if (mps_expansion.up.requires_grad or mps_full.gates.requires_grad or
                mps_bridge.coefficients.requires_grad):
            raise AssertionError("S154 inherited S142 learned paths were trainable")

        physical=a.out/"S154_UNACCEPTED_FIRST_SKILL_FP16.safetensors"
        learned={f"s151_neural_skill.{k}":v.detach().cpu().to(torch.float16).contiguous()
                 for k,v in skill.state_dict().items()}
        save_file(learned,str(physical))
        loaded=load_file(str(physical))
        if set(loaded)!=set(learned) or any(
                not torch.equal(learned[k],loaded[k]) for k in loaded):
            raise AssertionError("S154 physical FP16 Skill serialization failed")
        cpu_skill=LayerwiseNeuralSkillBank(320,12)
        cpu_skill.add_skill(spec)
        cpu_skill.load_state_dict({
            k.removeprefix("s151_neural_skill."):v.float()
            for k,v in loaded.items()},strict=True)
        cpu_hooks=S93ScopedNeuralSkillHookBridge(cpu_skill,cpu_bridge).install(cpu[0])
        proposed_old,proposed_new=run_outputs(
            cpu,old,new,old_ids,new_ids,old_routes,new_routes,tok,bank,copy_net,
            cpu_bridge,cpu_full,cpu_expansion)
        audit=retention(parent_old,proposed_old,old,parent_new,proposed_new,new)
        runtime=a.out/"S154_SKILL_RUNTIME.json"
        runtime.write_text(json.dumps({
            "stage":"S154_REAL_S142_NEURAL_SKILL",
            "model_accepted":False,"source_S142_SHA":sha(a.S142_parent_sidecar.read_bytes()),
            "skill_id":SKILL_ID,"skill_rank":SKILL_RANK,"layers":SKILL_LAYERS,
            "train_steps":MPS_STEPS,"training_only_authorized_S122":True,"protected_original_S93_replay":True,
            "source_original_immutable":True,"never_used_QA":True,
            "real_neural_skill_train":True,"not_production":True,
        },sort_keys=True,separators=(",",":")))
        physical_stats=candidate_bytes_and_bpp(
            guard,physical,runtime,cpu_full,cpu_expansion,cpu_skill)
        speed=paired_speed(cpu,old,old_ids,old_routes,tok,bank,copy_net,
                           cpu_bridge,cpu_full,cpu_expansion,cpu_hooks,
                           parent_old,proposed_old)
        all_kept=not any(audit[k] for k in (
            "lost_old_full_ids","lost_old_first_ids",
            "lost_new_full_ids","lost_new_first_ids"))
        gains=(audit["old_full_after"]>audit["old_full_before"] or
               audit["new_full_after"]>audit["new_full_before"] or
               audit["new_pairs_both_after"]>audit["new_pairs_both_before"])
        permitted=(all_kept and gains and
                   physical_stats["all_in_physical_bpp"]<=MAX_BPP and
                   speed["ratio"]>=MIN_CPU_RATIO)
        if not permitted:
            # A genuine rollback includes weights and Adam state; detached
            # CPU candidate hook removed. S142 stored checkpoint untouched.
            cpu_hooks.uninstall()
            skill.load_state_dict(initial,strict=True)
            optimizer.load_state_dict(initial_adam)
            if not all(torch.equal(v,skill.state_dict()[k]) for k,v in initial.items()):
                raise AssertionError("S154 rejected Skill rollback weights differ")
            if optimizer.state_dict()!=initial_adam:
                raise AssertionError("S154 rejected Skill rollback Adam differs")
        report={
            "stage":"S154_REAL_S142_FIRST_NEURAL_SKILL_MPS_TRIAL",
            "source_S142_FP16_sha256":sha(a.S142_parent_sidecar.read_bytes()),
            "real_S142_parent_old456":old_base,
            "real_S142_parent_new128":new_base,
            "training_steps":steps,
            "physical_skill_fp16_sha256":sha(physical.read_bytes()),
            "physical_safetensors_reloaded":True,
            "physical":physical_stats,
            "paired_CPU":speed,
            "full456_and_new128":audit,
            "strict_no_forgetting":all_kept,
            "TRAIN_only_skill_gain":gains,
            "research_safe_development_candidate":permitted,
            "J7_guarded_skill_rollback_verified":not permitted,
            "original_S142_core_and_S101_Q2_unchanged":True,
            "model_accepted":False,
            "independent_heldout_QA_never_run":True,
            "no_MAIN_LIVE_Jessica_Render_Supabase_APK_changes":True,
        }
        out=a.out/"S154_REAL_NEURAL_SKILL_456_128_AUDIT.json"
        out.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
        print(json.dumps({
            "event":"S154_FIRST_REAL_NEURAL_SKILL_PHYSICAL_456_128_AUDIT",
            "parent_old":old_base,"candidate_old":audit["old_full_after"],
            "parent_new":new_base,"candidate_new":audit["new_full_after"],
            "lost_old_full":len(audit["lost_old_full_ids"]),
            "lost_new_full":len(audit["lost_new_full_ids"]),
            "all_in_bpp":physical_stats["all_in_physical_bpp"],
            "paired_CPU_ratio":speed["ratio"],"safe_candidate":permitted,
            "rolled_back":not permitted,"model_accepted":False,
        }),flush=True)
    finally:
        if cpu_hooks is not None:cpu_hooks.uninstall()
        if cpu_expansion is not None:cpu_expansion.uninstall()
        if cpu_full is not None:cpu_full.uninstall()
        if cpu_bridge is not None:cpu_bridge.uninstall()
        if cpu is not None:cpu[6].uninstall()
        if mps_hooks is not None:mps_hooks.uninstall()
        if mps_expansion is not None:mps_expansion.uninstall()
        if mps_full is not None:mps_full.uninstall()
        if mps_bridge is not None:mps_bridge.uninstall()
        if mps_inner is not None:mps_inner.uninstall()


if __name__=="__main__":
    main()
