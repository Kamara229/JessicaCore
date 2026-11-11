"""CALIBRI S136: FIVE real physically guarded MPS cycles on 46.70M core
then conditional zero-initialized growth to 50.14M. Train ONLY newly
activated 24 channels/layer; freeze S101 and S131+S132 learned weights.
No heldout QA/S118 or production Jessica/Render/Supabase/APK changes.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import dataclass
from calibri2.s125.j7_atomic_physical_retention import PhysicalRetentionCheck,atomic_train_block
from calibri2.s129.additive_core_expansion import (UP_KEY,ACTIVE_RANK,SEED as EXP_SEED,EXTRA_CORE_PARAMS,EXPANDED_CORE_PARAMS)
from calibri2.s130.route_protected_expansion import RouteIsolatedAdditiveCore
from calibri2.s131.protected_gradients import project_against_old_loss_increase
from calibri2.s132.growth_schedule import ActualJ7Cycle,authorize_growth,architecture_params,START_RANK
from calibri2.s132.progressive_core import IDS_KEY,WEIGHTS_KEY,initial_active_indices
from calibri2.s136.verify_stage3_entry import verified_s135_stage3_entry
from calibri2.s136.stage3_bootstrap import mask_gradient_to_stage3_new_only,verify_frozen_stage1_stage2
from calibri2.s132.progressive_core import ProgressiveRouteCore,expand_preserving_previous
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
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
from calibri2.s95.train_and_audit_router import CHOICES,prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import apply_profile,load_frozen_numeric_weights
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s102.hybrid_retention import strict_train_retention
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import (
    cohort,load_cpu_original,all_456_cached_cpu,
)
from calibri2.s115.temporal_rejected_s112_full456_cpu import (
    candidate_actual_CPU,paired_speed_original_vs_phase,
)
from calibri2.s122.independent_binding_train_curriculum import materialize_train_only
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import export_compact_q2,load_compact_q2,PREFIX
from calibri2.s127.q2_prefill_gate_control import prefill_only_tunnel,restrict_to_gate_only
from calibri2.s115.cached_phase_gate import frozen_cpu_phase_scope
from calibri2.s126.real_s101_full456_zero_parity import paired_speed as s126_pair_speed
from calibri2.s82.real_mps_generic_token_context_copy import candidate_positions
from v176.operators import clear


STEPS=10
TRANSACTION_STEPS=2
TRIAL_SCALES=(1.0,0.5,0.25,0.125,0.0625,0.03125)
LEARNING_RATE=0.000078125
NEW_PAIRS_PER_STEP=3
PROTECTED_EXTRA_PER_STEP=2
GATE_CLAMP=0.05
FIXED_FP16_EVAL_SCALES=(0.35,1.00)
KEY="conditional_dual_bridge.coefficients"
SEED=1290129


def pair_causal_train_gradient(student,bridge,full,tok,row,first,later,pointer,head,*,protected:bool):
    """Teacher-force every original TRAIN token via true differentiable S101 KV.

    S122 novel TRAIN labels have 2 content tokens; the frozen original
    456 TRAIN contains historically correct examples of other lengths.
    All original protected examples, including those with one content
    token plus EOS, must be eligible for S125 rehearsal.
    """
    gold=row["expected"]
    if len(gold)<2 or gold[-1]!=tok.eos_token_id or tok.eos_token_id in gold[:-1]:
        raise AssertionError("S125 protected TRAIN target has invalid EOS ABI")
    prompt=tok.apply_chat_template(
        [{"role":"user","content":row["question"]}],tokenize=False,
        add_generation_prompt=True)
    context=tok.encode(prompt,add_special_tokens=False)
    feat=torch.as_tensor(prompt_only_vector(tok,row["question"]),
                         dtype=torch.float32)
    inp=torch.tensor([context],dtype=torch.long,device="mps")
    with prefill_only_tunnel(student,full),bridge.prompt_scope("S93",feat):
        out=student(input_ids=inp,past_key_values=None,use_cache=True,
                    output_hidden_states=True)
        past=out.past_key_values
        if past is None:
            raise AssertionError("True S101 GPU KV prefill returned no cache")
        hs=out.hidden_states[-1][0]
        scores=[out.logits[0,-1]+F.linear(first(hs[-1]),head)]
        eligible=candidate_positions(tok,context)
        keys=hs[torch.as_tensor(eligible,dtype=torch.long,device="mps")].detach() if eligible else None
        ids=torch.as_tensor([context[i] for i in eligible],dtype=torch.long,
                            device="mps") if eligible else None
        bridge._active_profile=None
        try:
            for index in range(1,len(gold)):
                out=student(
                    input_ids=torch.tensor([[gold[index-1]]],device="mps"),
                    past_key_values=past,use_cache=True,output_hidden_states=True)
                past=out.past_key_values
                h=out.hidden_states[-1][0,-1]
                logits=out.logits[0,-1]+F.linear(later(h),head)
                if index==1 and eligible:
                    bonus=pointer.full_vocab_bonus(h,keys,ids)
                    rival=logits.masked_fill(
                        torch.arange(logits.numel(),device="mps")
                        ==tok.eos_token_id,-1.e8).max()
                    soft=torch.sigmoid((rival-logits[tok.eos_token_id])/0.7)
                    logits=logits.index_add(0,ids,bonus*soft)
                scores.append(logits)
        finally:
            bridge._active_profile="S93"
    if len(scores)!=len(gold):
        raise AssertionError("S125 causal KV continued wrong number of tokens")
    weights=[(2.0 if i<2 else 1.5) if not protected
             else (2.5 if i==len(gold)-1 else 0.75 if i==0 else 1.5)
             for i in range(len(gold))]
    ce=[]
    margins=[]
    for logits,label,w in zip(scores,gold,weights):
        ce.append(w*F.cross_entropy(logits.unsqueeze(0),
            torch.tensor([label],dtype=torch.long,device="mps")))
        if protected:
            alternate=logits.masked_fill(
                torch.arange(logits.numel(),device="mps")==label,-1.e8).max()
            margins.append(F.softplus(1.25-(logits[label]-alternate)))
    result=sum(ce)/sum(weights)
    if protected:
        result=0.45*result+1.65*torch.stack(margins).mean()
    if not bool(torch.isfinite(result).item()):
        raise AssertionError("S125 nonfinite full+EOS protected MPS loss")
    return result


def frozen_routes(tok,router,rows):
    keys=sorted(rows)
    vectors=torch.as_tensor(np.stack([
        prompt_only_vector(tok,rows[k]["question"]) for k in keys]),
        dtype=torch.float32)
    with torch.inference_mode():
        ids=router(vectors).argmax(dim=1).tolist()
    return {k:CHOICES[int(v)] for k,v in zip(keys,ids)}


def select_train_pairs(new,route):
    pairs=defaultdict(dict)
    for row in new:
        pairs[row["pair_id"]][row["side"]]=row
    if len(pairs)!=64 or any(set(p)!={"left","right"} for p in pairs.values()):
        raise AssertionError("S122 original new TRAIN 64 paired ABI modified")
    selected={}
    for pid in sorted(pairs):
        pair=pairs[pid]
        left,right=pair["left"],pair["right"]
        if route[left["id"]]!="S93" or route[right["id"]]!="S93":
            continue
        label=(left["answer"].split()[0],right["answer"].split()[0])
        if label[0]==label[1]:
            raise AssertionError("Not independent color assignment")
        # Precommitted first S93-eligible pair for each ordered TRAIN
        # color combination; no QA or model-output-based selection.
        selected.setdefault(label,(left,right))
    if len(selected)<8:
        raise AssertionError("Insufficient S93-route original S122 different color pairs")
    return [selected[x] for x in sorted(selected)]


def original_new_train_outputs(state,rows,keys,routes,tok,bank,copy_net):
    model,sources,first,later,pointer,head,tunnel,centroids,old_gt=state
    last=None
    output={}
    for key in keys:
        route=routes[key]
        if route!=last:
            apply_profile(sources,tunnel,route,bank,centroids,old_gt)
            last=route
        tokens,_=cached_full_cpu_with_neural_copy(
            model,sources,first,later,head,tok,
            rows[key]["question"],pointer,copy_net)
        output[key]=tokens
    return output


def score_full_and_pairs(rows,prediction):
    result={
        "full":sum(prediction[k]==rows[k]["expected"] for k in rows),
        "first":sum(bool(prediction[k]) and prediction[k][0]==rows[k]["expected"][0]
                    for k in rows),
    }
    by_pair=defaultdict(dict)
    for key,row in rows.items():
        by_pair[row["pair_id"]][row["side"]]=(
            prediction[key]==row["expected"])
    if len(by_pair)!=64 or any(set(p)!={"left","right"} for p in by_pair.values()):
        raise AssertionError("S125 new TRAIN pair coverage incomplete")
    result["counterfactual_pairs_both_correct"]=sum(
        bool(p["left"] and p["right"]) for p in by_pair.values())
    return result



def paired_speed_S131(state,rows,keys,routes,tok,bank,copy_net,full,bridge,expansion,
                      original,proposed):
    """Actual same-host 48 alternating S101 greedy KV evaluations."""
    model,sources,first,later,pointer,head,inner,centroids,old_gt=state
    from calibri2.s126.real_s101_full456_zero_parity import SPEED_SAMPLE
    import statistics
    before=[]
    after=[]
    last=None
    for i,key in enumerate(keys[:SPEED_SAMPLE]):
        route=routes[key]
        if route!=last:
            apply_profile(sources,inner,route,bank,centroids,old_gt)
            last=route
        q=rows[key]["question"]
        feat=torch.as_tensor(prompt_only_vector(tok,q),dtype=torch.float32)
        def base():
            full.uninstall()
            bridge.uninstall()
            try:
                start=time.perf_counter()
                out,_=cached_full_cpu_with_neural_copy(
                    model,sources,first,later,head,tok,q,pointer,copy_net)
                before.append(time.perf_counter()-start)
            finally:
                bridge.install(model)
                full.install(model)
            if out!=original[key]:
                raise AssertionError("S131 CPU baseline pair token mismatch")
        def new():
            with torch.inference_mode():
                with expansion.prefill_only(model),prefill_only_tunnel(model,full):
                    with frozen_cpu_phase_scope(
                        model,bridge,route,feat,phase="prefill_only"):
                        start=time.perf_counter()
                        out,_=cached_full_cpu_with_neural_copy(
                            model,sources,first,later,head,tok,q,pointer,copy_net)
                        after.append(time.perf_counter()-start)
            if out!=proposed[key]:
                raise AssertionError("S131 CPU proposal pair token mismatch")
        if i%2:
            new();base()
        else:
            base();new()
    return {"ratio":statistics.median(before)/statistics.median(after),
        "paired_examples":SPEED_SAMPLE,
        "baseline_median_seconds":statistics.median(before),
        "candidate_median_seconds":statistics.median(after),
        "same_host_alternating":True}

@dataclass(frozen=True)
class GainOnlyRetentionCheck(PhysicalRetentionCheck):
    """J7 requires BOTH full retention and actual original TRAIN improvement.

    Baseline is permitted only as proof of exact rollback, not commit.
    """
    safe_train_gain: bool=False
    verified_base_rollback: bool=False
    new_correct_identities_lost:int=0

    def valid(self):
        return (super().valid() and self.new_correct_identities_lost==0 and
                (self.safe_train_gain or self.verified_base_rollback))


def main():
    p=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S101_full456_report",
        "S122_curriculum_report","S112_report","S126Q_physical","S126Q_report","S87_report","S91_report",
        "S93_report","S135_parent_sidecar","S135_parent_report","S132_parent_sidecar",
    ):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Real Apple MPS GPU is required for S125")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    p99=json.loads(a.S99_full456_report.read_text())
    p101=json.loads(a.S101_full456_report.read_text())
    p122=json.loads(a.S122_curriculum_report.read_text())
    p112=json.loads(a.S112_report.read_text())
    p126q=json.loads(a.S126Q_report.read_text())
    if (p126q.get("stage")!="S126Q_REAL_S101_ZERO_FULL_LAYER_GT12_2BIT"
            or p126q.get("real_original_S101_456_all_token_exact_parity") is not True
            or p126q.get("real_new_S122_128_exact_parity") is not True
            or p126q.get("physical_below_0_7") is not True
            or p126q.get("CPU_at_least_0_9") is not True
            or p126q.get("zero_stage_eligible_for_future_training") is not True
            or sha(a.S126Q_physical.read_bytes())!=p126q.get("physical_sha256")):
        raise AssertionError("S131 requires actual physically proven S126-Q checkpoint")
    if (p101.get("cached_full_EOS_correct")!=353
            or p101.get("new_first_token_correct")!=387
            or p99.get("S99_new_correct_full_EOS")!=353
            or p122.get("stage")!="S122_S82_ONLY_DETERMINISTIC_CROSS_COLOR_COUNTERFACTUAL_TRAIN"
            or p122.get("new_TRAIN_questions")!=128
            or p122.get("new_TRAIN_counterfactual_pairs")!=64
            or p112.get("stage")!="S112_REAL_MPS_CONDITIONAL_340_FP16_456_CPU_TRANSACTION"
            or p112.get("guard",{}).get("lost_full")!=7
            or p112.get("rollback",{}).get("committed") is not False):
        raise AssertionError("S125 immutable original S101/S112/S122 TRAIN source invalid")
    for path,digest in (
        (a.S93_CB61,p99["immutable_S95_source_sha256"]),
        (a.S94_bank,p99["S94_weightbank_sha256"]),
        (a.S95_router,p99["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net,p99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes())!=digest:
            raise AssertionError("Original S101 physical CB61/S94/S95/S98 SHA changed")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new_rows,new_digest,_=materialize_train_only(tok)
    if (new_digest!=p122["new_TRAIN_question_hashset_sha256"]
            or len(new_rows)!=128):
        raise AssertionError("S122 new authorized TRAIN 128 cohort changed")
    new={r["id"]:r for r in new_rows}
    newkeys=sorted(new)
    _,_,_,bank,router,copy_net=load_frozen_numeric_weights(a)
    new_routes=frozen_routes(tok,router,new)
    selected_pairs=select_train_pairs(new_rows,new_routes)
    protected_orig_reports={
        name:freeze_report(getattr(a,name+"_report"))
        for name in ("S87","S91","S93")}
    historical,histkeys,histroutes=cohort(tok,protected_orig_reports,router)
    expected={k:historical[k]["expected"] for k in histkeys}
    side=load_file(str(a.S93_sidecar))
    baseline_q2=load_file(str(a.S126Q_physical))
    zero_q2={k:v for k,v in baseline_q2.items() if k.startswith(PREFIX)}
    if (set(baseline_q2)!=set(side)|set(zero_q2)
            or len(zero_q2)!=6
            or any(not torch.equal(baseline_q2[k],v) for k,v in side.items())):
        raise AssertionError("S131 Q2 original S101 physical source mismatch")
    if not bool(torch.count_nonzero(zero_q2[PREFIX+"gates_fp16"])==0):
        raise AssertionError("S131 Q2 prerequisite is not exact zero gates")
    validated_stage3=verified_s135_stage3_entry(
        a.S135_parent_report,a.S135_parent_sidecar,a.S132_parent_sidecar)
    prior=json.loads(a.S135_parent_report.read_text())
    if validated_stage3["stage3_architecture_params"]!=46_702_400:
        raise AssertionError("S136 requires genuine S132 46.70M physical growth")
    inherited=load_file(str(a.S135_parent_sidecar))
    if (set(inherited)!=set(side)|set(zero_q2)|{IDS_KEY,WEIGHTS_KEY}
        or any(not torch.equal(inherited[k],v) for k,v in side.items())
        or any(not torch.equal(inherited[k],v) for k,v in zero_q2.items())
        or inherited[IDS_KEY].dtype!=torch.int32
        or tuple(inherited[IDS_KEY].tolist())!=initial_active_indices(1344)
        or inherited[WEIGHTS_KEY].shape!=(12,320,72)
        or inherited[WEIGHTS_KEY].dtype!=torch.float16):
        raise AssertionError("S136 43M checkpoint corrupt or changed S101/Q2")
    raw=s87.open_envelope(a.S93_CB61.read_bytes())
    cpu=load_cpu_original(raw,side)
    try:
        before,_=all_456_cached_cpu(
            cpu,historical,histkeys,histroutes,tok,bank,copy_net)
        orig_full={k for k in histkeys if before[k]==expected[k]}
        orig_first={k for k in histkeys if before[k] and before[k][0]==expected[k][0]}
        if len(orig_full)!=353 or len(orig_first)!=387:
            raise AssertionError("S125 immutable actual S101 456 baseline changed")
        new_baseline=original_new_train_outputs(
            cpu,new,newkeys,new_routes,tok,bank,copy_net)
        new_base_stats=score_full_and_pairs(new,new_baseline)
    finally:
        cpu[6].uninstall()

    fragile=set(p112["guard"]["lost_full_identities"])
    if (len(fragile)!=7 or not fragile.issubset(orig_full)
            or any(histroutes[k]!="S93" for k in fragile)):
        raise AssertionError("S125 original 7 known S112 damaged S93 skills invalid")
    extra=[k for k in histkeys if
           k in orig_full and histroutes[k]=="S93" and k not in fragile]
    if len(extra)<2:
        raise AssertionError("S132 needs at least two protected S93 TRAIN items")
    # Five cycles may repeat S93 TRAIN identities, never look at QA/VALID.
    by_family=defaultdict(list)
    for k in extra:by_family[k.split("/",1)[0]].append(k)
    ordered=[]
    while any(by_family.values()):
        for family in sorted(by_family):
            if by_family[family]:
                ordered.append(by_family[family].pop(0))
    rotating=[ordered[i%len(ordered)] for i in range(STEPS*PROTECTED_EXTRA_PER_STEP)]

    student,sources=create_small_paid_model(
        CB59(raw),build_config(MODEL_SPEC))
    student.to("mps").eval().requires_grad_(False)
    inner=IntegratedGreenTunnel(layers=(3,9),rank=8).to("mps").eval()
    inner.load_state_dict({
        k.removeprefix("internal_tunnel."):v for k,v in side.items()
        if k.startswith("internal_tunnel.")
    },strict=True)
    inner.requires_grad_(False)
    install_verified_fastpath(inner)
    inner.install(student)
    bridge=PromptConditionedDualTunnel(inner).to("mps").eval().install(student)
    first,later,pointer=s91.frozen_controllers(side)
    head=student.lm_head.source().detach()
    bridge.coefficients.requires_grad_(False)
    full=FullLayerStateTunnel(width=320,depth=12,rank=4).to("mps").eval()
    load_compact_q2(full,zero_q2)
    if any(not torch.equal(export_compact_q2(full)[k],v) for k,v in zero_q2.items()):
        raise AssertionError("S131 actual S126-Q loaded Q2 physical source mismatch")
    restrict_to_gate_only(full)
    full.gates.requires_grad_(False)
    if not torch.equal(full.gates.detach().cpu(),torch.zeros(12)):
        raise AssertionError("S131 packed baseline not zero initialized")
    full.install(student)
    expansion=ProgressiveRouteCore(rank=1344).to('mps').eval().bind_original_route_bridge(bridge).install(student)
    parameters=expansion.trainable_initial_parameters()
    if EXP_SEED!=SEED or expansion.architectural_new_parameter_count()!=10_321_920:
        raise AssertionError('S136 actual 46.70M 896-rank core ABI differs')
    optimizer=torch.optim.Adam(parameters,lr=LEARNING_RATE)
    traces=[]
    transactions=[]
    physical=[]
    best_safe=None
    cpu=load_cpu_original(raw,side)
    try:
        old_again,_=all_456_cached_cpu(
            cpu,historical,histkeys,histroutes,tok,bank,copy_net)
        if old_again!=before:
            raise AssertionError("S125 physical original S101 all456 parity drift")
        cpu_bridge=PromptConditionedDualTunnel(
            cpu[6],total_layers=12,donor_layers=(3,9)).eval().install(cpu[0])
        cpu_bridge.coefficients.requires_grad_(False)
        cpu_full=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
        load_compact_q2(cpu_full,zero_q2)
        cpu_full.install(cpu[0])
        cpu_expansion=ProgressiveRouteCore(rank=1344).eval().bind_original_route_bridge(cpu_bridge).install(cpu[0])
        try:
            # Continue from actual physically committed S131, never restart
            # five cycles from an obsolete S101 zero-new-weight checkpoint.
            expansion.load_active_fp16({IDS_KEY:inherited[IDS_KEY],WEIGHTS_KEY:inherited[WEIGHTS_KEY]})
            cpu_expansion.load_active_fp16({IDS_KEY:inherited[IDS_KEY],WEIGHTS_KEY:inherited[WEIGHTS_KEY]})
            immutable_before=expansion.up.detach().cpu().clone()
            original_before=before
            with cpu_expansion.prefill_only(cpu[0]),prefill_only_tunnel(cpu[0],cpu_full):
                inherited_before=candidate_actual_CPU(
                    cpu,historical,histkeys,histroutes,tok,bank,copy_net,
                    cpu_bridge,"prefill_only")
                inherited_new=candidate_actual_CPU(
                    cpu,new,newkeys,new_routes,tok,bank,copy_net,
                    cpu_bridge,"prefill_only")
            old_ids_full={k for k in histkeys if original_before[k]==expected[k]}
            old_ids_first={k for k in histkeys if original_before[k] and original_before[k][0]==expected[k][0]}
            if (len(old_ids_full)!=353 or len(old_ids_first)!=387
                or any(inherited_before[k]!=expected[k] for k in old_ids_full)
                or any(not inherited_before[k] or inherited_before[k][0]!=expected[k][0] for k in old_ids_first)
                or sum(inherited_before[k]==expected[k] for k in histkeys)!=prior["old_456_full_correct"]
                or score_full_and_pairs(new,inherited_new)["full"]!=prior["new_128_full_correct"]):
                raise AssertionError("S132 S131+5 physical parent replay differs from actual S131 result")
            prior_gained={k for k in histkeys
                           if inherited_before[k]==expected[k]
                           and original_before[k]!=expected[k]}
            if len(prior_gained)<5:
                raise AssertionError("S136 must protect at least five gained old full TRAIN identities")
            before=inherited_before
            new_base_stats=score_full_and_pairs(new,inherited_new)
            parent_correct_new_full={k for k in new if inherited_new[k]==new[k]["expected"]}
            parent_correct_new_first={k for k in new if inherited_new[k] and
                                    inherited_new[k][0]==new[k]["expected"][0]}
            cycle_evidence=[]
            for block in range(STEPS//TRANSACTION_STEPS):
                last_record=None
                block_start=expansion.up.detach().clone()
                proposed_delta=None
                def train_block():
                    nonlocal proposed_delta
                    for offset in range(TRANSACTION_STEPS):
                        step=block*TRANSACTION_STEPS+offset
                        pairs=[selected_pairs[
                            (step*NEW_PAIRS_PER_STEP+i)%len(selected_pairs)]
                            for i in range(NEW_PAIRS_PER_STEP)]
                        new_batch=[row for pair in pairs for row in pair]
                        old_anchor=('S30/5002/left','S30/5004/left')
                        if any(k not in orig_full or histroutes[k]!='S93' for k in old_anchor):
                            raise AssertionError('S131 original S131 fragile historical TRAIN anchors invalid')
                        old_ids=list(dict.fromkeys([*old_anchor,*sorted(fragile),*sorted(prior_gained)]))
                        old_batch=[historical[k] for k in old_ids]
                        old_batch += [historical[k] for k in
                            rotating[step*PROTECTED_EXTRA_PER_STEP:
                                     (step+1)*PROTECTED_EXTRA_PER_STEP]]
                        optimizer.zero_grad(set_to_none=True)
                        losses=[]
                        # Separate exact new and old TRAIN per-example gradients.
                        for row in new_batch:
                            clear(sources)
                            with expansion.prefill_only(student):
                                loss=pair_causal_train_gradient(
                                    student,bridge,full,tok,row,first,later,pointer,head,
                                    protected=False)
                            (0.55/len(new_batch)*loss).backward()
                            losses.append(float(loss.detach()))
                            clear(sources)
                        novel_grad=expansion.up.grad.detach().clone()
                        optimizer.zero_grad(set_to_none=True)
                        old_gradients=[]
                        for row in old_batch:
                            clear(sources)
                            with expansion.prefill_only(student):
                                loss=pair_causal_train_gradient(
                                    student,bridge,full,tok,row,first,later,pointer,head,
                                    protected=True)
                            (0.45/len(old_batch)*loss).backward()
                            g=expansion.up.grad
                            if g is None or not bool(torch.isfinite(g).all()):
                                raise AssertionError('S131 protected old gradient invalid')
                            old_gradients.append(g.detach().clone())
                            losses.append(float(loss.detach()))
                            optimizer.zero_grad(set_to_none=True)
                            clear(sources)
                        proposed=novel_grad+torch.stack(old_gradients).sum(dim=0)
                        protected_direction,projection=project_against_old_loss_increase(
                            proposed,old_gradients,passes=24)
                        expansion.up.grad=protected_direction
                        grad=expansion.up.grad
                        if grad is None or not bool(torch.isfinite(grad).all()) or int(torch.count_nonzero(grad))<1:
                            raise AssertionError('S131 missing genuine MPS new-core constrained gradient')
                        stage3_grad_mask=mask_gradient_to_stage3_new_only(expansion)
                        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.0))
                        optimizer.step()
                        with torch.no_grad():
                            expansion.up[:,:,896:920].clamp_(-GATE_CLAMP,GATE_CLAMP)
                            expansion.assert_inactive_zero()
                            verify_frozen_stage1_stage2(expansion,immutable_before)
                        torch.mps.synchronize()
                        trace={
                            "step":step+1,"block":block+1,
                            "new_original_S122_TRAIN_examples":len(new_batch),
                            "protected_original_S101_TRAIN_examples":len(old_batch),
                            "mean_loss":sum(losses)/len(losses),
                            "nonzero_grad_coefficients":int(torch.count_nonzero(grad)),
                            "gradient_norm":norm,
                            "protected_projection":projection,
                            "stage3_new_only_mask":stage3_grad_mask,
                            "max_abs_coefficient":float(expansion.up[:,:,896:920].abs().max()),
                        }
                        traces.append(trace)
                        print(json.dumps({"event":"S136_REAL_MPS_Q2_J7_GUARDED_STEP",**trace}),flush=True)
                    proposed_delta=(expansion.up.detach()-block_start).clone()
                    if not bool(torch.isfinite(proposed_delta).all()):
                        raise AssertionError('S131 nonfinite additive-core update')
                    return TRANSACTION_STEPS

                def persisted_cpu_guard():
                    nonlocal last_record
                    restored_base=torch.equal(expansion.up.detach(),block_start)
                    scales=(0.0,) if restored_base else TRIAL_SCALES
                    last_guard=None
                    for trial_scale in scales:
                        with torch.no_grad():
                            expansion.up.copy_(block_start+trial_scale*proposed_delta)
                        index=len(physical)+1
                        learned=expansion.export_active_fp16()
                        if not bool(torch.isfinite(learned[WEIGHTS_KEY]).all()):
                            raise AssertionError("S131 nonfinite Q2 gate after MPS training")
                        cpu_expansion.load_active_fp16(learned)
                        paid=dict(side)
                        paid.update(export_compact_q2(cpu_full))
                        paid.update(cpu_expansion.export_active_fp16())
                        if any(not torch.equal(paid[k],zero_q2[k]) for k in zero_q2):
                            raise AssertionError("S131 frozen Q2 rank4 projection or mix changed")
                        label=f"block{block+1}_check{index}"
                        fp16=a.out/f"S136_UNACCEPTED_{label}.safetensors"
                        save_file(paid,str(fp16))
                        reloaded=load_file(str(fp16))
                        roundtrip=(set(reloaded)==set(paid) and
                                   all(torch.equal(reloaded[k],v) for k,v in paid.items()))
                        originals=(roundtrip and
                                   all(torch.equal(reloaded[k],side[k]) for k in side))
                        if not originals:
                            raise AssertionError("S125 immutable source weights changed or FP16 reload failed")
                        q2={k:v for k,v in reloaded.items() if k.startswith(PREFIX)}
                        load_compact_q2(cpu_full,q2)
                        if any(not torch.equal(q2[k],v) for k,v in export_compact_q2(cpu_full).items()):
                            raise AssertionError('S131 frozen Q2 reloaded differently')
                        cpu_expansion.load_active_fp16({IDS_KEY:reloaded[IDS_KEY],WEIGHTS_KEY:reloaded[WEIGHTS_KEY]})
                        if not torch.equal(cpu_expansion.export_active_fp16()[WEIGHTS_KEY],reloaded[WEIGHTS_KEY]):
                            raise AssertionError('S131 physically reloaded learned new parameters differ')
                        cfg={
                            "stage":"S131_Q2_REAL_J7_GATE_ONLY",
                            "gate_phase":"prefill_only",
                            "rank":4,"bits":2,"layers":12,
                            "old_donors":[3,9],
                            "S111_coefficients":"frozen_zero_unstored",
                            "frozen_Q2_projections":True,
                            "expanded_original_core_params":SOURCE_PARAMS,
                            "expanded_new_core_params":expansion.architectural_new_parameter_count(),
                            "expanded_total_core_params":architecture_params(1344),
                            "expansion_rank":1344,"enabled_rank":len(expansion.active_indices),
                            "fixed_down_seed":EXP_SEED,
                            "inactive_up": "exact_implicit_zero",
                            "new_active_up": "persisted_fp16_after_every_J7_block",
                        }
                        cfg_path=a.out/f"S136_UNACCEPTED_{label}_CONFIG.json"
                        cfg_path.write_text(json.dumps(
                            cfg,sort_keys=True,separators=(",",":"))
                        )
                        extra_bytes=fp16.stat().st_size-a.S93_sidecar.stat().st_size+cfg_path.stat().st_size
                        bpp=8*(float(p101["S99_all_inclusive_physical_bpp"])
                               *SOURCE_PARAMS/8+extra_bytes)/(SOURCE_PARAMS+
                               cpu_full.new_parameter_count()+expansion.architectural_new_parameter_count())
                        with cpu_expansion.prefill_only(cpu[0]),prefill_only_tunnel(cpu[0],cpu_full):
                            original_prediction=candidate_actual_CPU(
                                cpu,historical,histkeys,histroutes,tok,bank,
                                copy_net,cpu_bridge,"prefill_only")
                        speed=paired_speed_S131(
                            cpu,historical,histkeys,histroutes,tok,bank,copy_net,
                            cpu_full,cpu_bridge,cpu_expansion,original_before,original_prediction)
                        guard=strict_train_retention(
                            expected,original_before,original_prediction,
                            physical_bpp=bpp,cpu_ratio=speed["ratio"],speed_required=True)
                        with cpu_expansion.prefill_only(cpu[0]),prefill_only_tunnel(cpu[0],cpu_full):
                            new_prediction=candidate_actual_CPU(
                                cpu,new,newkeys,new_routes,tok,bank,
                                copy_net,cpu_bridge,"prefill_only")
                        new_score=score_full_and_pairs(new,new_prediction)
                        # S102's strict gate deliberately anchors the original
                        # 353 S101 correct IDs; S132 additionally protects all
                        # 358+ skills learned by earlier J7-committed stages.
                        current_full={k for k in histkeys if before[k]==expected[k]}
                        current_first={k for k in histkeys if before[k] and
                                      before[k][0]==expected[k][0]}
                        current_lost_full=sorted(k for k in current_full
                                                 if original_prediction[k]!=expected[k])
                        current_lost_first=sorted(k for k in current_first
                                                  if not original_prediction[k] or
                                                  original_prediction[k][0]!=expected[k][0])
                        current_new_full=sum(
                            original_prediction[k]==expected[k]
                            for k in histkeys if k not in current_full)
                        results={
                            "block":block+1,"check":index,"trial_scale":trial_scale,
                            "original_S101_456_correct_full":sum(
                                original_prediction[k]==expected[k] for k in histkeys),
                            "original_S101_456_correct_first":sum(
                                bool(original_prediction[k]) and
                                original_prediction[k][0]==expected[k][0]
                                for k in histkeys),
                            "lost_original_full":len(current_lost_full),
                            "lost_original_full_identities":current_lost_full,
                            "lost_original_first_identities":current_lost_first,
                            "lost_original_first":len(current_lost_first),
                            "gained_original_full":current_new_full,
                            "new_S122_128_TRAIN_full":new_score["full"],
                            "new_S122_128_TRAIN_first":new_score["first"],
                            "new_S122_64_both_sides_full":
                                new_score["counterfactual_pairs_both_correct"],
                            "physical_bpp":bpp,
                            "CPU_ratio":speed["ratio"],
                            "exact_S101_original_physical_tensors":originals,
                            "sidecar_SHA256":sha(fp16.read_bytes()),
                            "runtime_SHA256":sha(cfg_path.read_bytes()),
                            "frozen_original_all_S101_tensors":originals,
                            "model_accepted":False,
                            "new_core_params":expansion.architectural_new_parameter_count(),
                            "expanded_core_params":architecture_params(1344),
                            "active_new_fp16_weights":12*320*len(expansion.active_indices),
                        }
                        lost_parent_new_full=sum(new_prediction[k]!=new[k]['expected']
                            for k in parent_correct_new_full)
                        lost_parent_new_first=sum(
                            not new_prediction[k] or new_prediction[k][0]!=new[k]['expected'][0]
                            for k in parent_correct_new_first)
                        actual_gain=(lost_parent_new_full==0 and lost_parent_new_first==0 and
                                     (new_score['full']>new_base_stats['full'] or
                                     new_score['counterfactual_pairs_both_correct']>
                                       new_base_stats['counterfactual_pairs_both_correct'] or
                                     current_new_full>0))
                        guard_evidence=GainOnlyRetentionCheck(
                            original_456_checked=len(histkeys),
                            original_full_correct=len(orig_full),
                            original_first_correct=len(orig_first),
                            lost_original_full=len(current_lost_full),
                            lost_original_first=len(current_lost_first),
                            newly_correct_original_full=current_new_full,
                            new_S122_train_full=new_score["full"],
                            new_S122_train_both_correct_pairs=
                                new_score["counterfactual_pairs_both_correct"],
                            physical_all_inclusive_bpp=bpp,
                            paired_original_S101_CPU_speed_ratio=speed["ratio"],
                            physical_FP16_reload_verified=roundtrip,
                            immutable_original_S101_bytes_verified=originals,
                            S95_S98_unchanged=True,
                            safe_train_gain=bool(actual_gain and not restored_base),
                            verified_base_rollback=bool(restored_base),
                            new_correct_identities_lost=lost_parent_new_full+lost_parent_new_first,
                        )
                        results["strict_no_forgetting_guard"]=PhysicalRetentionCheck.valid(guard_evidence)
                        results["protected_new_full_losses"] = lost_parent_new_full
                        results["protected_new_first_losses"] = lost_parent_new_first
                        results["safe_gain"] = guard_evidence.safe_train_gain
                        results["J7_gain_qualified"] = guard_evidence.valid()
                        physical.append(results)
                        last_record=results
                        print(json.dumps({
                            "event":"S136_J7_REAL_Q2_PHYSICAL_456_128_CPU_TRANSACTION",
                            **{k:v for k,v in results.items()
                               if not k.endswith("SHA256")},
                        }),flush=True)
                        last_guard=guard_evidence
                        if guard_evidence.valid():
                            return guard_evidence
                    if last_guard is None:
                        raise AssertionError('S131 no actual physical candidate verified')
                    return last_guard

                result=atomic_train_block(
                    parameters,optimizer,train_block,
                    persisted_cpu_guard,reduction_factor=0.5)
                if result.committed:
                    # The *new* accepted research checkpoint becomes the
                    # protected baseline for the next of five J7 cycles.
                    # This prevents stage-local catastrophic forgetting.
                    with cpu_expansion.prefill_only(cpu[0]),prefill_only_tunnel(cpu[0],cpu_full):
                        accepted_old=candidate_actual_CPU(
                            cpu,historical,histkeys,histroutes,tok,bank,copy_net,
                            cpu_bridge,"prefill_only")
                        accepted_new=candidate_actual_CPU(
                            cpu,new,newkeys,new_routes,tok,bank,copy_net,
                            cpu_bridge,"prefill_only")
                    old_check=strict_train_retention(
                        expected,original_before,accepted_old,
                        physical_bpp=last_record["physical_bpp"],
                        cpu_ratio=last_record["CPU_ratio"],
                        speed_required=True)
                    # Recompute pre-transaction protected identities in
                    # this outer scope; per-trial current_full/current_first
                    # belong to persisted_cpu_guard and must not leak here.
                    inherited_full={k for k in histkeys if before[k]==expected[k]}
                    inherited_first={k for k in histkeys
                                     if before[k] and before[k][0]==expected[k][0]}
                    if (old_check["lost_full"]!=0 or old_check["lost_first"]!=0
                        or any(accepted_old[k]!=expected[k] for k in inherited_full)
                        or any(not accepted_old[k] or accepted_old[k][0]!=expected[k][0]
                               for k in inherited_first)
                        or any(accepted_new[k]!=new[k]["expected"]
                               for k in parent_correct_new_full)
                        or any(not accepted_new[k]
                               or accepted_new[k][0]!=new[k]["expected"][0]
                               for k in parent_correct_new_first)):
                        raise AssertionError("S132 staged J7 commit changed previously learned identities")
                    before=accepted_old
                    new_base_stats=score_full_and_pairs(new,accepted_new)
                    parent_correct_new_full|={
                        k for k in new if accepted_new[k]==new[k]["expected"]}
                    parent_correct_new_first|={
                        k for k in new if accepted_new[k] and
                        accepted_new[k][0]==new[k]["expected"][0]}
                    prior_gained={
                        k for k in histkeys if before[k]==expected[k]
                        and original_before[k]!=expected[k]}
                tx={
                    "block":block+1,
                    "steps":result.block_steps,
                    "committed":result.committed,
                    "rollback_verified":result.rollback_verified,
                    "lr_after":result.learning_rate_after,
                    "failed_candidate_original_full_losses":result.guard.lost_original_full,
                    "failed_candidate_original_first_losses":result.guard.lost_original_first,
                    "model_accepted":False,
                }
                transactions.append(tx)
                if last_record is None:
                    raise AssertionError("S132 no physically verified 456+128 candidate for J7 cycle")
                cycle_evidence.append(ActualJ7Cycle(
                    cycle=block+1,old_456_checked=len(histkeys),
                    lost_previously_correct_full=last_record["lost_original_full"],
                    lost_previously_correct_first=last_record["lost_original_first"],
                    physical_bpp=last_record["physical_bpp"],
                    paired_cpu_ratio=last_record["CPU_ratio"],
                    physical_reloaded=last_record["exact_S101_original_physical_tensors"],
                    source_immutable=last_record["frozen_original_all_S101_tensors"],
                    old_full_count=last_record["original_S101_456_correct_full"],
                    old_first_count=last_record["original_S101_456_correct_first"],
                    new_full_count=last_record["new_S122_128_TRAIN_full"],
                    new_pairs_both_count=last_record["new_S122_64_both_sides_full"],
                    transaction_committed=result.committed,
                    transaction_rollback_verified=result.rollback_verified,
                ))

                if result.committed and last_record is not None and (
                    last_record["new_S122_128_TRAIN_full"]>
                       new_base_stats["full"] or
                    last_record["new_S122_64_both_sides_full"]>
                       new_base_stats["counterfactual_pairs_both_correct"]
                ):
                    best_safe={
                        "block":block+1,
                        "physical_FP16_SHA":last_record["sidecar_SHA256"],
                        "S122_TRAIN_full":last_record["new_S122_128_TRAIN_full"],
                        "S122_TRAIN_both_sides":
                          last_record["new_S122_64_both_sides_full"],
                        "research_only_not_accepted":True,
                    }
                print(json.dumps({
                    "event":"S136_J7_ATOMIC_ADAM_TRANSACTION_END",**tx,
                }),flush=True)
            next_rank=authorize_growth(cycle_evidence,1344)
            # Complete five J7 physical audits before changing model capacity.
            learned_fp16=expansion.export_active_fp16()
            cpu_expansion.load_active_fp16(learned_fp16)
            with cpu_expansion.prefill_only(cpu[0]),prefill_only_tunnel(cpu[0],cpu_full):
                old_stage_456=candidate_actual_CPU(
                    cpu,historical,histkeys,histroutes,tok,bank,copy_net,
                    cpu_bridge,"prefill_only")
                old_stage_128=candidate_actual_CPU(
                    cpu,new,newkeys,new_routes,tok,bank,copy_net,
                    cpu_bridge,"prefill_only")
            prior_stage=ProgressiveRouteCore(rank=1344).eval()
            prior_stage.load_active_fp16(learned_fp16)
            verify_frozen_stage1_stage2(prior_stage,immutable_before)
            next_stage=expand_preserving_previous(prior_stage,next_rank).eval()
            if (not torch.equal(next_stage.up[:,:,:1344],prior_stage.up)
                    or next_stage.architectural_new_parameter_count()!=
                        architecture_params(next_rank)-SOURCE_PARAMS):
                raise AssertionError("S132 expanded structural core did not retain prior learned weights")
            cpu_expansion.uninstall()
            next_stage.bind_original_route_bridge(cpu_bridge).install(cpu[0])
            try:
                growth_sidecar=dict(side)
                growth_sidecar.update(export_compact_q2(cpu_full))
                growth_sidecar.update(next_stage.export_active_fp16())
                growth_physical=a.out/"S136_GROWN_50M_ZERO_NEW_CHANNELS.safetensors"
                save_file(growth_sidecar,str(growth_physical))
                reloaded_growth=load_file(str(growth_physical))
                if (set(reloaded_growth)!=set(growth_sidecar)
                    or any(not torch.equal(reloaded_growth[k],v)
                           for k,v in growth_sidecar.items())
                    or any(not torch.equal(reloaded_growth[k],v)
                           for k,v in side.items())):
                    raise AssertionError("S132 full paid compressed archive roundtrip lost original tensors")
                next_stage.load_active_fp16({
                    k:reloaded_growth[k] for k in next_stage.export_active_fp16()
                })
                cfg={
                    "stage":"S136_STAGE3_FIVE_CYCLES_THEN_ZERO_GROWTH",
                    "next_rank":next_rank,
                    "previous_core_architecture":architecture_params(1344),
                    "next_core_architecture":architecture_params(next_rank),
                    "inherited_trainable_channel_count":24,
                    "stage3_trainable_new_channel_count":24,
                    "active_added_channel_ids":list(next_stage.active_indices),
                    "fixed_down_seeds":"1290129+chunk_index",
                    "new_channels_trained":False,
                    "new_architecture_initial_up_exact_zero":True,
                    "S101_original_core_immutable":True,
                    "S126Q_quantization_immutable":True,
                    "no_QA_no_LIVE":True,
                }
                cfg_path=a.out/"S136_GROWN_50M_RUNTIME.json"
                cfg_path.write_text(json.dumps(cfg,separators=(",",":"),sort_keys=True))
                with next_stage.prefill_only(cpu[0]),prefill_only_tunnel(cpu[0],cpu_full):
                    grown_456=candidate_actual_CPU(
                        cpu,historical,histkeys,histroutes,tok,bank,copy_net,
                        cpu_bridge,"prefill_only")
                    grown_128=candidate_actual_CPU(
                        cpu,new,newkeys,new_routes,tok,bank,copy_net,
                        cpu_bridge,"prefill_only")
                if grown_456!=old_stage_456 or grown_128!=old_stage_128:
                    raise AssertionError("S132 appended parameters changed old+new full/EOS replies")
                speed_grown=paired_speed_S131(
                    cpu,historical,histkeys,histroutes,tok,bank,copy_net,
                    cpu_full,cpu_bridge,next_stage,original_before,grown_456)
                extra_bytes=(growth_physical.stat().st_size
                            -a.S93_sidecar.stat().st_size+cfg_path.stat().st_size)
                paid_bpp=8*(float(p101["S99_all_inclusive_physical_bpp"])
                          *SOURCE_PARAMS/8+extra_bytes)/(
                          SOURCE_PARAMS+cpu_full.new_parameter_count()
                          +next_stage.architectural_new_parameter_count())
                if paid_bpp>.7 or speed_grown["ratio"]<.9:
                    raise AssertionError("S132 expanded research checkpoint exceeds physical bpp or CPU")
                growth_report={
                    **cfg,
                    "five_J7_cycles_verified":[c.__dict__ for c in cycle_evidence],
                    "old_456_exact_EOS_parity":True,
                    "new_128_exact_EOS_parity":True,
                    "old_456_full_correct":sum(grown_456[k]==expected[k] for k in histkeys),
                    "new_128_full_correct":score_full_and_pairs(new,grown_128)["full"],
                    "rate_bpp_all_runtime_included":paid_bpp,
                    "CPU_speed_ratio":speed_grown["ratio"],
                    "additional_structural_core_params":
                        architecture_params(next_rank)-architecture_params(1344),
                    "newly_learned_params_from_structural_growth":0,
                    "physically_persisted_active_coefficients":
                        12*320*len(next_stage.active_indices),
                    "physical_sha256":sha(growth_physical.read_bytes()),
                    "not_promoted_to_Jessica_or_MAIN":True,
                }
                (a.out/"S136_NEXT_STAGE_50M_PHYSICAL_GROWTH_REPORT.json").write_text(
                    json.dumps(growth_report,indent=2,sort_keys=True,ensure_ascii=False))
                print(json.dumps({
                    "event":"S136_FIVE_J7_CYCLES_AND_ZERO_GROWTH_COMPLETE",
                    "next_core_params":architecture_params(next_rank),
                    "old_456_full":growth_report["old_456_full_correct"],
                    "new_128_full":growth_report["new_128_full_correct"],
                    "bpp":paid_bpp,
                    "CPU_ratio":speed_grown["ratio"],
                    "new_channels_not_yet_trained":True,
                }),flush=True)
            finally:
                next_stage.uninstall()
        finally:
            cpu_expansion.uninstall()
            cpu_full.uninstall()
            cpu_bridge.uninstall()
    finally:
        cpu[6].uninstall()
        expansion.uninstall()
        full.uninstall()
        bridge.uninstall()
        inner.uninstall()

    report={
        "stage":"S136_REAL_MPS_46M_STAGE3_J7",
        "original_S101_456_full":353,
        "original_S101_456_first":387,
        "new_S122_128_original_baseline":new_base_stats,
        "S122_new_TRAIN_64_counterfactual_pairs":64,
        "old_protected_original_S101_7_fragile_replayed_per_step":True,
        "original_core_params":SOURCE_PARAMS,
        "additional_core_params":10_321_920,
        "new_total_core_params":architecture_params(1344),
        "new_expansion_active_channels_per_layer":72,
        "new_expansion_total_channels_per_layer":1344,
        "new_expansion_trainable_active_weights":12*320*24,
        "original_S101_S132_S135_learned_channels_frozen":True,
        "five_stage3_training_cycles":5,
        "rest_expansion_inactive_zero":True,
        "immutable_original_core":True,
        "real_MPS_new_additive_up_weights":True,
        "Q2_matrices_and_gates_frozen":True,
        "true_differentiable_MPS_KV_prefill_continuation_EOS":True,
        "S87_J7_method_full_Adam_and_weight_rollback":True,
        "steps_per_physical_transaction":TRANSACTION_STEPS,
        "MPS_training_step_trace":traces,
        "J7_physical_transactions":transactions,
        "every_real_physical_guard_evaluation":physical,
        "atomic_committed_blocks":sum(int(t["committed"]) for t in transactions),
        "atomic_rollback_blocks":sum(int(not t["committed"]) for t in transactions),
        "best_safe_development_TRAIN_checkpoint":best_safe,
        "no_heldout_QA_or_inspected_S118_access":True,
        "S118_QA_permanently_embargoed":True,
        "all_versions_not_accepted":True,
        "do_not_promote_without_independent_unseen_QA":True,
        "no_main_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    (a.out/"S136_REAL_Q2_J7_MPS_PREFILL_456_128_CPU_TRANSACTION_REPORT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
    print(json.dumps({
        "event":"S136_Q2_REAL_MPS_J7_TRANSACTION_COMPLETE",
        "committed_blocks":report["atomic_committed_blocks"],
        "rolled_back_blocks":report["atomic_rollback_blocks"],
        "safe_development_TRAIN_improvement":best_safe is not None,
        "candidate_accepted":False,
        "never_used_QA":True,
    }),flush=True)


if __name__=="__main__":
    main()
