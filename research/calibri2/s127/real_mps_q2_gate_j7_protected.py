"""CALIBRI S127: REAL 12-layer packed rank4 Q2 gate-only MPS TRAIN.
 
Runs an isolated GPU causal S101 KV-gradient and genuinely persisted 456+128
CPU physical guard; S125 J7 transaction rolls back both 12 gates and full
Adam optimizer when any old-skill identity, physical bpp or CPU ratio fails.
Frozen Q2 encoder/decoder and original S101 weights must remain unchanged.
S118 independent QA and production Jessica are NEVER used.
 
Original S125 design follows for reproducibility:

No QA/TEST is opened (particularly S118, already inspected/embargoed).
S122 64 owner-binding counterfactual pairs are generated from historical
S82 TRAIN nouns, frame text, owner names and four original color tokens;
only 12 route-eligible original TRAIN-derived cross-color pairs teach GT.

S125 trains ONLY the 340 FP32 coefficients of an already paid dual
GT3/GT9 S111 bridge. The original 36M CB61, original S87/S91 S94
bank, S95 router, S98 copy and original donor blocks are immutable.

Unlike S112's one-pass full-sequence surrogate, training explicitly
constructs a differentiable S101 KV-cache: Green Tunnel ON ONLY for the
prompt prefill and OFF for the 2 teacher-forced cached continuation
steps (noun and EOS). All proposals are physically FP16 saved/reloaded
and 456 original TRAIN correct full+EOS identities protected with zero
forgetting; 128 new S122 TRAIN is development, not held-out QA.
No model promoted, accepted, or deployed.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from calibri2.s125.j7_atomic_physical_retention import PhysicalRetentionCheck,atomic_train_block
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


STEPS=2
TRANSACTION_STEPS=2
LEARNING_RATE=0.00125
NEW_PAIRS_PER_STEP=3
PROTECTED_EXTRA_PER_STEP=2
GATE_CLAMP=0.05
FIXED_FP16_EVAL_SCALES=(0.35,1.00)
KEY="conditional_dual_bridge.coefficients"
SEED=1270127


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



def paired_speed_S127(state,rows,keys,routes,tok,bank,copy_net,full,bridge,
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
                raise AssertionError("S127 CPU baseline pair token mismatch")
        def new():
            with torch.inference_mode():
                with prefill_only_tunnel(model,full):
                    with frozen_cpu_phase_scope(
                        model,bridge,route,feat,phase="prefill_only"):
                        start=time.perf_counter()
                        out,_=cached_full_cpu_with_neural_copy(
                            model,sources,first,later,head,tok,q,pointer,copy_net)
                        after.append(time.perf_counter()-start)
            if out!=proposed[key]:
                raise AssertionError("S127 CPU proposal pair token mismatch")
        if i%2:
            new();base()
        else:
            base();new()
    return {"ratio":statistics.median(before)/statistics.median(after),
        "paired_examples":SPEED_SAMPLE,
        "baseline_median_seconds":statistics.median(before),
        "candidate_median_seconds":statistics.median(after),
        "same_host_alternating":True}

def main():
    p=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S101_full456_report",
        "S122_curriculum_report","S112_report","S87_report","S91_report",
        "S93_report",
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
    if len(extra)<STEPS*PROTECTED_EXTRA_PER_STEP:
        raise AssertionError("S125 insufficient original correct S93 replay pool")
    # Round-robin diverse old TRAIN families (zero QA labels).
    by_family=defaultdict(list)
    for k in extra:by_family[k.split("/",1)[0]].append(k)
    rotating=[]
    while len(rotating)<STEPS*PROTECTED_EXTRA_PER_STEP:
        for fam in sorted(by_family):
            if len(rotating)>=STEPS*PROTECTED_EXTRA_PER_STEP:break
            if by_family[fam]:
                rotating.append(by_family[fam].pop(0))
    if len(rotating)!=STEPS*PROTECTED_EXTRA_PER_STEP:
        raise AssertionError("Cannot construct diverse original TRAIN protection")

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
    zero_q2=export_compact_q2(full)
    load_compact_q2(full,zero_q2)
    parameters=restrict_to_gate_only(full)
    if not torch.equal(full.gates.detach().cpu(),torch.zeros(12)):
        raise AssertionError("S127 packed baseline not zero initialized")
    full.install(student)
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
        try:
            for block in range(STEPS//TRANSACTION_STEPS):
                last_record=None
                def train_block():
                    for offset in range(TRANSACTION_STEPS):
                        step=block*TRANSACTION_STEPS+offset
                        pairs=[selected_pairs[
                            (step*NEW_PAIRS_PER_STEP+i)%len(selected_pairs)]
                            for i in range(NEW_PAIRS_PER_STEP)]
                        new_batch=[row for pair in pairs for row in pair]
                        old_batch=[historical[k] for k in sorted(fragile)]
                        old_batch += [historical[k] for k in
                            rotating[step*PROTECTED_EXTRA_PER_STEP:
                                     (step+1)*PROTECTED_EXTRA_PER_STEP]]
                        optimizer.zero_grad(set_to_none=True)
                        losses=[]
                        for row in new_batch:
                            clear(sources)
                            loss=pair_causal_train_gradient(
                                student,bridge,full,tok,row,first,later,pointer,head,
                                protected=False)
                            (0.60/len(new_batch)*loss).backward()
                            losses.append(float(loss.detach()))
                            clear(sources)
                        for row in old_batch:
                            clear(sources)
                            loss=pair_causal_train_gradient(
                                student,bridge,tok,row,first,later,pointer,head,
                                protected=True)
                            (0.40/len(old_batch)*loss).backward()
                            losses.append(float(loss.detach()))
                            clear(sources)
                        grad=full.gates.grad
                        if grad is None or not bool(torch.isfinite(grad).all()) or int(torch.count_nonzero(grad))<1:
                            raise AssertionError("S127 missing genuine 12D full GT MPS gradient")
                        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.0))
                        optimizer.step()
                        with torch.no_grad():
                            full.gates.clamp_(-GATE_CLAMP,GATE_CLAMP)
                        torch.mps.synchronize()
                        trace={
                            "step":step+1,"block":block+1,
                            "new_original_S122_TRAIN_examples":len(new_batch),
                            "protected_original_S101_TRAIN_examples":len(old_batch),
                            "mean_loss":sum(losses)/len(losses),
                            "nonzero_grad_coefficients":int(torch.count_nonzero(grad)),
                            "gradient_norm":norm,
                            "max_abs_coefficient":float(full.gates.abs().max()),
                        }
                        traces.append(trace)
                        print(json.dumps({"event":"S127_REAL_MPS_Q2_J7_GUARDED_STEP",**trace}),flush=True)
                    return TRANSACTION_STEPS

                def persisted_cpu_guard():
                    nonlocal last_record
                    index=len(physical)+1
                    learned=full.gates.detach().to("cpu").clone()
                    if not bool(torch.isfinite(learned).all()):
                        raise AssertionError("S127 nonfinite Q2 gate after MPS training")
                    with torch.no_grad():
                        cpu_full.gates.copy_(learned)
                    paid=dict(side)
                    paid.update(export_compact_q2(cpu_full))
                    if any(not torch.equal(paid[k],zero_q2[k]) for k in zero_q2
                           if k!=PREFIX+"gates_fp16"):
                        raise AssertionError("S127 frozen Q2 rank4 projection or mix changed")
                    label=f"block{block+1}_check{index}"
                    fp16=a.out/f"S127_UNACCEPTED_{label}.safetensors"
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
                        raise AssertionError("S127 actual physical Q2 reload failed")
                    cfg={
                        "stage":"S127_Q2_REAL_J7_GATE_ONLY",
                        "gate_phase":"prefill_only",
                        "rank":4,"bits":2,"layers":12,
                        "old_donors":[3,9],
                        "S111_coefficients":"frozen_zero_unstored",
                        "frozen_Q2_projections":True,
                    }
                    cfg_path=a.out/f"S127_UNACCEPTED_{label}_CONFIG.json"
                    cfg_path.write_text(json.dumps(
                        cfg,sort_keys=True,separators=(",",":"))
                    )
                    extra_bytes=fp16.stat().st_size-a.S93_sidecar.stat().st_size+cfg_path.stat().st_size
                    bpp=8*(float(p101["S99_all_inclusive_physical_bpp"])
                           *SOURCE_PARAMS/8+extra_bytes)/(SOURCE_PARAMS+
                           cpu_full.new_parameter_count())
                    with prefill_only_tunnel(cpu[0],cpu_full):
                        original_prediction=candidate_actual_CPU(
                            cpu,historical,histkeys,histroutes,tok,bank,
                            copy_net,cpu_bridge,"prefill_only")
                    speed=paired_speed_S127(
                        cpu,historical,histkeys,histroutes,tok,bank,copy_net,
                        cpu_full,cpu_bridge,before,original_prediction)
                    guard=strict_train_retention(
                        expected,before,original_prediction,
                        physical_bpp=bpp,cpu_ratio=speed["ratio"],speed_required=True)
                    with prefill_only_tunnel(cpu[0],cpu_full):
                        new_prediction=candidate_actual_CPU(
                            cpu,new,newkeys,new_routes,tok,bank,
                            copy_net,cpu_bridge,"prefill_only")
                    new_score=score_full_and_pairs(new,new_prediction)
                    results={
                        "block":block+1,"check":index,
                        "original_S101_456_correct_full":sum(
                            original_prediction[k]==expected[k] for k in histkeys),
                        "original_S101_456_correct_first":sum(
                            bool(original_prediction[k]) and
                            original_prediction[k][0]==expected[k][0]
                            for k in histkeys),
                        "lost_original_full":guard["lost_full"],
                        "lost_original_first":guard["lost_first"],
                        "gained_original_full":guard["gained_full"],
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
                    }
                    guard_evidence=PhysicalRetentionCheck(
                        original_456_checked=len(histkeys),
                        original_full_correct=len(orig_full),
                        original_first_correct=len(orig_first),
                        lost_original_full=guard["lost_full"],
                        lost_original_first=guard["lost_first"],
                        newly_correct_original_full=guard["gained_full"],
                        new_S122_train_full=new_score["full"],
                        new_S122_train_both_correct_pairs=
                            new_score["counterfactual_pairs_both_correct"],
                        physical_all_inclusive_bpp=bpp,
                        paired_original_S101_CPU_speed_ratio=speed["ratio"],
                        physical_FP16_reload_verified=roundtrip,
                        immutable_original_S101_bytes_verified=originals,
                        S95_S98_unchanged=True,
                    )
                    results["strict_no_forgetting_guard"]=guard_evidence.valid()
                    physical.append(results)
                    last_record=results
                    print(json.dumps({
                        "event":"S127_J7_REAL_Q2_PHYSICAL_456_128_CPU_TRANSACTION",
                        **{k:v for k,v in results.items()
                           if not k.endswith("SHA256")},
                    }),flush=True)
                    return guard_evidence

                result=atomic_train_block(
                    parameters,optimizer,train_block,
                    persisted_cpu_guard,reduction_factor=0.5)
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
                    "event":"S127_J7_ATOMIC_ADAM_TRANSACTION_END",**tx,
                }),flush=True)
        finally:
            cpu_full.uninstall()
            cpu_bridge.uninstall()
    finally:
        cpu[6].uninstall()
        full.uninstall()
        bridge.uninstall()
        inner.uninstall()

    report={
        "stage":"S127_REAL_MPS_Q2_RANK4_12LAYER_GATE_ONLY_J7",
        "original_S101_456_full":353,
        "original_S101_456_first":387,
        "new_S122_128_original_baseline":new_base_stats,
        "S122_new_TRAIN_64_counterfactual_pairs":64,
        "old_protected_original_S101_7_fragile_replayed_per_step":True,
        "real_MPS_12_full_layer_gate_coefficients_only":True,
        "Q2_matrices_frozen_2bits":True,
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
        "no_main_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    (a.out/"S127_REAL_Q2_J7_MPS_PREFILL_456_128_CPU_TRANSACTION_REPORT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
    print(json.dumps({
        "event":"S127_Q2_REAL_MPS_J7_TRANSACTION_COMPLETE",
        "committed_blocks":report["atomic_committed_blocks"],
        "rolled_back_blocks":report["atomic_rollback_blocks"],
        "safe_development_TRAIN_improvement":best_safe is not None,
        "candidate_accepted":False,
        "never_used_QA":True,
    }),flush=True)


if __name__=="__main__":
    main()
