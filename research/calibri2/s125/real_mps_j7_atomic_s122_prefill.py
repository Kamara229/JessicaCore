"""CALIBRI S125: REAL MPS prefill-only training on S122 CROSS-COLOR TRAIN.

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
from calibri2.s82.real_mps_generic_token_context_copy import candidate_positions
from v176.operators import clear


STEPS=4
TRANSACTION_STEPS=2
LEARNING_RATE=0.0025
NEW_PAIRS_PER_STEP=3
PROTECTED_EXTRA_PER_STEP=2
GATE_CLAMP=0.05
FIXED_FP16_EVAL_SCALES=(0.35,1.00)
KEY="conditional_dual_bridge.coefficients"
SEED=1250125


def pair_causal_train_gradient(student,bridge,tok,row,first,later,pointer,head,*,protected:bool):
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
    with bridge.prompt_scope("S93",feat):
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
    optimizer=torch.optim.Adam([bridge.coefficients],lr=LEARNING_RATE)
    trace=[]
    try:
        for step in range(STEPS):
            # Each original S122 TRAIN color-pair has BOTH owner sides.
            pairs=[selected_pairs[(step*NEW_PAIRS_PER_STEP+i)%len(selected_pairs)]
                   for i in range(NEW_PAIRS_PER_STEP)]
            new_batch=[x for pair in pairs for x in pair]
            old_batch=[historical[k] for k in sorted(fragile)]
            old_batch.extend(historical[k] for k in
                             rotating[step*PROTECTED_EXTRA_PER_STEP:
                                      (step+1)*PROTECTED_EXTRA_PER_STEP])
            optimizer.zero_grad(set_to_none=True)
            values=[]
            for row in new_batch:
                clear(sources)
                loss=pair_causal_train_gradient(
                    student,bridge,tok,row,first,later,pointer,head,
                    protected=False)
                (0.60/len(new_batch)*loss).backward()
                values.append(float(loss.detach()))
                clear(sources)
            for row in old_batch:
                clear(sources)
                loss=pair_causal_train_gradient(
                    student,bridge,tok,row,first,later,pointer,head,
                    protected=True)
                (0.40/len(old_batch)*loss).backward()
                values.append(float(loss.detach()))
                clear(sources)
            grad=bridge.coefficients.grad
            if (grad is None or not bool(torch.isfinite(grad).all())
                    or int(torch.count_nonzero(grad))<100):
                raise AssertionError("S125 missing substantial real MPS 340D gradient")
            grad_norm=float(torch.nn.utils.clip_grad_norm_(
                [bridge.coefficients],1.0))
            optimizer.step()
            with torch.no_grad():
                bridge.coefficients.clamp_(-GATE_CLAMP,GATE_CLAMP)
            torch.mps.synchronize()
            stepinfo={
                "step":step+1,
                "new_authorized_S122_TRAIN_examples":len(new_batch),
                "old_protected_original_S101_examples":len(old_batch),
                "mean_unweighted_loss":sum(values)/len(values),
                "gradient_nonzero_coefficients":int(torch.count_nonzero(grad)),
                "gradient_norm":grad_norm,
                "maximum_abs_coefficient":float(bridge.coefficients.abs().max()),
            }
            trace.append(stepinfo)
            print(json.dumps({"event":"S125_REAL_MPS_PREFILL_ONLY_UPDATE",**stepinfo}),flush=True)
    finally:
        bridge.uninstall()
        inner.uninstall()
    learned=bridge.coefficients.detach().cpu().clone()
    if (not bool(torch.isfinite(learned).all())
            or not bool(torch.count_nonzero(learned))):
        raise AssertionError("S125 real MPS zero/no new learned conditional gates")

    cpu=load_cpu_original(raw,side)
    physical=[]
    best=None
    try:
        base_old,_=all_456_cached_cpu(
            cpu,historical,histkeys,histroutes,tok,bank,copy_net)
        if base_old!=before:
            raise AssertionError("S125 second physical S101 original 456 outputs differ")
        cpu_bridge=PromptConditionedDualTunnel(
            cpu[6],total_layers=12,donor_layers=(3,9)).eval().install(cpu[0])
        try:
            for scale in FIXED_FP16_EVAL_SCALES:
                with torch.no_grad():
                    cpu_bridge.coefficients.copy_(learned*scale)
                physical_tensor=cpu_bridge.export_physical_gain()
                candidate=dict(side)
                candidate.update(physical_tensor)
                suffix=str(scale).replace(".","p")
                sidecar=a.out/f"S125_UNACCEPTED_S122_PREFILL_{suffix}_S93_SIDECAR.safetensors"
                save_file(candidate,str(sidecar))
                reloaded=load_file(str(sidecar))
                if (set(candidate)!=set(reloaded) or
                        any(not torch.equal(candidate[k],reloaded[k]) for k in candidate)):
                    raise AssertionError("S125 physical S101 + FP16 coefficient sidecar not lossless")
                if any(not torch.equal(side[k],reloaded[k]) for k in side):
                    raise AssertionError("Original S101 source weights modified")
                cpu_bridge.load_physical_gain({KEY:reloaded[KEY]})
                cfg={
                    "stage":"S125_UNACCEPTED_REAL_MPS_CROSS_COLOR_PREFILL_ONLY",
                    "green_tunnel_kv_phase":"prefill_only",
                    "enable_only_route":"S93",
                    "donor_layers":[3,9],
                    "feature_source":"frozen_S95_prompt_only_vector",
                    "fixed_projection":"sum_16x32_L2_normalized",
                    "sidecar_key":KEY,
                    "dtype":"float16",
                }
                configpath=a.out/f"S125_UNACCEPTED_S122_PREFILL_{suffix}_CONFIG.json"
                configpath.write_text(json.dumps(
                    cfg,sort_keys=True,separators=(",",":"))
                )
                extra_bytes=(sidecar.stat().st_size-
                             a.S93_sidecar.stat().st_size+
                             configpath.stat().st_size)
                bpp=float(p101["S99_all_inclusive_physical_bpp"])+(
                    8*extra_bytes/SOURCE_PARAMS)
                if not math.isfinite(bpp) or bpp>0.7:
                    raise AssertionError("S125 exceeds all-inclusive physical 0.7 bpp")

                proposed=candidate_actual_CPU(
                    cpu,historical,histkeys,histroutes,tok,bank,
                    copy_net,cpu_bridge,"prefill_only")
                speed=paired_speed_original_vs_phase(
                    cpu,historical,histkeys,histroutes,tok,bank,
                    copy_net,cpu_bridge,"prefill_only",base_old,proposed)
                guard=strict_train_retention(
                    expected,base_old,proposed,physical_bpp=bpp,
                    cpu_ratio=speed["ratio"],speed_required=True)
                new_proposed=candidate_actual_CPU(
                    cpu,new,newkeys,new_routes,tok,bank,
                    copy_net,cpu_bridge,"prefill_only")
                new_stats=score_full_and_pairs(new,new_proposed)
                old_correct_full=sum(proposed[k]==expected[k] for k in histkeys)
                old_correct_first=sum(
                    bool(proposed[k]) and proposed[k][0]==expected[k][0]
                    for k in histkeys)
                is_safe=bool(guard["transaction_can_commit"])
                gained_training=(new_stats["full"]>new_base_stats["full"] or
                                 new_stats["counterfactual_pairs_both_correct"]>
                                 new_base_stats["counterfactual_pairs_both_correct"])
                record={
                    "coefficient_scale":scale,
                    "old_original_456_correct_full":old_correct_full,
                    "old_original_456_correct_first":old_correct_first,
                    "old_lost_full":guard["lost_full"],
                    "old_lost_first":guard["lost_first"],
                    "old_gained_full":guard["gained_full"],
                    "old_protected_no_forgetting":is_safe,
                    "new_S122_128_authorized_TRAIN_full":new_stats["full"],
                    "new_S122_128_authorized_TRAIN_first":new_stats["first"],
                    "new_S122_64_pairs_both_full":new_stats["counterfactual_pairs_both_correct"],
                    "new_TRAIN_full_or_pairs_improved_vs_original":gained_training,
                    "physical_all_inclusive_bpp":bpp,
                    "paired_CPU_ratio_vs_original_S101":speed["ratio"],
                    "physical_sidecar_sha256":sha(sidecar.read_bytes()),
                    "physical_runtime_config_sha256":sha(configpath.read_bytes()),
                    "train_candidate_for_future_independent_validation":
                        is_safe and gained_training,
                    "model_accepted":False,
                }
                physical.append(record)
                print(json.dumps({
                    "event":"S125_CROSS_COLOR_MPS_PREFILL_PHYSICAL_456_128_TRAIN",
                    "scale":scale,"old_full":old_correct_full,
                    "old_lost_full":record["old_lost_full"],
                    "old_lost_first":record["old_lost_first"],
                    "new_128_full":new_stats["full"],
                    "new_64_both_pairs":new_stats["counterfactual_pairs_both_correct"],
                    "train_candidate":record["train_candidate_for_future_independent_validation"],
                    "physical_bpp":bpp,"cpu_ratio":speed["ratio"],
                }),flush=True)
                if record["train_candidate_for_future_independent_validation"] and (
                    best is None or
                    (new_stats["counterfactual_pairs_both_correct"],new_stats["full"])>
                    (best["new_S122_64_pairs_both_full"],best["new_S122_128_authorized_TRAIN_full"])
                ):
                    best=record
        finally:
            cpu_bridge.uninstall()
    finally:
        cpu[6].uninstall()
    report={
        "stage":"S125_REAL_MPS_CAUSAL_PREFILL_CROSS_COLOR_NEW_TRAIN_456_GUARD",
        "MPS_steps":STEPS,
        "new_authorized_S122_TRAIN_pair_examples_used_per_step":2*NEW_PAIRS_PER_STEP,
        "old_S101_protected_full_examples_per_step":7+PROTECTED_EXTRA_PER_STEP,
        "only_340_S111_coefficients_updated":True,
        "same_FROZEN_S95_route_and_S98_copy":True,
        "true_differentiable_S101_KV_train_prefill_only":True,
        "all_456_original_TRAIN_reference_verified":True,
        "original_456_full":353,
        "original_456_first":387,
        "new_S122_128_baseline":new_base_stats,
        "new_S122_frozen_S95_routes":dict(Counter(new_routes.values())),
        "complete_color_pair_training_groups":len(selected_pairs),
        "train_step_trace":trace,
        "physical_variant_results":physical,
        "strictly_protected_NEW_TRAIN_candidate_scale":None if best is None
            else best["coefficient_scale"],
        "NO_heldout_or_once_inspected_QA_used":True,
        "S118_QA_embargoed":True,
        "new_128_are_TRAIN_not_independent_validation":True,
        "all_models_not_accepted":True,
        "no_production_or_main_changes":True,
    }
    (a.out/"S125_CROSS_BINDING_MPS_PREFILL_FULL456_GUARD_REPORT.json").write_text(
        json.dumps(report,sort_keys=True,indent=2))
    print(json.dumps({
        "event":"S125_CROSS_COLOR_MPS_PREFILL_TRAIN_EXPERIMENT_END",
        "new_128_base_full":new_base_stats["full"],
        "safe_new_TRAIN_candidate":report["strictly_protected_NEW_TRAIN_candidate_scale"],
        "variants":len(physical),
        "model_accepted":False,
        "no_QA":True,
    }),flush=True)


if __name__=="__main__":
    main()
