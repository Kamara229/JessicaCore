"""S126 CPU physical stage zero: actual S101 + old GT3/9 + zero S111
plus NEW sequential 12-layer GT; independent exact 456+128 TRAIN parity.

Never trains or accepts a model. S118 QA embargo respected. All 12 hooks
are active in real cached S101 CPU inference even with zero new gates;
physical FP16 storage and CPU slowdown must be measured, not inferred.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s95.train_and_audit_router import prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    apply_profile,load_frozen_numeric_weights)
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s112.prompt_conditioned_first_token_mps import (
    cohort,load_cpu_original,all_456_cached_cpu)
from calibri2.s115.temporal_rejected_s112_full456_cpu import (
    candidate_actual_CPU)
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s122.independent_binding_train_curriculum import materialize_train_only
from calibri2.s125.real_mps_j7_atomic_s122_prefill import (
    frozen_routes,original_new_train_outputs,score_full_and_pairs)
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel


SPEED_SAMPLE=48


def paired_speed(state,rows,keys,routes,tok,bank,copy_net,full,bridge,
                 original,candidate):
    model,sources,first,later,pointer,head,inner,centroids,old_gt=state
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
                duration=time.perf_counter()-start
            finally:
                bridge.install(model)
                full.install(model)
            if out!=original[key]:
                raise AssertionError("Original S101 paired 456 token drift")
            before.append(duration)
        def new():
            with full.scope(),bridge.prompt_scope(route,feat):
                start=time.perf_counter()
                # With zero new gates, full.active is safe even on cached
                # continuation, but trained S126 MUST be prefill-only.
                out,_=cached_full_cpu_with_neural_copy(
                    model,sources,first,later,head,tok,q,pointer,copy_net)
                duration=time.perf_counter()-start
            if out!=candidate[key]:
                raise AssertionError("S126 zero candidate paired token drift")
            after.append(duration)
        if i%2:
            new();base()
        else:
            base();new()
    return {"paired_examples":SPEED_SAMPLE,
            "baseline_median_seconds":statistics.median(before),
            "candidate_median_seconds":statistics.median(after),
            "same_host_alternating":True,
            "ratio":statistics.median(before)/statistics.median(after)}


def main():
    p=argparse.ArgumentParser()
    for name in ("S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_report","S101_report","S122_report",
        "S87_report","S91_report","S93_report"):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    p99=json.loads(a.S99_report.read_text())
    p101=json.loads(a.S101_report.read_text())
    p122=json.loads(a.S122_report.read_text())
    if (p99.get("S99_new_correct_full_EOS")!=353
        or p101.get("cached_full_EOS_correct")!=353
        or p101.get("new_first_token_correct")!=387
        or p122.get("new_TRAIN_questions")!=128
        or p122.get("new_TRAIN_counterfactual_pairs")!=64):
        raise AssertionError("Original S101/S122 source reports invalid")
    for path,digest in (
        (a.S93_CB61,p99["immutable_S95_source_sha256"]),
        (a.S94_bank,p99["S94_weightbank_sha256"]),
        (a.S95_router,p99["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net,p99["S98_neural_copy_selector_sha256"])):
        if sha(path.read_bytes())!=digest:
            raise AssertionError("Immutable original compressed source SHA changed")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new_rows,new_digest,_=materialize_train_only(tok)
    if new_digest!=p122["new_TRAIN_question_hashset_sha256"]:
        raise AssertionError("S122 TRAIN dataset hash mismatch")
    new={r["id"]:r for r in new_rows}
    newkeys=sorted(new)
    _,_,_,bank,router,copy_net=load_frozen_numeric_weights(a)
    newroutes=frozen_routes(tok,router,new)
    reports={nm:freeze_report(getattr(a,nm+"_report")) for nm in ("S87","S91","S93")}
    rows,keys,routes=cohort(tok,reports,router)
    side=load_file(str(a.S93_sidecar))
    state=load_cpu_original(s87.open_envelope(a.S93_CB61.read_bytes()),side)
    model=state[0]
    bridge=None
    full=None
    try:
        old,_=all_456_cached_cpu(state,rows,keys,routes,tok,bank,copy_net)
        expected={key:rows[key]["expected"] for key in keys}
        original_full={key for key in keys if old[key]==expected[key]}
        original_first={key for key in keys if old[key] and old[key][0]==expected[key][0]}
        if len(original_full)!=353 or len(original_first)!=387:
            raise AssertionError("Original S101 456 physical baseline changed")
        newbase=original_new_train_outputs(
            state,new,newkeys,newroutes,tok,bank,copy_net)
        newscore=score_full_and_pairs(new,newbase)
        if newscore!={"full":14,"first":39,"counterfactual_pairs_both_correct":0}:
            raise AssertionError("Frozen S123 original S101 S122 TRAIN changed")
        bridge=PromptConditionedDualTunnel(state[6]).eval().install(model)
        full=FullLayerStateTunnel(width=320,depth=12,rank=8).eval().install(model)
        # This proves actual S101 full autoregressive CPU for original
        # S87/S91/S93 456 and new S122 128, not synthetic fake layers.
        with full.scope():
            proposed=candidate_actual_CPU(
                state,rows,keys,routes,tok,bank,copy_net,
                bridge,"prefill_only")
        if proposed!=old or full.last_forward_coverage!=tuple(range(12)):
            raise AssertionError("S126 zero gate actual S101 full456 parity FAILED")
        with full.scope():
            newproposed=candidate_actual_CPU(
                state,new,newkeys,newroutes,tok,bank,copy_net,
                bridge,"prefill_only")
        if newproposed!=newbase:
            raise AssertionError("S126 new S122 128 physical zero parity FAILED")

        all_tensors=dict(side)
        all_tensors.update(bridge.export_physical_gain())
        all_tensors.update(full.export_fp16())
        physical_path=a.out/"S126_UNACCEPTED_ZERO_GT12_FULL_PHYSICAL.safetensors"
        save_file(all_tensors,str(physical_path))
        reloaded=load_file(str(physical_path))
        if (set(reloaded)!=set(all_tensors)
            or any(not torch.equal(reloaded[k],v) for k,v in all_tensors.items())
            or any(not torch.equal(reloaded[k],v) for k,v in side.items())):
            raise AssertionError("Original full FP16 sidecar changed")
        config={"stage":"S126_FULL_LAYER_GT12_ZERO_PARITY",
            "existing_donors":[3,9],"new_sequential_layers":list(range(12)),
            "runtime":"NEW_FULL_LAYER_ZERO_GATE_ONLY",
            "context_reset":"every_forward","rank":8,
            "original_S111_phase":"prefill_only",
            "S126_training_permitted":False}
        conf_path=a.out/"S126_UNACCEPTED_ZERO_GT12_CONFIG.json"
        conf_path.write_text(json.dumps(config,sort_keys=True,separators=(",",":")))
        # Never omit actual sidecar headers or runtime JSON from bpp.
        extra=physical_path.stat().st_size-a.S93_sidecar.stat().st_size+conf_path.stat().st_size
        all_in_bytes=p101["S99_all_inclusive_physical_bpp"]*SOURCE_PARAMS/8+extra
        full_bpp=8*all_in_bytes/(SOURCE_PARAMS+full.new_parameter_count()+bridge.coefficients.numel())
        speed=paired_speed(state,rows,keys,routes,tok,bank,copy_net,
                           full,bridge,old,proposed)
        accepted_as_training_base=bool(full_bpp<=0.7 and speed["ratio"]>=0.9)
        report={"stage":"S126_REAL_S101_ZERO_FULL_LAYER_GT12",
            "real_original_S101_456_all_token_exact_parity":True,
            "original_S101_full":353,"original_S101_first":387,
            "new_S122_TRAIN_baseline":newscore,
            "real_new_S122_128_exact_parity":True,
            "recurrent_state_through_all_12":True,
            "S126_new_trainable_parameters":full.new_parameter_count(),
            "physical_bpp_with_full_runtime":full_bpp,
            "physical_below_0_7":full_bpp<=0.7,
            "CPU_48_alternating":speed,
            "CPU_at_least_0_9":speed["ratio"]>=0.9,
            "physical_sha256":sha(physical_path.read_bytes()),
            "zero_stage_eligible_for_future_training":accepted_as_training_base,
            "model_trained":False,"model_accepted":False,
            "no_heldout_QA_or_inspected_S118_access":True}
        (a.out/"S126_REAL_S101_FULL456_ZERO_GT12_REPORT.json").write_text(
            json.dumps(report,sort_keys=True,indent=2))
        print(json.dumps({"event":"S126_REAL_ZERO_GT12_CPU_456_128_COMPLETE",
            "full456_token_parity":True,"new128_token_parity":True,
            "physical_bpp":full_bpp,"cpu_ratio":speed["ratio"],
            "research_eligible":accepted_as_training_base,
            "candidate_accepted":False}),flush=True)
    finally:
        if full is not None:full.uninstall()
        if bridge is not None:bridge.uninstall()
        state[6].uninstall()


if __name__=="__main__":
    main()
