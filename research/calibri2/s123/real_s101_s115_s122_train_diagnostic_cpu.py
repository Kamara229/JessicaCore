"""CALIBRI S123: REAL original S101 cached CPU cross-attribute TRAIN diagnostic.

NO TRAINING, NO new QA. Compare physically frozen original S101 and the
REJECTED-as-generalization S115 prefill-only TRAIN research candidate
on the exact precommitted S122 64 opposite-owner cross-color TRAIN pairs
(128 TRAIN questions). S122 new TRAIN is NOT independent evaluation.

Use ONLY the original S82 TRAIN noun inventory and original S82 TRAIN
name/frame patterns through the exact S122 validated curriculum generator.
Prior one-shot S118 QA was already inspected and MUST NOT be read,
imported, used in hypotheses, or re-evaluated by S123.

Output aggregate route/first/second/EOS and paired-answer counts only,
with exact original S101 456 TRAIN baseline and immutable SHA provenance.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s95.train_and_audit_router import CHOICES,prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import apply_profile,load_frozen_numeric_weights
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import (
    cohort,load_cpu_original,all_456_cached_cpu,
)
from calibri2.s115.cached_phase_gate import frozen_cpu_phase_scope
from calibri2.s122.independent_binding_train_curriculum import (
    materialize_train_only,
)

KEY="conditional_dual_bridge.coefficients"


def error_phase(gold,actual):
    if gold==actual:return "full+EOS"
    if not actual or gold[0]!=actual[0]:return "first-token"
    if len(actual)<2 or gold[1]!=actual[1]:return "second-token"
    if len(actual)<3 or gold[2]!=actual[2]:return "EOS-or-later"
    return "extra-tokens"


def main():
    p=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S101_full456_report",
        "S115_phase_report","S115_prefill_sidecar","S115_prefill_config",
        "S122_curriculum_report","S87_report","S91_report","S93_report",
    ):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    p99=json.loads(a.S99_full456_report.read_text())
    p101=json.loads(a.S101_full456_report.read_text())
    p115=json.loads(a.S115_phase_report.read_text())
    p122=json.loads(a.S122_curriculum_report.read_text())
    if (p101.get("cached_full_EOS_correct")!=353
            or p101.get("new_first_token_correct")!=387
            or p99.get("S99_new_correct_full_EOS")!=353
            or p115.get("stage")!="S115_REJECTED_S112_TEMPORAL_KV_PHASE_REAL_456_TRAIN_CPU"
            or p115.get("safe_with_real_new_full_answers")!="prefill_only"
            or p115.get("model_accepted") is not False
            or p122.get("stage")!="S122_S82_ONLY_DETERMINISTIC_CROSS_COLOR_COUNTERFACTUAL_TRAIN"
            or p122.get("new_TRAIN_questions")!=128
            or p122.get("new_TRAIN_counterfactual_pairs")!=64
            or p122.get("not_QA_or_independent_test") is not True):
        raise AssertionError("S123 prerequisite original S101/S115/S122 data evidence differs")
    case=[x for x in p115["results"] if x["phase"]=="prefill_only"]
    if (len(case)!=1 or case[0]["full_correct"]!=354
            or case[0]["first_correct"]!=388
            or case[0]["lost_full"]!=0
            or case[0]["gained_full"]!=1):
        raise AssertionError("Frozen S115 prefill-only TRAIN candidate provenance changed")
    for path,digest in (
        (a.S93_CB61,p99["immutable_S95_source_sha256"]),
        (a.S94_bank,p99["S94_weightbank_sha256"]),
        (a.S95_router,p99["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net,p99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes())!=digest:
            raise AssertionError("S123 immutable original S101 source SHA changed")
    if (sha(a.S115_prefill_sidecar.read_bytes())!=
            case[0]["physical_FP16_sidecar_SHA256"]
            or sha(a.S115_prefill_config.read_bytes())!=
            case[0]["physical_phase_config_SHA256"]):
        raise AssertionError("S123 original frozen S115 research sidecar SHA changed")
    old_side=load_file(str(a.S93_sidecar))
    side=load_file(str(a.S115_prefill_sidecar))
    config=json.loads(a.S115_prefill_config.read_text())
    if (set(side)!=set(old_side)|{KEY}
            or side[KEY].shape!=(2,10,17)
            or side[KEY].dtype!=torch.float16
            or not bool(torch.isfinite(side[KEY]).all())
            or config.get("green_tunnel_kv_phase")!="prefill_only"
            or config.get("enable_only_route")!="S93"
            or config.get("sidecar_key")!=KEY):
        raise AssertionError("Physical S115 prefill-only paid model ABI changed")
    if any(not torch.equal(side[k],old_side[k]) for k in old_side):
        raise AssertionError("Original S101 weight mutated in S115")
    extra=(a.S115_prefill_sidecar.stat().st_size
           -a.S93_sidecar.stat().st_size
           +a.S115_prefill_config.stat().st_size)
    bpp=float(p101["S99_all_inclusive_physical_bpp"])+8*extra/SOURCE_PARAMS
    if bpp>0.7 or abs(bpp-case[0]["physical_bpp"])>1e-8:
        raise AssertionError("S123 S115 physical bytes mismatch")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new,digest,specs=materialize_train_only(tok)
    if (len(new)!=128 or len(specs)!=64 or
            digest!=p122["new_TRAIN_question_hashset_sha256"]):
        raise AssertionError("S123 did not reconstruct exact precommitted S122 TRAIN only data")

    _,_,_,bank,router,copy_net=load_frozen_numeric_weights(a)
    reports={name:freeze_report(getattr(a,name+"_report"))
             for name in ("S87","S91","S93")}
    oldrows,oldkeys,oldroutes=cohort(tok,reports,router)
    rows={row["id"]:row for row in new}
    keys=sorted(rows)
    features=torch.as_tensor(np.stack([
        prompt_only_vector(tok,rows[k]["question"]) for k in keys]),
        dtype=torch.float32)
    with torch.inference_mode():
        routes={k:CHOICES[int(j)] for k,j in zip(
            keys,router(features).argmax(dim=1).tolist())}
    if set(routes)!=set(rows) or len(set(r["question"] for r in new))!=128:
        raise AssertionError("Frozen S95 route or S122 TRAIN uniqueness broken")

    state=load_cpu_original(
        s87.open_envelope(a.S93_CB61.read_bytes()),old_side)
    model,sources,first,later,pointer,head,tunnel,centroids,old_tunnel=state
    route_stats=defaultdict(Counter)
    errors=defaultdict(Counter)
    by_pair=defaultdict(dict)
    overall=Counter()
    current=None
    try:
        original_old,_=all_456_cached_cpu(
            state,oldrows,oldkeys,oldroutes,tok,bank,copy_net,bridge=None)
        old_correct_full=sum(original_old[k]==oldrows[k]["expected"] for k in oldkeys)
        old_correct_first=sum(
            bool(original_old[k]) and original_old[k][0]==oldrows[k]["expected"][0]
            for k in oldkeys)
        if (old_correct_full,old_correct_first)!=(353,387):
            raise AssertionError("Original S101 real old 456 TRAIN baseline drift")

        bridge=PromptConditionedDualTunnel(tunnel).eval()
        bridge.load_physical_gain({KEY:side[KEY]})
        try:
            for key in keys:
                row=rows[key]
                route=routes[key]
                if route!=current:
                    apply_profile(sources,tunnel,route,bank,centroids,old_tunnel)
                    current=route
                question=row["question"]
                gold=row["expected"]
                with torch.inference_mode():
                    base,base_copy=cached_full_cpu_with_neural_copy(
                        model,sources,first,later,head,tok,question,pointer,copy_net)
                    bridge.install(model)
                    try:
                        with frozen_cpu_phase_scope(
                                model,bridge,route,
                                torch.as_tensor(prompt_only_vector(tok,question),
                                                dtype=torch.float32)
                                if route=="S93" else None,
                                phase="prefill_only"):
                            proposed,proposal_copy=cached_full_cpu_with_neural_copy(
                                model,sources,first,later,head,tok,question,
                                pointer,copy_net)
                    finally:
                        bridge.uninstall()
                good_base=(base==gold)
                good_new=(proposed==gold)
                first_base=bool(base) and base[0]==gold[0]
                first_new=bool(proposed) and proposed[0]==gold[0]
                for name,val in (
                    ("count",1),("original_full",good_base),("S115_full",good_new),
                    ("original_first",first_base),("S115_first",first_new),
                    ("gained_full",not good_base and good_new),
                    ("lost_full",good_base and not good_new),
                    ("gained_first",not first_base and first_new),
                    ("lost_first",first_base and not first_new),
                    ("S98_copy_original_eligible",base_copy["candidate_present"]),
                    ("S98_copy_new_eligible",proposal_copy["candidate_present"]),
                    ("S98_copy_original_override",base_copy["neural_override_applied"]),
                    ("S98_copy_new_override",proposal_copy["neural_override_applied"]),
                ):
                    overall[name]+=int(val)
                    route_stats[route][name]+=int(val)
                errors[route]["original_"+error_phase(gold,base)]+=1
                errors[route]["S115_"+error_phase(gold,proposed)]+=1
                pair=row["pair_id"]
                sidekey=row["side"]
                by_pair[pair][sidekey]=(good_base,good_new)
        finally:
            bridge.uninstall()
    finally:
        tunnel.uninstall()
    if (overall["count"]!=128 or len(by_pair)!=64
            or any(set(x)!={"left","right"} for x in by_pair.values())):
        raise AssertionError("S123 missing original 64 TRAIN counterfactual owner pairs")
    paired=Counter()
    for pair in by_pair.values():
        paired["original_both_correct"]+=int(pair["left"][0] and pair["right"][0])
        paired["S115_both_correct"]+=int(pair["left"][1] and pair["right"][1])
    report={
        "stage":"S123_REAL_S101_VS_REJECTED_S115_S122_NEW128_TRAIN_DIAG",
        "S122_new_64_counterfactual_pairs_reconstructed":True,
        "original_456_full_EOS":353,
        "original_456_first":387,
        "S122_128_new_train_only":True,
        "historical_456_and_new_128_train_disjoint":True,
        "new_TRAIN_question_SHA256_set_digest":digest,
        "frozen_S95_routes":dict(Counter(routes.values())),
        "original_vs_S115_S122_TRAIN":dict(overall),
        "per_frozen_S95_route":{k:dict(v) for k,v in route_stats.items()},
        "first_second_EOS_error_decomposition_by_route":{
            k:dict(v) for k,v in errors.items()},
        "all_64_two_way_counterfactual_pair_score":dict(paired),
        "S115_physical_all_inclusive_bpp":bpp,
        "S115_not_accepted_due_to_prior_independent_holdout_negative":True,
        "existing_and_new_QA_never_read":True,
        "S118_inspected_QA_embargoed":True,
        "no_new_weights_no_training":True,
        "no_raw_TRAIN_questions_or_targets_logged":True,
        "main_LIVE_Jessica_Render_Supabase_APK_untouched":True,
    }
    (a.out/"S123_CROSS_BINDING_TRAIN_ONLY_S101_VS_S115_DIAG.json").write_text(
        json.dumps(report,sort_keys=True,indent=2))
    print(json.dumps({
        "event":"S123_CROSS_BINDING_128_NEW_TRAIN_DIAGNOSTIC",
        "frozen_routes":report["frozen_S95_routes"],
        "original_full":overall["original_full"],
        "S115_full":overall["S115_full"],
        "original_first":overall["original_first"],
        "S115_first":overall["S115_first"],
        "original_both_pairs":paired["original_both_correct"],
        "S115_both_pairs":paired["S115_both_correct"],
        "model_accepted":False,
        "no_QA":True,
    }),flush=True)


if __name__=="__main__":
    main()
