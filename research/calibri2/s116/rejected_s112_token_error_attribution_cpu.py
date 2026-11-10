"""S116: causal S112 forgotten full-answer token-position / S98 copy attribution.

NO TRAINING, no new neural weights, no LIVE/main/Render/Supabase/APK.
The S112 FP16 340-parameter candidate has been REJECTED (+1/−7 full,
+1/−0 first tokens). Study why exactly seven old complete answers are
forgotten by replaying ORIGINAL S101 S95/S98 cached CPU inference twice:
unhooked and with immutable S112 saved trained conditional GT weights.

For gained/lost identities, record first diverging output token POSITION
and changes to S98 copy eligibility/override, never expose QA or actual
answers/prompts in output. Preserve original 353/387 and candidate 347/388,
matching S112 recorded provenance; reject any discrepancy.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s95.train_and_audit_router import prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    load_frozen_numeric_weights,apply_profile)
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s102.hybrid_retention import strict_train_retention
from calibri2.s110.frozen_s101_answer_bottleneck import first_divergence
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import cohort,load_cpu_original

KEY="conditional_dual_bridge.coefficients"


def main():
    p=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S101_full456_report","S111_parity_report",
        "S112_report","S112_rejected_sidecar","S112_runtime_config",
        "S87_report","S91_report","S93_report",
    ):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)

    s99=json.loads(a.S99_full456_report.read_text())
    s101=json.loads(a.S101_full456_report.read_text())
    s111=json.loads(a.S111_parity_report.read_text())
    s112=json.loads(a.S112_report.read_text())
    if (s101.get("cached_full_EOS_correct")!=353
            or s101.get("new_first_token_correct")!=387
            or s111.get("full_456_exact_parity") is not True
            or s111.get("candidate_full_EOS_correct")!=353
            or s112.get("stage")!="S112_REAL_MPS_CONDITIONAL_340_FP16_456_CPU_TRANSACTION"
            or s112.get("candidate_full")!=347
            or s112.get("candidate_first")!=388
            or s112.get("guard",{}).get("lost_full")!=7
            or s112.get("guard",{}).get("gained_full")!=1
            or s112.get("guard",{}).get("lost_first")!=0
            or s112.get("rollback",{}).get("committed") is not False
            or s112.get("rollback",{}).get("rollback_verified") is not True):
        raise AssertionError("S116 needs exact known rejected S112 evidence")
    for path,digest in (
        (a.S93_CB61,s99["immutable_S95_source_sha256"]),
        (a.S94_bank,s99["S94_weightbank_sha256"]),
        (a.S95_router,s99["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net,s99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes())!=digest:
            raise AssertionError("Original S101 physical source changed")
    if (sha(a.S93_CB61.read_bytes())!=s112["S101_original_CB61_sha256"]
            or sha(a.S112_rejected_sidecar.read_bytes())!=s112["S112_sidecar_SHA256"]):
        raise AssertionError("Rejected S112 model bytes mismatch")
    runtime=json.loads(a.S112_runtime_config.read_text())
    if (runtime.get("enable_only_route")!="S93"
            or runtime.get("sidecar_key")!=KEY
            or runtime.get("donor_layers")!=[3,9]):
        raise AssertionError("S112 S95 route/scoped donor descriptor invalid")
    orig_side=load_file(str(a.S93_sidecar))
    reject=load_file(str(a.S112_rejected_sidecar))
    if set(reject)!=set(orig_side)|{KEY}:
        raise AssertionError("Unexpected newly stored S112 tensors")
    for name in orig_side:
        if not torch.equal(reject[name],orig_side[name]):
            raise AssertionError("A frozen original S101 tensor changed")
    if reject[KEY].shape!=(2,10,17) or reject[KEY].dtype!=torch.float16:
        raise AssertionError("S112 FP16 340 matrix shape mismatch")

    _,_,_,bank,router,scorer=load_frozen_numeric_weights(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    reports={p:freeze_report(getattr(a,p+"_report"))
             for p in ("S87","S91","S93")}
    rows,keys,routes=cohort(tok,reports,router)
    gold={key:rows[key]["expected"] for key in keys}
    state=load_cpu_original(
        s87.open_envelope(a.S93_CB61.read_bytes()),orig_side)
    model,sources,first,later,pointer,head,tunnel,centroids,old_tunnel=state
    bridge=PromptConditionedDualTunnel(tunnel).eval()
    bridge.load_physical_gain({KEY:reject[KEY]})
    selected=None
    baseline={}
    modified={}
    old_notes={}
    new_notes={}
    try:
        with torch.inference_mode():
            for key in keys:
                profile=routes[key]
                if profile!=selected:
                    apply_profile(sources,tunnel,profile,bank,centroids,old_tunnel)
                    selected=profile
                question=rows[key]["question"]
                original,old_note=cached_full_cpu_with_neural_copy(
                    model,sources,first,later,head,tok,question,pointer,scorer)
                bridge.install(model)
                try:
                    features=torch.as_tensor(
                        prompt_only_vector(tok,question),dtype=torch.float32)
                    with bridge.prompt_scope(profile,features):
                        candidate,new_note=cached_full_cpu_with_neural_copy(
                            model,sources,first,later,head,tok,
                            question,pointer,scorer)
                finally:
                    bridge.uninstall()
                baseline[key]=original
                modified[key]=candidate
                old_notes[key]=old_note
                new_notes[key]=new_note
    finally:
        bridge.uninstall()
        tunnel.uninstall()
    full_before=sum(baseline[k]==gold[k] for k in keys)
    full_after=sum(modified[k]==gold[k] for k in keys)
    first_before=sum(bool(baseline[k]) and baseline[k][0]==gold[k][0] for k in keys)
    first_after=sum(bool(modified[k]) and modified[k][0]==gold[k][0] for k in keys)
    if (full_before,full_after,first_before,first_after)!=(353,347,387,388):
        raise AssertionError("S116 did not reproduce original rejected S112 actual CPU result")
    guard=strict_train_retention(
        gold,baseline,modified,
        physical_bpp=s112["bpp"],cpu_ratio=1.0,speed_required=True)
    if (guard["lost_full"]!=7 or guard["gained_full"]!=1
            or guard["lost_first"]!=0
            or set(guard["lost_full_identities"])!=set(
                s112["guard"]["lost_full_identities"])
            or set(guard["gained_full_identities"])!=set(
                s112["guard"]["gained_full_identities"])):
        raise AssertionError("S116 identities differ from original rejected S112")
    lost={k for k in keys if baseline[k]==gold[k] and modified[k]!=gold[k]}
    gained={k for k in keys if baseline[k]!=gold[k] and modified[k]==gold[k]}
    cases=Counter()
    profiles=Counter()
    families=Counter()
    notes=Counter()
    output_changed=Counter()
    hashed_cases=[]
    for label,identities in (("lost",lost),("gained",gained)):
        for k in sorted(identities):
            gold_ids=gold[k]
            d=first_divergence(gold_ids,modified[k])
            d_old=first_divergence(gold_ids,baseline[k])
            if label=="lost" and d_old!=-1:
                raise AssertionError("Forgotten baseline was not full correct")
            if label=="gained" and d!=-1:
                raise AssertionError("Gained candidate not exact correct")
            phase=(
                "first" if d==0 else
                "second" if d==1 else
                "final_EOS" if d==len(gold_ids)-1 else
                "later_or_truncation"
            ) if label=="lost" else (
                "old_first" if d_old==0 else
                "old_second" if d_old==1 else
                "old_final_EOS" if d_old==len(gold_ids)-1 else
                "old_later_or_truncation"
            )
            cases[(label,phase)]+=1
            profiles[(label,routes[k])]+=1
            families[(label,k.split("/",1)[0])]+=1
            for field in ("candidate_present","neural_override_applied"):
                notes[(label,field,
                       bool(old_notes[k].get(field)),
                       bool(new_notes[k].get(field)))]+=1
            output_changed[(label,first_divergence(baseline[k],modified[k]))]+=1
            hashed_cases.append({
                "id_sha256":sha(k.encode()),"outcome":label,
                "first_changed_position":first_divergence(baseline[k],modified[k]),
                "correctness_break_position":d if label=="lost" else d_old,
                "source_profile":routes[k],
                "original_S98_copy_eligible":bool(old_notes[k].get("candidate_present")),
                "new_S98_copy_eligible":bool(new_notes[k].get("candidate_present")),
                "original_S98_copy_applied":bool(old_notes[k].get("neural_override_applied")),
                "new_S98_copy_applied":bool(new_notes[k].get("neural_override_applied")),
            })
    report={
        "stage":"S116_TRUE_S101_REJECTED_S112_LOST7_GAINED1_CAUSAL_TOKEN_AUDIT",
        "original_full_EOS":full_before,"rejected_S112_full_EOS":full_after,
        "original_first":first_before,"rejected_S112_first":first_after,
        "original_full_lost":guard["lost_full"],
        "original_full_gained":guard["gained_full"],
        "original_first_lost":guard["lost_first"],
        "lost_vs_gained_first_corruption":{
            f"{k[0]}/{k[1]}":v for k,v in cases.items()},
        "lost_vs_gained_by_profile":{
            f"{k[0]}/{k[1]}":v for k,v in profiles.items()},
        "lost_vs_gained_by_family":{
            f"{k[0]}/{k[1]}":v for k,v in families.items()},
        "S98_copy_eligibility_and_override_changes":{
            f"{k[0]}/{k[1]}/old_{k[2]}/new_{k[3]}":v
            for k,v in notes.items()},
        "output_first_changed_token_position":{
            f"{k[0]}/{k[1]}":v for k,v in output_changed.items()},
        "hashed_TRAIN_identification_for_research_only":hashed_cases,
        "no_new_weights_no_training":True,
        "all_456_original_TRAIN_verified":True,
        "model_accepted":False,
        "sealed_QA_TEST_not_opened":True,
        "main_LIVE_Render_Supabase_APK_unchanged":True,
    }
    (a.out/"S116_REJECTED_S112_TOKEN_POSITION_S98_ATTRIBUTION.json").write_text(
        json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({
        "event":"S116_REJECTED_S112_TOKEN_ATTRIBUTION_456_TRAIN",
        "lost":len(lost),"gained":len(gained),
        "wrong_positions":report["lost_vs_gained_first_corruption"],
        "S98_changes":report["S98_copy_eligibility_and_override_changes"],
        "no_training":True,"model_accepted":False,
    }),flush=True)


if __name__=="__main__":
    main()
