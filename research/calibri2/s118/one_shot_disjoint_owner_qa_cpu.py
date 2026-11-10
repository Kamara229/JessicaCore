"""CALIBRI S118: genuinely fresh, one-shot independent synthetic QA for S115.

S115 is FROZEN before this QA is generated and evaluated. Compare its
physical FP16 340-coefficient S93 prefill-ONLY Green Tunnel and original
immutable S101 on 64 new owner-binding counterfactual QA pairs (128
full answers). No QA used to train/choose model, NO reinspection of older
sealed QA/TEST, no acceptance/promotion, no LIVE/main changes.

Generate new frames, names and nouns unseen in the 456 original TRAIN
questions. Include same-color and different-color assignments. Score
full response+EOS and first token, both sides of each counterfactual
pair. Log aggregate only, never output QA text/targets/identity lookup.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s95.train_and_audit_router import CHOICES,prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import (
    apply_profile,load_frozen_numeric_weights)
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import load_cpu_original
from calibri2.s115.cached_phase_gate import frozen_cpu_phase_scope

SEED=1180118
KEY="conditional_dual_bridge.coefficients"
PAIRS=64
QA_EXTRAS=(
    "camera","cactus","planet","copper","lemon","orange","butter",
    "bucket","paper","rocket","whistle","pillow","basket","rabbit",
    "coconut","violin","bridge","tunnel","engine","needle","castle",
    "turtle","lantern","pocket","canyon","bucket","meadow","beacon",
    "marble","guitar","helmet","feather","saddle","puzzle","shadow",
    "bottle","cushion","harbor","painter","market","garden","ribbon",
)
QA_NAMES=(
    ("Rilom","Padel"),("Mireth","Veskar"),("Lunito","Harven"),
    ("Dervas","Panilo"),("Kelmar","Nuvin"),("Astrel","Covin"),
    ("Raveto","Delpin"),("Morlin","Fentis"),
)
QA_FRAMES=(
    "Archive ledger: {a} was assigned {left}, while {b} was assigned {right}. Give the exact two-word entry for {a}.",
    "Distribution sheet: {b} has the label {right}. In a different row, {a} has {left}. Write precisely the two words belonging to {a}.",
    "Workshop register: the item logged under {a} reads {left}; the item under {b} reads {right}. What is the two-word entry for {a}?",
    "Parcel log: {b}'s parcel is marked {right}. {a}'s parcel is marked {left}. State only {a}'s two-word label.",
)


def new_heldout_pairs(tok):
    """Deterministic QA recipe pre-committed before any S115 evaluation.

    Exact S82 TRAIN uses accepted[:24] original words and four entirely
    different TRAIN frames/names. No model outputs influence this recipe.
    """
    prior,used_words=s82.newly_authored_train(tok)
    assert len(prior)==128 and len(used_words)==24
    used=set(used_words)
    colors=tuple(s30.TRAIN_ELIGIBLE)
    pool=[]
    for noun in s82.NOUNS+QA_EXTRAS:
        if noun in used or noun in pool:continue
        if all(len(tok.encode(f"{color} {noun}",add_special_tokens=False))==2
               for color in colors):
            pool.append(noun)
    if len(pool)<12:
        raise AssertionError("S118 insufficient strictly disjoint two-token QA nouns")
    old_questions={
        sha(pair[side]["question"].encode())
        for family,pairs in s87.families(tok).items()
        for pair in pairs for side in ("left","right")
    }
    assert len(old_questions)==456
    new_rows=[]
    ids=set()
    for i in range(PAIRS):
        frame=QA_FRAMES[i%len(QA_FRAMES)]
        a,b=QA_NAMES[(i//4)%len(QA_NAMES)]
        ca=colors[i%4]
        cb=ca if i<PAIRS//2 else colors[(i%4+1+(i//4)%3)%4]
        if i>=PAIRS//2 and ca==cb:
            raise AssertionError("Distinct-color QA category not actually distinct")
        one=f"{ca} {pool[(i*3)%len(pool)]}"
        two=f"{cb} {pool[(i*3+7)%len(pool)]}"
        if one==two:
            raise AssertionError("Counterfactual QA labels must differ")
        samples=[]
        for side,left,right in (("left",one,two),("right",two,one)):
            question=frame.format(a=a,b=b,left=left,right=right)
            name_sha=sha(question.encode())
            if name_sha in ids or name_sha in old_questions:
                raise AssertionError("S118 QA prompt duplicates existing TRAIN or another QA prompt")
            ids.add(name_sha)
            target=tok.encode(left,add_special_tokens=False)+[tok.eos_token_id]
            if len(target)!=3 or target[0]==tok.eos_token_id:
                raise AssertionError("QA label is not exact two-token phrase+EOS")
            samples.append({
                "prompt":question,
                "gold":target,
                "hash":name_sha,
                "mode":"same_color" if i<PAIRS//2 else "cross_color",
                "pair_id":i,
                "side":side,
            })
        new_rows.extend(samples)
    if len(new_rows)!=128 or len(ids)!=128:
        raise AssertionError("Exactly 128 unseen QA questions are mandatory")
    # A report contains only one seed/hash and aggregate performance;
    # the QA strings and labels exist in memory ONLY for this single test.
    digest=sha("\n".join(sorted(ids)).encode())
    return new_rows,digest,len(pool)


def main():
    p=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report","S95_router","S95_report",
        "S94_bank","S94_report","S93_CB61","S93_sidecar",
        "S99_full456_report","S101_full456_report",
        "S115_phase_report","S115_prefill_sidecar","S115_prefill_config",
    ):
        p.add_argument(name,type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    p99=json.loads(a.S99_full456_report.read_text())
    p101=json.loads(a.S101_full456_report.read_text())
    p115=json.loads(a.S115_phase_report.read_text())
    if (p101.get("cached_full_EOS_correct")!=353
            or p101.get("new_first_token_correct")!=387
            or p99.get("S99_new_correct_full_EOS")!=353
            or p115.get("stage")!="S115_REJECTED_S112_TEMPORAL_KV_PHASE_REAL_456_TRAIN_CPU"
            or p115.get("safe_with_real_new_full_answers")!="prefill_only"
            or p115.get("max_safe_new_full_EOS_answers")!=1
            or p115.get("model_accepted") is not False):
        raise AssertionError("S118 requires frozen fully TRAIN-guarded S115 research candidate")
    chosen=[x for x in p115["results"] if x["phase"]=="prefill_only"]
    if len(chosen)!=1:
        raise AssertionError("Exactly one immutable S115 prefill candidate required")
    evidence=chosen[0]
    if (evidence.get("full_correct")!=354
            or evidence.get("first_correct")!=388
            or evidence.get("lost_full")!=0
            or evidence.get("lost_first")!=0
            or evidence.get("gained_full")!=1
            or evidence.get("train_retention_safe") is not True
            or evidence.get("physical_bpp",1)>0.7
            or evidence.get("CPU_relative_original_S101",0)<0.90):
        raise AssertionError("S115 prefill-only quality/physical TRAIN evidence invalid")
    for path,digest in (
        (a.S93_CB61,p99["immutable_S95_source_sha256"]),
        (a.S94_bank,p99["S94_weightbank_sha256"]),
        (a.S95_router,p99["S95_neural_profile_selector_sha256"]),
        (a.S98_copy_net,p99["S98_neural_copy_selector_sha256"]),
    ):
        if sha(path.read_bytes())!=digest:
            raise AssertionError("Immutable original S101 source SHA changed")
    if (sha(a.S115_prefill_sidecar.read_bytes())!=
            evidence["physical_FP16_sidecar_SHA256"]
            or sha(a.S115_prefill_config.read_bytes())!=
            evidence["physical_phase_config_SHA256"]):
        raise AssertionError("Selected frozen prefill-only candidate SHA mismatch")

    source_side=load_file(str(a.S93_sidecar))
    adopted=load_file(str(a.S115_prefill_sidecar))
    cfg=json.loads(a.S115_prefill_config.read_text())
    if (set(adopted)!=set(source_side)|{KEY}
            or cfg.get("green_tunnel_kv_phase")!="prefill_only"
            or cfg.get("enable_only_route")!="S93"
            or cfg.get("sidecar_key")!=KEY
            or cfg.get("donor_layers")!=[3,9]
            or adopted[KEY].shape!=(2,10,17)
            or adopted[KEY].dtype!=torch.float16):
        raise AssertionError("S115 original FP16 conditional model ABI invalid")
    for k in source_side:
        if not torch.equal(adopted[k],source_side[k]):
            raise AssertionError("S115 changed original frozen S101 weights")
    physical_added=(a.S115_prefill_sidecar.stat().st_size
                    -a.S93_sidecar.stat().st_size
                    +a.S115_prefill_config.stat().st_size)
    bpp=float(p101["S99_all_inclusive_physical_bpp"])+(
        8*physical_added/SOURCE_PARAMS)
    if abs(bpp-evidence["physical_bpp"])>1e-8 or bpp>0.7:
        raise AssertionError("Independent physical S115 bpp accounting mismatch")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    # This is ONE fresh QA run. DO NOT reuse results to train/tune S115.
    cohort,digest,number_nouns=new_heldout_pairs(tok)
    _,_,_,bank,router,scorer=load_frozen_numeric_weights(a)
    fv=torch.as_tensor(
        np.stack([prompt_only_vector(tok,x["prompt"]) for x in cohort]),
        dtype=torch.float32)
    with torch.inference_mode():
        classes=router(fv).argmax(dim=1).tolist()
    routes=[CHOICES[int(v)] for v in classes]
    cpu=load_cpu_original(s87.open_envelope(a.S93_CB61.read_bytes()),
                          source_side)
    model,sources,first,later,pointer,head,tunnel,original_core,original_gt=cpu
    bridge=PromptConditionedDualTunnel(tunnel).eval()
    bridge.load_physical_gain({KEY:adopted[KEY]})
    stats=defaultdict(Counter)
    route_stats=defaultdict(Counter)
    by_pair=defaultdict(dict)
    source_speeds=[]
    target_speeds=[]
    current=None
    try:
        with torch.inference_mode():
            for i,(item,route) in enumerate(zip(cohort,routes)):
                if route!=current:
                    apply_profile(sources,tunnel,route,bank,original_core,original_gt)
                    current=route
                q=item["prompt"]
                def baseline():
                    t0=time.perf_counter()
                    out,_=cached_full_cpu_with_neural_copy(
                        model,sources,first,later,head,tok,q,pointer,scorer)
                    return out,time.perf_counter()-t0
                def proposed():
                    features=fv[i] if route=="S93" else None
                    with frozen_cpu_phase_scope(
                            model,bridge,route,features,phase="prefill_only"):
                        t0=time.perf_counter()
                        out,_=cached_full_cpu_with_neural_copy(
                            model,sources,first,later,head,tok,q,pointer,scorer)
                        return out,time.perf_counter()-t0
                # Alternating order for fair sampled CPU times.
                if i%2==0:
                    before,tb=baseline()
                    bridge.install(model)
                    try:
                        after,ta=proposed()
                    finally:
                        bridge.uninstall()
                else:
                    bridge.install(model)
                    try:
                        after,ta=proposed()
                    finally:
                        bridge.uninstall()
                    before,tb=baseline()
                if i<48:
                    source_speeds.append(tb);target_speeds.append(ta)
                gold=item["gold"]
                old_full=before==gold
                new_full=after==gold
                old_first=bool(before) and before[0]==gold[0]
                new_first=bool(after) and after[0]==gold[0]
                mode=item["mode"]
                stats[mode]["count"]+=1
                route_stats[route]["count"]+=1
                for name,good in (
                    ("baseline_full",old_full),("candidate_full",new_full),
                    ("baseline_first",old_first),("candidate_first",new_first),
                    ("gained_full",not old_full and new_full),
                    ("lost_full",old_full and not new_full),
                    ("gained_first",not old_first and new_first),
                    ("lost_first",old_first and not new_first),
                    ("tokens_changed",before!=after),
                ):
                    stats[mode][name]+=int(good)
                    route_stats[route][name]+=int(good)
                by_pair[item["pair_id"]][item["side"]]=(old_full,new_full)
    finally:
        bridge.uninstall()
        tunnel.uninstall()

    pairs=defaultdict(Counter)
    for pid,rows in by_pair.items():
        if set(rows)!={"left","right"}:
            raise AssertionError("Both counterfactual sides must be evaluated once")
        mode="same_color" if pid<PAIRS//2 else "cross_color"
        oldboth=rows["left"][0] and rows["right"][0]
        newboth=rows["left"][1] and rows["right"][1]
        pairs[mode]["n_pairs"]+=1
        pairs[mode]["baseline_both_full"]+=int(oldboth)
        pairs[mode]["candidate_both_full"]+=int(newboth)
        pairs[mode]["new_both_pairs"]+=int(newboth and not oldboth)
        pairs[mode]["lost_both_pairs"]+=int(oldboth and not newboth)

    baseline_full=sum(x["baseline_full"] for x in stats.values())
    candidate_full=sum(x["candidate_full"] for x in stats.values())
    old_first=sum(x["baseline_first"] for x in stats.values())
    new_first=sum(x["candidate_first"] for x in stats.values())
    lost=sum(x["lost_full"] for x in stats.values())
    gained=sum(x["gained_full"] for x in stats.values())
    cpu_ratio=statistics.median(source_speeds)/statistics.median(target_speeds)
    if (sum(x["count"] for x in stats.values())!=128
            or sum(x["n_pairs"] for x in pairs.values())!=64
            or len(source_speeds)!=48 or len(target_speeds)!=48
            or not math.isfinite(cpu_ratio)):
        raise AssertionError("S118 once-only fresh QA incomplete")

    report={
        "stage":"S118_ONCE_ONLY_NEW_DISJOINT_128_SYNTHETIC_QA_S101_VS_S115",
        "candidate_frozen_before_QA":True,
        "S115_original_456_TRAIN_safe_full_354_first_388":True,
        "new_QA_seed":SEED,
        "new_QA_prompt_hash_set_SHA256":digest,
        "fresh_noun_vocabulary_size":number_nouns,
        "novel_QA_counterfactual_pairs":PAIRS,
        "novel_QA_questions":len(cohort),
        "QA_question_SHA_disjoint_from_456_original_TRAIN":True,
        "never_opened_old_QA_TEST":True,
        "one_shot_no_optimization_on_QA":True,
        "baseline_fresh_QA_full_EOS":baseline_full,
        "candidate_fresh_QA_full_EOS":candidate_full,
        "baseline_fresh_QA_first":old_first,
        "candidate_fresh_QA_first":new_first,
        "fresh_QA_lost_full":lost,
        "fresh_QA_gained_full":gained,
        "per_fresh_mode":{k:dict(v) for k,v in stats.items()},
        "per_frozen_S95_route":{k:dict(v) for k,v in route_stats.items()},
        "counterfactual_pairs":{k:dict(v) for k,v in pairs.items()},
        "same_host_48_prompt_cpu_ratio":cpu_ratio,
        "inclusive_real_physical_bpp":bpp,
        "candidate_S115_sidecar_sha256":sha(a.S115_prefill_sidecar.read_bytes()),
        "QA_used_only_once_candidate_not_updated":True,
        "this_fresh_QA_is_now_INSPECTED_no_future_tuning_or_retests":True,
        "model_accepted":False,
        "LIVE_main_Render_Supabase_APK_untouched":True,
    }
    (a.out/"S118_ONCE_ONLY_NEW_QA_S101_S115_SUMMARY.json").write_text(
        json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({
        "event":"S118_ONCE_ONLY_NEW_QA_RESULT",
        "qa_count":128,"baseline_full":baseline_full,
        "candidate_full":candidate_full,
        "baseline_first":old_first,"candidate_first":new_first,
        "lost_full":lost,"gained_full":gained,
        "paired_CPU_ratio":cpu_ratio,"physical_bpp":bpp,
        "candidate_accepted":False,
    }),flush=True)


if __name__=="__main__":
    main()
