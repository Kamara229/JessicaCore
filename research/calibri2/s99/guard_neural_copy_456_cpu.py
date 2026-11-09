"""S99: physically verify an S98 tiny neural noun-copy scorer IN the S95 model.

RESEARCH ONLY. Uses immutable training artifacts of S94/S95/S98. Rebuilds
the actual compressed 36M S93 CPU model from authenticated physical CB61,
then switches numeric S87/S91/S93 codebooks and Green Tunnel states using
the frozen S95 numeric prompt-only router. Integrates the numeric S98 MPS-
trained tiny neural source-span selector ONLY for the second generated token.

For all 456 frozen TRAIN questions, verify the exact S95 original policy
first by independently archived per-identity CPU outcomes, then measure
actual S99 autoregressive tokens including EOS. NO S96 previously inspected
new prompts, sealed QA/TEST, new training, teacher labels at inference,
prompt->answer table, or LIVE changes.

S99 MUST not be accepted without zero losses on 282 older union + all prior
first tokens, new genuinely full/EOS successes, CPU>=90% same-run original
policy, <=0.7 bpp all-inclusive, and later independent untouched QA.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from torch.nn import functional as F

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s94.physical_profile_bank import (
    CHANGED, PROFILES, sha, per_identity_union, freeze_report)
from calibri2.s95.train_and_audit_router import (
    CHOICES, PromptOnlyRouter, prompt_only_vector, score_picks,
    read_verified_bank)
from calibri2.s98.train_owner_anchored_copy import TinyCopyScorer, candidates
from calibri2.s8.broad_replay import SOURCE_PARAMS
from v176.operators import clear

CONFIDENCE_THRESHOLD=0.95
MIN_SPEED_RATIO=0.90
MAX_BPP=0.7
SAMPLE_SPEED_ROWS=48


def full_cpu_with_optional_neural_copy(
        model,sources,first,later,head,tok,question,pointer,copy_net):
    """The same physical S82 greedy decode with a learned second-token
    candidate override. No gold labels or historical answer lookup here.
    """
    prompt=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    context=tok.encode(prompt,add_special_tokens=False)
    ids=torch.tensor([context],dtype=torch.long)
    kept=s82.candidate_positions(tok,context) if pointer is not None else []
    token_ids=torch.tensor([context[i] for i in kept],dtype=torch.long)
    generated=[]
    eligible=False
    applied=False
    for step in range(s82.s56.MAX_NEW):
        with torch.inference_mode():
            clear(sources)
            out=model(input_ids=ids,use_cache=False,output_hidden_states=True)
            hidden=out.hidden_states[-1][0]
            h=hidden[-1]
            logits=out.logits[0,-1]+F.linear(
                first(h) if step==0 else later(h),head)
            if step==1 and pointer is not None and kept:
                if int(logits.argmax())!=tok.eos_token_id:
                    keys=hidden[torch.tensor(kept,dtype=torch.long)]
                    bonus=pointer.full_vocab_bonus(h,keys,token_ids)
                    logits=logits+torch.zeros_like(logits).index_add(
                        0,token_ids,bonus)
            chosen=int(logits.argmax())
            if step==1 and copy_net is not None and generated:
                pair=candidates(tok,question,generated[0])
                if len(pair)==2 and pair[0][0]!=pair[1][0]:
                    eligible=True
                    vectors=torch.tensor(np.stack([x[1] for x in pair]),
                                           dtype=torch.float32)
                    scores=copy_net(vectors)
                    probs=torch.softmax(scores,dim=0)
                    winner=int(scores.argmax())
                    if float(probs[winner])>=CONFIDENCE_THRESHOLD:
                        suggested=int(pair[winner][0])
                        if suggested!=chosen and suggested!=tok.eos_token_id:
                            chosen=suggested
                            applied=True
            generated.append(chosen)
            del out
            clear(sources)
            if chosen==tok.eos_token_id:
                break
            ids=torch.cat([ids,torch.tensor([[chosen]],dtype=torch.long)],
                          dim=1)
    return generated,{"candidate_present":eligible,
                      "neural_override_applied":applied}


def load_frozen_numeric_weights(a):
    p94=json.loads(a.S94_report.read_text())
    p95=json.loads(a.S95_report.read_text())
    p98=json.loads(a.S98_report.read_text())
    if (p94.get("stage")!="S94_COMPACT_EXACT_THREE_PROFILE_FEASIBILITY_TRAIN_ONLY" or
        p95.get("stage")!="S95_PROMPT_ONLY_NEURAL_THREE_PROFILE_ROUTER_TRAIN_RESEARCH" or
        p98.get("stage")!="S98_GENERIC_NEURAL_OWNER_ANCHORED_NOUN_COPY_TRAIN_ONLY" or
        p95.get("candidate_accepted") is not False or
        p98.get("candidate_accepted") is not False or
        not p95.get("all_456_EOS_output_exact_routed") or
        p95.get("frozen_three_profile_union_oracle_only")!=287 or
        not p98.get("physical_bpp_guard_passed") or
        not p94.get("all_inclusive_bpp_at_most_0_7")):
        raise AssertionError("S99 requires genuine frozen S94/S95/S98 evidence")
    if (sha(a.S94_bank.read_bytes())!=p94["physical_bank_sha256"] or
        sha(a.S95_router.read_bytes())!=p95["router_sha256"] or
        sha(a.S93_CB61.read_bytes())!=p94["source_SHA256"]["S93"] or
        sha(a.S93_sidecar.read_bytes())!=p94["sidecar_SHA256"]["S93"]):
        raise AssertionError("Frozen physical source, bank or first router changed")
    bank=read_verified_bank(a.S94_bank,p94["physical_bank_sha256"])
    net=PromptOnlyRouter()
    net.load_state_dict(load_file(str(a.S95_router)),strict=True)
    net.eval().requires_grad_(False)
    scorer=TinyCopyScorer()
    scorer.load_state_dict(load_file(str(a.S98_copy_net)),strict=True)
    scorer.eval().requires_grad_(False)
    if not torch.all(torch.isfinite(t) for t in scorer.parameters()):
        raise AssertionError("Nonfinite trained S98 neural copy scorer")
    return p94,p95,p98,bank,net,scorer


def apply_profile(sources,inner,profile,bank,original,original_tunnel):
    with torch.no_grad():
        for key in CHANGED:
            value=original[key] if profile=="S93" else bank[profile][key]
            sources[key].centroids.copy_(value)
        tunnel=original_tunnel if profile=="S93" else {
            k.removeprefix("internal_tunnel."):v
            for k,v in bank[profile].items()
            if k.startswith("internal_tunnel.")}
        inner.load_state_dict(tunnel,strict=True)
    clear(sources)


def main():
    ap=argparse.ArgumentParser()
    for name in (
        "S98_copy_net","S98_report",
        "S95_router","S95_report",
        "S94_bank","S94_report",
        "S93_CB61","S93_sidecar",
        "S87_report","S91_report","S93_report"):
        ap.add_argument(name,type=Path)
    ap.add_argument("out",type=Path)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    p94,p95,p98,bank,router,copy_net=load_frozen_numeric_weights(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    full={"S87":freeze_report(a.S87_report),
          "S91":freeze_report(a.S91_report),
          "S93":freeze_report(a.S93_report)}
    union,_,_=per_identity_union(full)
    rows={}
    for family,pairs in s87.families(tok).items():
        for pair in pairs:
            for side in ("left","right"):
                key=f'{family}/{pair["pair_id"]}/{side}'
                rows[key]=pair[side]["question"]
    ids=sorted(rows)
    if len(ids)!=456 or set(ids)!=set(full["S87"]):
        raise AssertionError("Immutable 456 TRAIN cohort changed")
    for ident in ids:
        if sha(rows[ident].encode())!=full["S87"][ident]["question_sha256"]:
            raise AssertionError("Frozen TRAIN prompt drift")
    x=torch.tensor(np.stack([prompt_only_vector(tok,rows[i]) for i in ids]),
                   dtype=torch.float32)
    with torch.inference_mode():
        choices=router(x).argmax(dim=1).tolist()
    initial=score_picks(full,ids,choices)
    if (initial["correct_full_plus_EOS"]!=287 or
        initial["lost_previously_full_count"]!=0 or
        initial["lost_previously_first_count"]!=0):
        raise AssertionError("Physically reloaded S95 initial routing changed")
    selected={ident:CHOICES[int(c)] for ident,c in zip(ids,choices)}
    original_side=load_file(str(a.S93_sidecar))
    model,sources,first,later,pointer,head,inner=s87.build_cpu(
        s87.open_envelope(a.S93_CB61.read_bytes()),original_side,True)
    if install_verified_fastpath(inner)!=("3","9"):
        raise AssertionError("Frozen internal GT exact CPU fastpath absent")
    original={k:sources[k].centroids.detach().clone() for k in CHANGED}
    oldtunnel={k.removeprefix("internal_tunnel."):v
            for k,v in original_side.items() if k.startswith("internal_tunnel.")}
    correct=0
    first_ok=0
    lost_legacy_full=[]
    lost_legacy_first=[]
    lost_S95=[]
    gained_S95=[]
    gained_beyond_legacy=[]
    eligible=0
    overrides=0
    profile_switches=0
    sample_speeds={"S95":[],"S99":[]}
    by_family={}
    previous=None
    audited=0
    try:
        for index,ident in enumerate(ids):
            profile=selected[ident]
            if previous!=profile:
                apply_profile(sources,inner,profile,bank,original,oldtunnel)
                previous=profile
                profile_switches+=1
            old=full[profile][ident]["S87_ids"]
            gold=full["S87"][ident]["target_ids"]
            if index<SAMPLE_SPEED_ROWS:
                start=time.perf_counter()
                cold=s82.full_cpu_generate(model,sources,first,later,
                                            head,tok,rows[ident],pointer)
                sample_speeds["S95"].append(time.perf_counter()-start)
                if cold!=old:
                    raise AssertionError("Same-run frozen S95 physical CPU parity lost: "+ident)
            start=time.perf_counter()
            changed,notes=full_cpu_with_optional_neural_copy(
                model,sources,first,later,head,tok,rows[ident],pointer,copy_net)
            if index<SAMPLE_SPEED_ROWS:
                sample_speeds["S99"].append(time.perf_counter()-start)
            good=changed==gold
            prior_full=any(full[p][ident]["S87_ids"]==gold for p in ("S87","S91"))
            prior_first=any(bool(full[p][ident]["S87_ids"]) and
                      full[p][ident]["S87_ids"][0]==gold[0]
                      for p in ("S87","S91"))
            first_correct=bool(changed) and changed[0]==gold[0]
            correct+=int(good)
            first_ok+=int(first_correct)
            eligible+=int(notes["candidate_present"])
            overrides+=int(notes["neural_override_applied"])
            audited+=1
            family=ident.split("/")[0]
            v=by_family.setdefault(family,{"count":0,"full":0,
                 "old_S95_full":0,"applied":0,"eligible":0})
            v["count"]+=1
            v["full"]+=int(good)
            v["old_S95_full"]+=int(old==gold)
            v["applied"]+=int(notes["neural_override_applied"])
            v["eligible"]+=int(notes["candidate_present"])
            if prior_full and not good:
                lost_legacy_full.append(ident)
            if prior_first and not first_correct:
                lost_legacy_first.append(ident)
            if old==gold and not good:
                lost_S95.append(ident)
            if old!=gold and good:
                gained_S95.append(ident)
            if not prior_full and good:
                gained_beyond_legacy.append(ident)
    finally:
        inner.uninstall()
        del model
    if audited!=456 or len(sample_speeds["S95"])!=48:
        raise AssertionError("Physical 456 inference incomplete")
    old_median=statistics.median(sample_speeds["S95"])
    new_median=statistics.median(sample_speeds["S99"])
    ratio=old_median/new_median
    # Include compact manifests/overheads even when some are diagnostics
    # rather than shipped; this is a conservative charged upper bound.
    manifest={
        "stage":"S99_PHYSICAL_S95_PLUS_S98_COPY_FULL456_TRAIN_GUARD",
        "TRAIN_456_actual_physically_generated":True,
        "immutable_S95_source_sha256":sha(a.S93_CB61.read_bytes()),
        "S94_weightbank_sha256":sha(a.S94_bank.read_bytes()),
        "S95_neural_profile_selector_sha256":sha(a.S95_router.read_bytes()),
        "S98_neural_copy_selector_sha256":sha(a.S98_copy_net.read_bytes()),
        "original_S95_correct_full_EOS":287,
        "S99_new_correct_full_EOS":correct,
        "S99_first_correct":first_ok,
        "S99_new_full_minus_S95":correct-287,
        "S99_legacy_union_282_lost_full_count":len(lost_legacy_full),
        "S99_legacy_union_lost_first_count":len(lost_legacy_first),
        "S99_S95_previously_correct_287_lost":len(lost_S95),
        "S99_new_full_gains_relative_to_S95":len(gained_S95),
        "S99_new_full_beyond_S87_S91_union":len(gained_beyond_legacy),
        "neural_source_candidates_available":eligible,
        "neural_second_token_override_applied":overrides,
        "frozen_S95_profile_switches":profile_switches,
        "per_TRAIN_family":by_family,
        "same_run_median_CPU_S95_over_S99_ratio_48":ratio,
        "sample_48_S95_vs_S99_CPU_speed_guard_90pct":ratio>=MIN_SPEED_RATIO,
        "no_legacy_282_full_or_first_regression":
               not lost_legacy_full and not lost_legacy_first,
        "no_S95_287_full_regression":not lost_S95,
        "new_full_EOS_gain_over_S95":len(gained_S95)>0,
        "old_train_full_loss_ids":lost_legacy_full,
        "old_train_first_loss_ids":lost_legacy_first,
        "old_S95_full_loss_ids":lost_S95,
        "gained_S95_full_ids":gained_S95,
        "S96_once_inspected_novel_QA_never_read_or_trained":True,
        "all_previously_inspected_QA_TEST_embargoed":True,
        "no_new_MPS_training_S98_weights_are_physically_frozen":True,
        "candidate_accepted":False,"LIVE_untouched":True,
        "native_C_and_GPU_inference_parity_pending":True,
        "independent_novel_QA_pending":True,
    }
    report=a.out/"S99_ALL456_NEW_NEURAL_COPY_PHYSICAL_TRAIN_AUDIT.json"
    for _ in range(40):
        raw=json.dumps(manifest,sort_keys=True,indent=2).encode()
        total=p98["conservative_all_inclusive_physical_bytes"]+len(raw)
        nextv={"S99_report_bytes_charged":len(raw),
               "conservative_entire_physical_bytes":total,
               "conservative_entire_physical_bpp":round(8*total/SOURCE_PARAMS,12),
               "below_0_7_bpp":8*total/SOURCE_PARAMS<=MAX_BPP}
        if all(manifest.get(k)==v for k,v in nextv.items()):
            break
        manifest.update(nextv)
    report.write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print(json.dumps({"event":"S99_PHYSICAL_456_TRAIN_NEURAL_COPY_END",
        "full_EOS_correct":correct,
        "frozen_S95_previous_full_EOS":287,
        "S95_skills_forgotten":len(lost_S95),
        "legacy_282_full_lost":len(lost_legacy_full),
        "legacy_first_lost":len(lost_legacy_first),
        "net_new_full_vs_S95":correct-287,
        "S98_copy_opportunities":eligible,
        "S98_copy_applied":overrides,
        "physical_bpp":manifest["conservative_entire_physical_bpp"],
        "paired_speed_ratio_48":ratio,
        "candidate_accepted":False}),flush=True)
    passed=(manifest["no_legacy_282_full_or_first_regression"] and
            manifest["no_S95_287_full_regression"] and
            manifest["new_full_EOS_gain_over_S95"] and
            manifest["sample_48_S95_vs_S99_CPU_speed_guard_90pct"] and
            manifest["below_0_7_bpp"])
    if not passed:
        raise SystemExit("REJECTED S99: one or more hard TRAIN retention, "
                         "new full/EOS, CPU speed or physical size guards failed")
    # Even if TRAIN passes, this is NOT a validated fresh QA or native-C model.


if __name__=="__main__":
    main()
