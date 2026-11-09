"""S95: learn a small, prompt-only three-profile selector and audit physical TRAIN.

The S94 weight-only bank contains NO question hashes, QA answers or labels.
This study uses frozen 456 TRAIN outcomes to supervise a compact feature-hash
neural selector, and independently checks exact switched physical CPU output.
The oracle union 287/456 is a ceiling, not an achieved result. The full
TRAIN audit and 4-fold TRAIN holdout remain distinct from sealed QA.
No LIVE changes. Never mark a candidate accepted without no-forgetting,
new full/EOS improvement, a proper speed audit and new independent QA.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import statistics
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s94.physical_profile_bank import (
    sha,freeze_report,per_identity_union,BANK_MAGIC,PROFILES,CHANGED)
from calibri2.s8.broad_replay import SOURCE_PARAMS
from v176.operators import clear

SEED=950095
DIM=512
HIDDEN=12
MAX_BPP=0.7
CHOICES=("S93","S91","S87")


def prompt_only_vector(tok,question):
    """Stable signed hash of token unigram, local bigram and character trigrams.

    NO gold answer, recorded greedy token or per-TRAIN case identity enters
    this runtime function. It works on entirely unseen prompt strings.
    """
    tokens=tok.encode(question,add_special_tokens=False)
    feats=[]
    for i,t in enumerate(tokens):
        feats.append("u:"+str(t))
        feats.append("slot:"+str(i//6)+":"+str(t))
        if i:
            feats.append("b:"+str(tokens[i-1])+":"+str(t))
    # Character ngrams give a little context-independent semantic transfer.
    normalized=" ".join(question.lower().split())
    feats.extend("c:"+normalized[i:i+3] for i in range(max(0,len(normalized)-2)))
    out=np.zeros(DIM,dtype=np.float32)
    for feat in feats:
        key=zlib.crc32(feat.encode("utf8")) & 0xffffffff
        slot=key%DIM
        sign=-1.0 if key & 0x80000000 else 1.0
        out[slot]+=sign
    norm=np.linalg.norm(out)
    if norm>0:out/=norm
    return out


class PromptOnlyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(DIM,HIDDEN),nn.Tanh(),
                               nn.Linear(HIDDEN,len(CHOICES)))
    def forward(self,x):
        return self.net(x)


def model_fit(x,y,weights,seed,epochs=1200,weight_decay=0.0002):
    torch.manual_seed(seed)
    model=PromptOnlyRouter()
    optim=torch.optim.AdamW(model.parameters(),lr=0.018,weight_decay=weight_decay)
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        loss=(F.cross_entropy(model(x),y,reduction="none")*weights).mean()
        loss.backward()
        optim.step()
    # The real physical router is stored as FP16, so train-only selection
    # must use the identical rounded weights, never the transient FP32.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.half().float())
    return model.eval()


def preferred_labels(reports,ids):
    """TRAIN-only supervision from independently generated frozen outcomes."""
    targets=[]
    weighting=[]
    for ident in ids:
        goal=reports["S87"][ident]["target_ids"]
        full={p:reports[p][ident]["S87_ids"]==goal for p in PROFILES}
        first={p:bool(reports[p][ident]["S87_ids"]) and
               reports[p][ident]["S87_ids"][0]==goal[0] for p in PROFILES}
        chosen=next((p for p in CHOICES if full[p]),None)
        if chosen is None:
            chosen=next((p for p in CHOICES if first[p]),"S93")
        targets.append(CHOICES.index(chosen))
        # Non-S93 profiles are necessary for strictly protected old skills.
        # Weight their rare corrections without hiding base profile skills.
        weighting.append(10.0 if chosen!="S93" and (full[chosen] or first[chosen])
                         else 4.0 if full["S93"] else 1.0)
    return torch.tensor(targets,dtype=torch.long),torch.tensor(weighting,dtype=torch.float32)


def score_picks(reports,ids,pred):
    by_id={ident:CHOICES[int(label)] for ident,label in zip(ids,pred)}
    full=0
    first=0
    lost_full=[]
    lost_first=[]
    gained=[]
    confusion={p:0 for p in PROFILES}
    for ident in ids:
        p=by_id[ident]
        confusion[p]+=1
        gold=reports["S87"][ident]["target_ids"]
        original={k:reports[k][ident]["S87_ids"] for k in PROFILES}
        ans=original[p]
        complete=(ans==gold)
        first_ok=bool(ans) and ans[0]==gold[0]
        full+=int(complete)
        first+=int(first_ok)
        prior_full=any(original[k]==gold for k in ("S87","S91"))
        prior_first=any(original[k] and original[k][0]==gold[0]
                        for k in ("S87","S91"))
        if prior_full and not complete:lost_full.append(ident)
        if prior_first and not first_ok:lost_first.append(ident)
        if complete and not prior_full:gained.append(ident)
    return {"correct_full_plus_EOS":full,"correct_first":first,
            "lost_previously_full_count":len(lost_full),
            "lost_previously_first_count":len(lost_first),
            "new_full_beyond_S87_S91":len(gained),
            "choices":confusion,
            "all_legacy_full_retained":not lost_full,
            "all_legacy_first_retained":not lost_first,
            "lost_full_ids":lost_full,"lost_first_ids":lost_first,
            "new_full_ids":gained}


def read_verified_bank(path,expected_sha):
    binary=path.read_bytes()
    if sha(binary)!=expected_sha or not binary.startswith(BANK_MAGIC):
        raise AssertionError("S94 physical bank SHA or version mismatch")
    offset=len(BANK_MAGIC)
    header_size=struct.unpack_from("<I",binary,offset)[0]
    off=offset+4
    header=json.loads(binary[off:off+header_size])
    if header["stage"]!="S94_LOSSLESS_WEIGHT_ONLY_THREE_PROFILE_BANK" or (
        not header["has_no_questions_answers_or_routing_table"]):
        raise AssertionError("S94 physical bank provenance incorrect")
    raw=zlib.decompress(binary[off+header_size:])
    if sha(raw)!=header["uncompressed_sha256"] or len(raw)!=header["uncompressed_bytes"]:
        raise AssertionError("S94 packed alternate numeric weights failed roundtrip")
    result={"S87":{},"S91":{}}
    for item in header["tensors"]:
        blob=raw[item["offset"]:item["offset"]+item["size"]]
        if sha(blob)!=item["sha256"]:
            raise AssertionError("Profile tensor changed: "+item["name"])
        dtype={"float16":"<f2","float32":"<f4"}[item["dtype"]]
        arr=np.frombuffer(blob,dtype=dtype).copy().reshape(item["shape"])
        result[item["profile"]][item["name"]]=torch.from_numpy(arr).float()
    for p in ("S87","S91"):
        if not set(CHANGED).issubset(result[p]):
            raise AssertionError("S94 bank lost alternate core codebook")
    return result


def actual_physical_456_cpu(a,selection,ids,questions,reports,physical_weights):
    """Full exact generation; verify each profile against its own frozen CPU report.

    This is an independent new CPU reconstruction of the S94 *weight bank*.
    It does NOT use frozen report outputs as inference results.
    """
    raw=a.S93_CB61.read_bytes()
    side=load_file(str(a.S93_sidecar))
    from calibri2.s94.physical_profile_bank import sha as hash_raw
    if hash_raw(raw)!=a.S93_expected_source_sha:
        raise AssertionError("S93 original packed source hash drift")
    model,sources,first,later,pointer,head,inner=s87.build_cpu(
        s87.open_envelope(raw),side,True)
    assert install_verified_fastpath(inner)==("3","9")
    original_tunnel={k.removeprefix("internal_tunnel."):v for k,v in side.items()
                     if k.startswith("internal_tunnel.")}
    originals={k:sources[k].centroids.detach().clone() for k in CHANGED}
    active=None
    elapsed=[]
    outcomes={}
    switched=0
    try:
        for ident in ids:
            profile=selection[ident]
            if profile!=active:
                with torch.no_grad():
                    for k in CHANGED:
                        chosen=originals[k] if profile=="S93" else physical_weights[profile][k]
                        sources[k].centroids.copy_(chosen)
                    gt=(original_tunnel if profile=="S93" else
                        {k.removeprefix("internal_tunnel."):v for k,v in
                         physical_weights[profile].items()
                         if k.startswith("internal_tunnel.")})
                    inner.load_state_dict(gt,strict=True)
                clear(sources)
                active=profile
                switched+=1
            start=time.perf_counter()
            result=s82.full_cpu_generate(model,sources,first,later,head,
                                         a.tok,questions[ident],pointer)
            elapsed.append(time.perf_counter()-start)
            if result!=reports[profile][ident]["S87_ids"]:
                raise AssertionError("S95 SWITCHED REAL CPU OUTPUT differs from frozen independently"+
                                     " physically measured report: "+profile+"/"+ident)
            outcomes[ident]=result
    finally:
        inner.uninstall()
        del model
    return outcomes,{
       "all_456_physically_generated":len(outcomes)==456,
       "all_456_selected_exact_token_replay_verified":True,
       "profile_switch_count":switched,
       "median_CPU_seconds_per_prompt":statistics.median(elapsed),
       "total_CPU_generation_seconds":sum(elapsed),
       "native_C_and_MPS_parity_not_yet_done":True
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("S94_bank",type=Path)
    p.add_argument("S94_report",type=Path)
    p.add_argument("S93_CB61",type=Path)
    p.add_argument("S93_sidecar",type=Path)
    p.add_argument("S87_report",type=Path)
    p.add_argument("S91_report",type=Path)
    p.add_argument("S93_report",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    info=json.loads(a.S94_report.read_text())
    if info.get("stage")!="S94_COMPACT_EXACT_THREE_PROFILE_FEASIBILITY_TRAIN_ONLY" or (
        not info.get("all_inclusive_bpp_at_most_0_7")) or (
        not info.get("switched_exact_CPU_subset_parity")):
        raise AssertionError("S95 requires completed physical S94 cost and switched CPU proof")
    a.S93_expected_source_sha=info["source_SHA256"]["S93"]
    if sha(a.S93_CB61.read_bytes())!=a.S93_expected_source_sha or (
        sha(a.S93_sidecar.read_bytes())!=info["sidecar_SHA256"]["S93"]):
        raise AssertionError("Frozen true S93 source/GT bytes changed")
    physical=read_verified_bank(a.S94_bank,info["physical_bank_sha256"])
    reports={"S87":freeze_report(a.S87_report),
             "S91":freeze_report(a.S91_report),
             "S93":freeze_report(a.S93_report)}
    union,_,_=per_identity_union(reports)
    if union["full_EOS_oracle_upper_bound_three_profiles"]!=287:
        raise AssertionError("S95 changed frozen TRAIN target universe")
    a.tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=a.tok
    questions={}
    for family,pairs in s87.families(a.tok).items():
        for pair in pairs:
            for side in ("left","right"):
                ident=f'{family}/{pair["pair_id"]}/{side}'
                questions[ident]=pair[side]["question"]
    ids=sorted(questions)
    if len(ids)!=456 or set(ids)!=set(reports["S87"]):
        raise AssertionError("Actual independent 456 TRAIN input drift")
    for ident in ids:
        if sha(questions[ident].encode())!=reports["S87"][ident]["question_sha256"]:
            raise AssertionError("Unauthorized prompt drift: "+ident)
    x=torch.tensor(np.stack([prompt_only_vector(a.tok,questions[k])
                              for k in ids]),dtype=torch.float32)
    y,w=preferred_labels(reports,ids)
    runs=[]
    best=None
    best_score=None
    for seed in (SEED,SEED+1,SEED+2,SEED+3):
        model=model_fit(x,y,w,seed)
        with torch.no_grad():
            chosen=model(x).argmax(dim=1).tolist()
        audit=score_picks(reports,ids,chosen)
        # Reject forgetting strongly; never silently optimize only total score.
        key=(-audit["lost_previously_full_count"]-
                audit["lost_previously_first_count"],
             audit["correct_full_plus_EOS"],
             audit["new_full_beyond_S87_S91"])
        runs.append({"seed":seed,**{k:v for k,v in audit.items()
            if k not in ("lost_full_ids","lost_first_ids","new_full_ids")}})
        if best_score is None or key>best_score:
            best_score=key
            best=(seed,model,chosen,audit)
    seed,router,pred,train=best
    folds=[]
    for fold in range(4):
        hold=[i for i,k in enumerate(ids) if int(sha(k.encode())[:8],16)%4==fold]
        seen=[i for i in range(len(ids)) if i not in set(hold)]
        gate=model_fit(x[seen],y[seen],w[seen],seed+fold+100,epochs=1000)
        with torch.no_grad():
            selected=gate(x[hold]).argmax(dim=1).tolist()
        results=score_picks(reports,[ids[i] for i in hold],selected)
        folds.append({"fold":fold,"heldout_TRAIN_rows":len(hold),
                      "strict_full":results["correct_full_plus_EOS"],
                      "forgot_full":results["lost_previously_full_count"],
                      "forgot_first":results["lost_previously_first_count"],
                      "beyond_old_union":results["new_full_beyond_S87_S91"]})
    # Store only trainable numeric model weights; NO prompt/response IDs.
    router_path=a.out/"S95_PROMPT_ONLY_HASHED_NEURAL_ROUTER_FP16.safetensors"
    save_file({k:v.detach().cpu().half().contiguous() for k,v
               in router.state_dict().items()},str(router_path))
    normalized=PromptOnlyRouter()
    normalized.load_state_dict(load_file(str(router_path)),strict=True)
    normalized.eval()
    with torch.no_grad():
        redone=normalized(x).argmax(dim=1).tolist()
    if redone!=pred:
        raise AssertionError("Physical FP16 router changed chosen frozen outputs")
    result={"stage":"S95_PROMPT_ONLY_NEURAL_THREE_PROFILE_ROUTER_TRAIN_RESEARCH",
       "candidate_accepted":False,"never_promote_from_only_TRAIN":True,
       "TRAIN_only_no_previous_QA":True,"LIVE_untouched":True,
       "green_tunnel_physical_bank_sha256":sha(a.S94_bank.read_bytes()),
       "base_S93_source_sha256":a.S93_expected_source_sha,
       "router_sha256":sha(router_path.read_bytes()),
       "router_numeric_FP16_parameters_only":True,
       "router_numeric_physical_bytes":router_path.stat().st_size,
       "no_question_answer_or_identity_lookup_at_inference":True,
       "frozen_three_profile_union_oracle_only":287,
       "all_training_seeds":[{"seed":t["seed"],
          "strict_correct":t["correct_full_plus_EOS"],
          "lost_full":t["lost_previously_full_count"],
          "lost_first":t["lost_previously_first_count"]} for t in runs],
       "selected_router_seed":seed,
       "frozen_report_routed_TRAIN_prediction":train,
       "fourfold_HELD_OUT_TRAIN_not_QA":folds,
       "strict_physically_reloaded_456_cpu_audit_pending":True}
    report_path=a.out/"S95_PROMPT_ROUTER_STRICT_TRAIN_AUDIT.json"
    # Over-count the existing S94 manifest (safer than missing it).
    previous=info["physical_all_inclusive_bytes"]
    for _ in range(30):
        raw=json.dumps(result,indent=2,sort_keys=True).encode()
        total=previous+len(raw)+router_path.stat().st_size
        fields={"manifest_bytes":len(raw),
            "physical_all_inclusive_bytes_conservative":total,
            "physical_bpp_conservative":round(8*total/SOURCE_PARAMS,12),
            "physical_bpp_at_most_0_7":8*total/SOURCE_PARAMS<=MAX_BPP}
        if all(result.get(k)==v for k,v in fields.items()):break
        result.update(fields)
    report_path.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({"event":"S95_TRAIN_ONLY_NEURAL_ROUTER_LEARNED",
      "physical_bpp_conservative":result["physical_bpp_conservative"],
      "train_full_correct_from_frozen_sources":train["correct_full_plus_EOS"],
      "train_union_lost_full":train["lost_previously_full_count"],
      "train_union_lost_first":train["lost_previously_first_count"],
      "new_full_beyond_legacy_union":train["new_full_beyond_S87_S91"],
      "crossval_heldout_TRAIN":folds,
      "still_not_accepted":True}),flush=True)
    if not result["physical_bpp_at_most_0_7"]:
        raise SystemExit("S95 rejected: router exceeds 0.7 bpp all-inclusive")
    selection={ident:CHOICES[int(n)] for ident,n in zip(ids,pred)}
    generated,performance=actual_physical_456_cpu(
         a,selection,ids,questions,reports,physical)
    if len(generated)!=456 or any(generated[k]!=reports[selection[k]][k]["S87_ids"]
                                  for k in ids):
        raise AssertionError("Physical generation not identical to frozen selectors")
    result["strict_physically_reloaded_456_cpu_audit_pending"]=False
    result["strict_physically_reloaded_456_cpu"]=performance
    result["all_456_EOS_output_exact_routed"]=(len(generated)==456)
    result["hard_no_forgetting_full_AND_first_pass"]=(
       train["lost_previously_full_count"]==0 and
       train["lost_previously_first_count"]==0)
    result["more_new_full_than_S87_S91_union"]=(
       train["new_full_beyond_S87_S91"]>0 and
       train["correct_full_plus_EOS"]>282)
    result["strict_train_eligibility_only"]=(
       result["hard_no_forgetting_full_AND_first_pass"] and
       result["more_new_full_than_S87_S91_union"])
    # The full CPU audit report is NOT included in the runtime artifact's
    # physical bpp. Runtime packaging uses a compact numeric manifest.
    # Explicitly account for this expanded diagnostic manifest too: the
    # conservative estimate can only rise, never fall, after CPU audit.
    for _ in range(30):
        manifest=json.dumps(result,indent=2,sort_keys=True).encode()
        total=previous+router_path.stat().st_size+len(manifest)
        fields={"manifest_bytes":len(manifest),
          "physical_all_inclusive_bytes_conservative":total,
          "physical_bpp_conservative":round(8*total/SOURCE_PARAMS,12),
          "physical_bpp_at_most_0_7":8*total/SOURCE_PARAMS<=MAX_BPP}
        if all(result.get(k)==v for k,v in fields.items()):break
        result.update(fields)
    report_path.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({"event":"S95_PHYSICAL_ROUTED_ALL456_CPU_PROOF",
       "full_correct":train["correct_full_plus_EOS"],
       "forgot_full":train["lost_previously_full_count"],
       "forgot_first":train["lost_previously_first_count"],
       "beyond_union":train["new_full_beyond_S87_S91"],
       "physical_bpp":result["physical_bpp_conservative"],
       "all456_switch_replay":True,
       "TRAIN_eligibility_only":result["strict_train_eligibility_only"],
       "candidate_accepted":False}),flush=True)
    if not result["physical_bpp_at_most_0_7"]:
        raise SystemExit("REJECTED S95 physical router: storage budget")
    if not result["strict_train_eligibility_only"]:
        raise SystemExit("REJECTED S95: lost legacy TRAIN skill or failed new full gain")


if __name__=="__main__":
    main()
