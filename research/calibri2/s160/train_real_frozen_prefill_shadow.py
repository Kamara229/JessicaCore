"""S160 real frozen S142/Green-Tunnel contextual feature probe for S159 slots.

Research-only, original 456 S87 and 128 S122 TRAIN. Pulls REAL frozen S142
physical FP16 sidecar and CB61, all S95 profiles, 3/9 GT, full GT and additive
core. One actual Apple MPS model PREFILL per TRAIN question; trains ONLY the
S159 low-rank shared-span slot head using these frozen contextual features.

Shadow head outputs are never presented as accepted CALIBRI responses. No QA,
no 43M gradients, no original decoding override, no LIVE services. Final
physically reloaded head is a REJECTED-UNTIL-J7 research artifact only.
"""
from __future__ import annotations
import copy
import json
import math
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
from calibri2.s8.broad_replay import MODEL_SPEC,SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s94.physical_profile_bank import freeze_report,sha
from calibri2.s95.train_and_audit_router import prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import apply_profile,load_frozen_numeric_weights
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import cohort
from calibri2.s122.independent_binding_train_curriculum import materialize_train_only
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import load_compact_q2
from calibri2.s127.q2_prefill_gate_control import prefill_only_tunnel
from calibri2.s132.progressive_core import ProgressiveRouteCore,IDS_KEY,WEIGHTS_KEY
from calibri2.s144.train_one_continued_j7 import frozen_routes
from calibri2.s158.targeted_retention_gpu import args_from_cli,check_sources
from calibri2.s159.parallel_slot_binding import ParallelSlotBindingHead,source_span_nll
from v176.operators import clear

SEED=1600160
SLOTS=2
RANK=16
BATCH=12
EPOCHS=20
LR=0.003
MAX_BPP=.7


def phrase_present(ids:torch.Tensor, pair:torch.Tensor)->bool:
    if len(pair)!=SLOTS:return False
    return any(torch.equal(ids[j:j+SLOTS],pair)
               for j in range(len(ids)-SLOTS+1))


def frozen_mps_rows(a,old,new,old_routes,new_routes,tok,side,q2,inherited,raw,bank):
    model,sources=create_small_paid_model(CB59(raw),build_config(MODEL_SPEC))
    model.to("mps").eval().requires_grad_(False)
    inner=IntegratedGreenTunnel(layers=(3,9),rank=8).to("mps").eval()
    inner.load_state_dict({k.removeprefix("internal_tunnel."):v for k,v in side.items()
                           if k.startswith("internal_tunnel.")},strict=True)
    inner.requires_grad_(False)
    if install_verified_fastpath(inner)!=("3","9"):
        raise AssertionError("S160 frozen 3/9 GT fastpath missing")
    inner.install(model)
    bridge=PromptConditionedDualTunnel(inner).to("mps").eval().install(model)
    bridge.coefficients.requires_grad_(False)
    full=FullLayerStateTunnel(width=320,depth=12,rank=4).to("mps").eval()
    load_compact_q2(full,q2)
    full.requires_grad_(False)
    full.install(model)
    expansion=ProgressiveRouteCore(rank=896).to("mps").eval().bind_original_route_bridge(bridge).install(model)
    expansion.load_active_fp16({IDS_KEY:inherited[IDS_KEY],WEIGHTS_KEY:inherited[WEIGHTS_KEY]})
    expansion.requires_grad_(False)
    from calibri2.s94.physical_profile_bank import CHANGED
    original={k:sources[k].centroids.detach().clone() for k in CHANGED}
    original_gt={k.removeprefix("internal_tunnel."):v for k,v in side.items()
                 if k.startswith("internal_tunnel.")}
    cache=[]
    details={"S87_456":{"total":len(old),"two_slot":0,"copyable":0},
             "S122_128":{"total":len(new),"two_slot":0,"copyable":0}}
    seen_route=None
    wall=[]
    try:
        for name,rows,routes in (("S87_456",old,old_routes),("S122_128",new,new_routes)):
            for k in sorted(rows):
                row=rows[k]
                gold=row["expected"]
                if len(gold)!=3 or gold[-1]!=tok.eos_token_id:
                    continue
                details[name]["two_slot"]+=1
                question=row["question"]
                prompt=tok.apply_chat_template([{"role":"user","content":question}],
                                               tokenize=False,add_generation_prompt=True)
                ids=torch.as_tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)
                expected=torch.as_tensor(gold[:2],dtype=torch.long)
                if not phrase_present(ids,expected):
                    continue
                details[name]["copyable"]+=1
                route=routes[k]
                if route!=seen_route:
                    apply_profile(sources,inner,route,bank,original,original_gt)
                    seen_route=route
                feat=(torch.as_tensor(prompt_only_vector(tok,question),dtype=torch.float32,
                                     device="mps") if route=="S93" else None)
                clear(sources)
                torch.mps.synchronize()
                t0=time.perf_counter()
                with torch.inference_mode():
                    with expansion.prefill_only(model),prefill_only_tunnel(model,full):
                        with bridge.prompt_scope(route,feat):
                            out=model(input_ids=ids.to("mps").unsqueeze(0),
                                      past_key_values=None,use_cache=False,
                                      output_hidden_states=True)
                torch.mps.synchronize()
                wall.append(time.perf_counter()-t0)
                hidden=out.hidden_states[-1][0].detach().cpu().float().contiguous()
                if hidden.shape!=(len(ids),320) or not bool(torch.isfinite(hidden).all()):
                    raise AssertionError("S160 invalid real S142 frozen 320D prefill")
                cache.append({"id":k,"origin":name,"question":question,
                              "expected":expected,"ids":ids,"states":hidden})
                del out
            print(json.dumps({"event":"S160_REAL_FROZEN_S142_PREFILL_COHORT",
                              "cohort":name,**details[name],"cached":len(cache)},ensure_ascii=False),flush=True)
        return cache,details,{"calls":len(wall),"median_MPS_prefill_seconds":statistics.median(wall)}
    finally:
        expansion.uninstall();full.uninstall();bridge.uninstall();inner.uninstall()


def collate(rows):
    maxlen=max(len(x["ids"]) for x in rows)
    batch=len(rows)
    states=torch.zeros((batch,maxlen,320),dtype=torch.float32)
    ids=torch.zeros((batch,maxlen),dtype=torch.long)
    mask=torch.zeros((batch,maxlen),dtype=torch.bool)
    gold=torch.empty((batch,SLOTS),dtype=torch.long)
    for i,row in enumerate(rows):
        n=len(row["ids"])
        states[i,:n]=row["states"]
        ids[i,:n]=row["ids"]
        mask[i,:n]=True
        gold[i]=row["expected"]
    return states,ids,mask,gold


@torch.no_grad()
def shadow_score(head,cache):
    head.eval()
    total={"S87_456":{"correct":0,"eligible":0},
           "S122_128":{"correct":0,"eligible":0}}
    found={}
    for i in range(0,len(cache),BATCH):
        rows=cache[i:i+BATCH]
        h,ids,mask,gold=collate(rows)
        out=head(h,ids,mask)
        predicted=out["slot_ids"]
        correct=predicted.eq(gold).all(dim=1)
        for j,row in enumerate(rows):
            total[row["origin"]]["eligible"]+=1
            total[row["origin"]]["correct"]+=int(correct[j])
            found[row["id"]]={"correct":bool(correct[j]),
                              "tokens":predicted[j].tolist(),
                              "confidence":float(out["confidence"][j]),
                              "margin":float(out["margin"][j])}
    pairs={}
    for k,rec in found.items():
        if k.startswith("S122/"):
            pair=k.rsplit("/",1)[0]
            pairs.setdefault(pair,{})[k.rsplit("/",1)[1]]=rec["correct"]
    pair_both=sum(v.get("left",False) and v.get("right",False) for v in pairs.values())
    return total,int(pair_both),found


def report_samples(cache,found,tok):
    # Original TRAIN case samples ONLY, deterministic IDs (no QA). Keep both
    # hits and misses so downstream researchers see actual generated strings.
    outcomes=[]
    for want in (True,False):
        matches=[row for row in cache if found[row["id"]]["correct"]==want]
        for row in matches[:10]:
            rec=found[row["id"]]
            outcomes.append({
                "cohort":row["origin"],"id":row["id"],"TRAIN_question":row["question"],
                "expected_two_words":tok.decode(row["expected"].tolist(),skip_special_tokens=True).strip(),
                "actual_shadow_two_words":tok.decode(rec["tokens"],skip_special_tokens=True).strip(),
                "exact":want,"confidence":rec["confidence"],
                "NOT_inherited_S142_autoregressive_answer":True,
            })
    return outcomes


def main():
    a=args_from_cli()
    a.out.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S160 real frozen feature extraction requires actual Apple MPS")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    guard,side,q2,inherited=check_sources(a)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new_rows,digest,_=materialize_train_only(tok)
    p122=json.loads(a.S122_curriculum_report.read_text())
    if digest!=p122["new_TRAIN_question_hashset_sha256"]:
        raise AssertionError("S160 S122 original TRAIN hash drift")
    new={row["id"]:row for row in new_rows}
    _,_,_,bank,router,_=load_frozen_numeric_weights(a)
    old,old_ids,old_routes=cohort(
        tok,{x:freeze_report(getattr(a,x+"_report")) for x in ("S87","S91","S93")},router)
    new_routes=frozen_routes(tok,router,new)
    if len(old)!=456 or len(new)!=128 or guard["old_456_full"]!=362 or guard["S122_128_full"]!=17:
        raise AssertionError("S160 original S142 baseline incorrect")
    cache,coverage,prefill=frozen_mps_rows(
        a,old,new,old_routes,new_routes,tok,side,q2,inherited,
        s87.open_envelope(a.S93_CB61.read_bytes()),bank)
    if coverage["S122_128"]["copyable"]<100 or coverage["S87_456"]["copyable"]<200:
        raise AssertionError("S160 shared-span approach insufficient original TRAIN source coverage")
    head=ParallelSlotBindingHead(hidden_size=320,slots=SLOTS,rank=RANK).cpu()
    before,pair_before,_=shadow_score(head,cache)
    opt=torch.optim.AdamW(head.parameters(),lr=LR,weight_decay=.002)
    initial=copy.deepcopy(head.state_dict())
    initial_adam=copy.deepcopy(opt.state_dict())
    history=[]
    # Stable original TRAIN-only ordering. All core/GT features are frozen and
    # detached. Training 30K-dimensional head on CPU avoids MPS tiny kernels.
    for epoch in range(EPOCHS):
        head.train()
        generator=torch.Generator(device="cpu").manual_seed(SEED+epoch)
        order=torch.randperm(len(cache),generator=generator).tolist()
        losses=[]
        for start in range(0,len(order),BATCH):
            selected=[cache[i] for i in order[start:start+BATCH]]
            h,ids,mask,gold=collate(selected)
            opt.zero_grad(set_to_none=True)
            out=head(h,ids,mask)
            loss=source_span_nll(out,ids,gold,mask)
            if not bool(torch.isfinite(loss).item()):
                raise AssertionError("S160 nonfinite genuine frozen S142-feature slot NLL")
            loss.backward()
            if not bool(torch.isfinite(head.keys.weight.grad).all()) or (
                not float(head.keys.weight.grad.abs().sum())>0):
                raise AssertionError("S160 missing real pretrained-feature slot gradients")
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.0)
            opt.step()
            losses.append(float(loss.detach()))
        row={"epoch":epoch+1,"TRAIN_nll":sum(losses)/len(losses)}
        if epoch in (0,4,9,19):
            score,pairs,_=shadow_score(head,cache)
            row["shadow_old"]=score["S87_456"]["correct"]
            row["shadow_new"]=score["S122_128"]["correct"]
            row["shadow_new_pairs_both"]=pairs
        history.append(row)
        print(json.dumps({"event":"S160_REAL_FROZEN_PREFILL_PARALLEL_SLOTS_EPOCH",**row}),flush=True)
    path=a.out/"S160_UNACCEPTED_PARALLEL_SLOTS_FP16.safetensors"
    save_file({k:v.detach().half().cpu().contiguous()
               for k,v in head.state_dict().items()},str(path))
    serialized=load_file(str(path))
    head_reloaded=ParallelSlotBindingHead(320,SLOTS,RANK).eval()
    head_reloaded.load_state_dict({k:v.float() for k,v in serialized.items()},strict=True)
    score,pair_after,found=shadow_score(head_reloaded,cache)
    examples=report_samples(cache,found,tok)
    sample=a.out/"S160_SHADOW_TRAIN_RESPONSES.json"
    sample.write_text(json.dumps(examples,indent=2,ensure_ascii=False))
    manifest=a.out/"S160_PARALLEL_SLOTS_RUNTIME.json"
    manifest.write_text(json.dumps({
        "stage":"S160_S142_REAL_FROZEN_PREFILL_SHADOW_SLOT_HEAD",
        "enabled_in_live":False,"enabled_in_S142_inference":False,
        "TRAIN_only_no_QA":True,"eos_after_two_slots":True,
        "source_span_binding_not_independent_slot_argmax":True,
        "source_S142_sha256":sha(a.S142_parent_sidecar.read_bytes()),
    },sort_keys=True))
    additional=sum(p.numel() for p in head_reloaded.parameters())
    if additional!=30721 or any(p.requires_grad for p in []):
        raise AssertionError("S160 parameter count mismatch")
    inherited_bpp=guard["physical_bpp"]
    old_arch=SOURCE_PARAMS+FullLayerStateTunnel(width=320,depth=12,rank=4).new_parameter_count()+ProgressiveRouteCore(rank=896).architectural_new_parameter_count()
    all_in_bpp=(inherited_bpp*old_arch+8*(path.stat().st_size+manifest.stat().st_size))/(old_arch+additional)
    report={
        "stage":"S160_REAL_S142_FROZEN_MPS_PREFILL_PARALLEL_SLOT_CPU_TRAIN",
        "source_S142_physical_checkpoint_sha256":sha(a.S142_parent_sidecar.read_bytes()),
        "baseline_S142_full_456":362,"baseline_S142_full_S122_128":17,
        "original_core_frozen":True,"original_green_tunnel_frozen":True,
        "training_only_authorized_S87_456_S122_128":True,
        "pretrained_real_MPS_frozen_prefill":prefill,
        "two_slot_source_coverage":coverage,
        "shadow_random_initial":before,
        "shadow_random_initial_new_pairs_both":pair_before,
        "shadow_trained_physically_reloaded":score,
        "shadow_new_pairs_both":pair_after,
        "epoch_history":history,
        "physically_reloaded_head":True,"head_parameter_count":additional,
        "new_FP16_bytes":path.stat().st_size,"new_manifest_bytes":manifest.stat().st_size,
        "all_in_physical_bpp_estimate":all_in_bpp,
        "under_0_7_bpp":bool(all_in_bpp<=MAX_BPP),
        "J7_full_autoregressive_456_128_not_yet_executed":True,
        "paired_CPU_full_generation_not_yet_measured":True,
        "original_S142_decoder_enabled":True,
        "new_head_decoder_enabled":False,
        "all_weights_and_adam_reverted_without_approval":True,
        "model_accepted":False,
        "QA_never_run":True,
        "no_MAIN_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    # Research probe cannot accept a head, irrespective of shadow scores.
    head.load_state_dict(initial,strict=True)
    opt.load_state_dict(initial_adam)
    if any(not torch.equal(v,head.state_dict()[k]) for k,v in initial.items()):
        raise AssertionError("S160 unaccepted head weights not rolled back")
    out=a.out/"S160_REAL_S142_PARALLEL_SLOT_TRAIN_SHADOW_AUDIT.json"
    out.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
    print(json.dumps({
        "event":"S160_REAL_FROZEN_S142_PARALLEL_SLOT_SHADOW_RESULT",
        "copyable_old":coverage["S87_456"]["copyable"],
        "copyable_new":coverage["S122_128"]["copyable"],
        "trained_shadow_old":score["S87_456"]["correct"],
        "trained_shadow_new":score["S122_128"]["correct"],
        "trained_shadow_new_pairs_both":pair_after,
        "physical_bpp":all_in_bpp,"model_accepted":False,
        "full_J7_pending":True,
    }),flush=True)


if __name__=="__main__":
    main()
