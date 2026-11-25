"""F34: reuse existing physical two-slot parallel binding head on F10.
Compare early pre-injection versus final causal states, matched head-only
training on fresh source-span pairs. One actual prefill for both slots.
No extra head in a persisted model, no protected data, no routing change.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from safetensors.torch import load,save
from calibri2.s159.parallel_slot_binding import ParallelSlotBindingHead
from calibri2.s160.real_parallel_slots_j7 import normalized_candidate_spans,joint_span_nll,MIN_CONFIDENCE,MIN_MARGIN
from research.calibri2.fusion_growth.f01_physical_budget import digest,metrics
from research.calibri2.fusion_growth.f09_head_rate_codec import HDR,open_bundle
from research.calibri2.fusion_growth.f13_profile_rebase import extra_open,extra_pack
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f06_integrated_recovery import forward
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from v176.operators import clear

PARENT="8138878ca404f82e833113b30cbc78337732e48b05b80c248d482c4a2e05583f"
PARALLEL="S160_REAL_PARALLEL_SLOT_FP16.safetensors"
SEED=20261218
STEPS=64
MODES=("EARLY","FINAL")

def corpus(tok,split):
    tr=split=="TRAIN";n=24 if tr else 12
    # Fixed independent authored lexical pools. Eligibility is tokenizer
    # feasibility only, never model outputs or score.
    colors=("red","blue","green","black") if tr else ("white","yellow","pink","brown")
    nouns=("book","key","box","cup","ball","door","ring","hat") if tr else ("chair","lamp","clock","bag","pen","shoe","plate","bell")
    names=("Ада","Клим") if tr else ("Лея","Ганс")
    labels=[f"{c} {w}" for c in colors for w in nouns if len(tok.encode(f"{c} {w}",add_special_tokens=False))==2]
    assert len(labels)>=8
    rows=[];seen=set()
    for i,a in enumerate(labels):
        for j,b in enumerate(labels):
            if i==j or a.split()[0]==b.split()[0] or a.split()[1]==b.split()[1]:continue
            # Each unordered assignment is used only once per split.
            if (a,b) in seen:continue
            pair=[]
            for who,y in zip(names,(a,b)):
                q=(f"В журнале меток: {names[0]} = {a}; {names[1]} = {b}. Какая метка у {who}? Перепиши оба слова."
                   if tr else f"Архив содержит записи ({names[0]}: {a}) и ({names[1]}: {b}). Укажи двухсловный код владельца {who}.")
                p,g=native(tok,q,y)
                if len(g)!=3 or g[-1]!=2:break
                canon=normalized_candidate_spans(tok,p)
                if not all(bool((canon==torch.tensor(tok.encode(x,add_special_tokens=False))).all(-1).any()) for x in (a,b)):break
                if not bool((canon==torch.tensor(g[:2])).all(-1).any()):break
                pair.append({"q":q,"p":p,"g":g,"split":split,"pair":len(rows)//2,
                             "features":prompt_features(tok,q),"canon":canon})
            if len(pair)!=2:continue
            seen.add((a,b));rows.extend(pair)
            if len(rows)==2*n:
                assert len({r["q"] for r in rows})==2*n
                return rows
    raise AssertionError("insufficient fresh tokenizer-compatible source pairs")

def collect(state,row):
    model=state[0];cap={};hooks=[]
    def early(_m,args,out):
        h=out[0] if isinstance(out,tuple) else out
        cap["EARLY"]=h.detach().cpu().float().clone()
    def final(_m,args):
        cap["FINAL"]=args[0].detach().cpu().float().clone()
    try:
        hooks.append(model.model.layers[5].register_forward_hook(early))
        hooks.append(model.lm_head.register_forward_pre_hook(final))
        z=forward(state,row["p"],row["features"]).detach()
    finally:
        for h in hooks:h.remove()
        clear(state[1])
    assert set(cap)==set(MODES) and all(v.shape==(1,len(row["p"]),320) and torch.isfinite(v).all() for v in cap.values())
    assert z.shape==(1,49152)
    return {**row,"states":cap,"native_first":int(z.argmax())}

def new_head(raw):
    h=ParallelSlotBindingHead(320,slots=2,rank=16)
    h.load_state_dict({k:v.float() for k,v in load(raw).items()},strict=True)
    return h

def score(head,row,mode):
    ids=torch.tensor([row["p"]],dtype=torch.long)
    raw=head(row["states"][mode],ids)["span_logits"]
    valid=row["canon"].ge(0).all(-1)
    assert int(valid.sum())>=2
    return raw.masked_fill(~valid[None],-torch.inf)

def audit(head,rows,mode):
    records=[]
    with torch.no_grad():
        for r in rows:
            z=score(head,r,mode);top=z[0].topk(2)
            chosen=r["canon"][int(top.indices[0])].tolist()+[2]
            confidence=float(z.softmax(-1).max());margin=float(top.values[0]-top.values[1])
            admitted=confidence>=MIN_CONFIDENCE and margin>=MIN_MARGIN
            records.append({"pair":r["pair"],"q":r["q"],"gold":r["g"],"generated":chosen,
                            "correct":chosen==r["g"],"confidence":confidence,"margin":margin,"admitted":admitted})
    return {"n":len(rows),"raw_exact":sum(r["correct"] for r in records),
            "both_pair_correct":sum(records[i]["correct"] and records[i+1]["correct"] for i in range(0,len(records),2)),
            "admitted":sum(r["admitted"] for r in records),
            "admitted_correct":sum(r["admitted"] and r["correct"] for r in records),
            "admitted_wrong":sum(r["admitted"] and not r["correct"] for r in records),
            "records":records}

def native_audit(state,rows):
    exact=0;eos=0;hist=Counter();calls=0;records=[]
    for r in rows:
        ids=[r["native_first"]];calls+=1
        for _ in range(7):
            if ids[-1]==2:break
            ids.append(int(forward(state,r["p"]+ids,r["features"]).detach().argmax()));calls+=1
        exact+=int(ids==r["g"]);eos+=int(ids[-1]==2);hist[ids[0]]+=1
        records.append({"q":r["q"],"gold":r["g"],"generated":ids})
    return {"n":len(rows),"exact":exact,"EOS":eos,"first_token_histogram":dict(hist),
            "core_calls_including_reused_prefill":calls,"records":records}

def brief(x):return {k:v for k,v in x.items() if k!="records"}

def run(f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(SEED)
    packet=(f10/"F10_Q8.bundle").read_bytes();assert digest(packet)==PARENT
    archive,side,legacy=open_bundle(packet)
    assert legacy==(f08/"legacy_extras.bundle").read_bytes()
    items,blobs=extra_open(legacy);original=blobs[PARALLEL]
    magic,nc,nh,nl=HDR.unpack_from(packet);assert magic==b"FG09" and len(packet)==HDR.size+nc+nh+nl
    fixed=packet[HDR.size:HDR.size+nc+nh]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train=corpus(tok,"TRAIN");dev=corpus(tok,"DEV")
    assert len(train)==48 and len(dev)==24 and not {r["q"] for r in train}&{r["q"] for r in dev}
    out.mkdir(parents=True,exist_ok=False)
    state=direct(archive,side)
    tr=[collect(state,r) for r in train];dv=[collect(state,r) for r in dev]
    native_result=native_audit(state,dv);del state
    print("F34_NATIVE",json.dumps(brief(native_result),sort_keys=True),flush=True)
    results={}
    for mode in MODES:
        head=new_head(original)
        before={"TRAIN":audit(head,tr,mode),"DEV":audit(head,dv,mode)}
        opt=torch.optim.Adam(head.parameters(),lr=.002)
        rng=np.random.default_rng(SEED);order=[int(i) for _ in range(3) for i in rng.permutation(len(tr)//2)]
        curve=[]
        for step,pair in enumerate(order[:STEPS]):
            opt.zero_grad(set_to_none=True)
            losses=[]
            for r in tr[2*pair:2*pair+2]:
                assert r["split"]=="TRAIN"
                z=score(head,r,mode)
                losses.append(joint_span_nll(z,r["canon"][None],torch.tensor([r["g"][:2]])))
            loss=torch.stack(losses).mean();assert torch.isfinite(loss)
            loss.backward()
            assert head.query.weight.grad is not None and torch.isfinite(head.query.weight.grad).all()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
            curve.append(float(loss.detach()))
        assert len(curve)==STEPS
        raw=save({k:v.detach().cpu().half().contiguous() for k,v in head.state_dict().items()})
        assert set(load(raw))==set(load(original))
        assert all(v.shape==load(original)[k].shape for k,v in load(raw).items())
        new_blobs=dict(blobs);new_blobs[PARALLEL]=raw
        updated=extra_pack(items,new_blobs)
        new_packet=HDR.pack(b"FG09",nc,nh,len(updated))+fixed+updated
        a,s,l=open_bundle(new_packet);assert a.blob==archive.blob and s==side
        assert extra_open(l)[1]==new_blobs
        assert all(new_blobs[k]==v for k,v in blobs.items() if k!=PARALLEL)
        (out/(mode+"_Q8.bundle")).write_bytes(new_packet)
        loaded=new_head(extra_open(l)[1][PARALLEL])
        after={"TRAIN":audit(loaded,tr,mode),"DEV":audit(loaded,dv,mode)}
        results[mode]={"before":before,"after":after,"steps":STEPS,
                       "head_parameters":sum(p.numel() for p in loaded.parameters()),
                       "known_payload":metrics(len(new_packet)),"sha256":digest(new_packet),
                       "first16_loss":float(np.mean(curve[:16])),"last16_loss":float(np.mean(curve[-16:])),
                       "core_prefill_per_parallel_request":1}
        print("F34_ARM",mode,json.dumps({**results[mode],
              **{k:{sp:brief(v) for sp,v in results[mode][k].items()} for k in ("before","after")}},sort_keys=True),flush=True)
    report={"stage":"F34_EXISTING_PARALLEL_HEAD_PREFILL","parent_sha256":PARENT,
            "native_DEV":native_result,"arms":results,"train_n":len(tr),"dev_n":len(dv),
            "new_parameters":0,"core_optimizer_steps":0,"FP32_teacher_calls":0,
            "candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Source-bound exact two-token spans only; EOS appended by fixed historical slot protocol, not learned general termination.",
                           "Raw parallel accuracy separate from original fixed confidence/margin admission; no old selector chain or fallback changed.",
                           "Early states captured before child injection; final states after all layers. Core frozen; no proof of improved core learning.",
                           "Synthetic new owner-binding prompts with English two-word tags, not math or universal generation.",
                           "Cached state audit after physical head reload; no wall-clock speed claim, no 616 replay, tokenizer/runtime separate."]}
    (out/"F34_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("F34_ASSESS",json.dumps({"stage":report["stage"],"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f10,a.f08,a.out)
