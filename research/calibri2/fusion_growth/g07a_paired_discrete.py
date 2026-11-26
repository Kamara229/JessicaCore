"""G07A: matched TRAIN-only discrete acceptance experiment on G05 133.701M.
G04 accepted 5312 swaps on individual TRAIN rows but regressed both DEV
domains. Test paired counterfactual acceptance with the same proposal
gradient, matrix order, and number of forwards. No updates to centroids.
Independent from unfinished G06; pinned physical G05 checkpoint only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import extra_pack,digest
from research.calibri2.fusion_growth.g02_attention_discrete import first_loss
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore,rate
from research.calibri2.fusion_growth.g06_causal_gradient import PIN,dataset,pair_audit,compact
from v176.operators import clear

SEED=20261230
TRIALS=24
CAP=64

def pairs(rows):
    result={}
    for r in rows:
        assert r["split"]=="TRAIN"
        result.setdefault(r["pair"],[]).append(r)
    assert all(len(v)==2 and v[0]["answer"]!=v[1]["answer"] for v in result.values())
    return [v for v in result.values() if v[0]["g"][0]!=v[1]["g"][0]]

def gradient(st,s,row):
    seen=[]
    def capture(_m,_a,w):seen.append(w)
    handle=s.register_forward_hook(capture)
    try:
        loss=first_loss(st,row)
        assert len(seen)==1 and seen[0].requires_grad
        g,=torch.autograd.grad(loss,seen[0])
        assert tuple(g.shape)==tuple(s.shape) and torch.isfinite(g).all()
        return float(loss.detach()),g.detach().numpy().reshape(-1,8).copy()
    finally:
        handle.remove();s.clear();clear(st[1])

def eval_loss(st,row):
    return float(first_loss(st,row).detach())

def proposal(s,g):
    centers,sc=s.fields()
    c=centers[:8].detach().numpy()
    ids=s.codes.detach().numpy().copy()
    scale=np.repeat(sc.detach().numpy(),s.shape[1]//8)
    old=c[ids]
    delta=c[None,:,:]-old[:,None,:]
    g=g*scale[:,None];norm=np.linalg.norm(g,axis=1)
    cost=np.einsum("nki,ni->nk",delta,g)/np.maximum(norm[:,None],1e-20)
    cost+=.05*np.sum(delta**2,axis=-1)
    winner=cost.argmin(axis=1)
    gain=-cost[np.arange(len(ids)),winner]
    order=np.argsort(-gain,kind="stable")
    ix=order[(gain[order]>1e-6)&(norm[order]>1e-15)][:CAP]
    return torch.from_numpy(ix.copy()).long(),torch.from_numpy(ids[ix].copy()).long(),torch.from_numpy(winner[ix].copy()).long()

def trial(st,bank,k,pair,mode):
    s=bank[k]
    before,grads=zip(*(gradient(st,s,r) for r in pair))
    ix,old,new=proposal(s,.5*(grads[0]+grads[1]))
    with torch.no_grad():s.codes[ix]=new
    s.clear();clear(st[1])
    after=tuple(eval_loss(st,r) for r in pair)
    # Identical compute in both arms. Only acceptance decision differs.
    finite=all(np.isfinite(v) for v in after)
    if mode=="SINGLE_ACCEPT":
        keep=finite and after[0]+1e-7<before[0]
    else:
        keep=finite and sum(after)+2e-7<sum(before) and all(a<=b+.005 for a,b in zip(after,before))
    keep=bool(keep and len(ix)>0)
    if not keep:
        with torch.no_grad():s.codes[ix]=old
        s.clear();clear(st[1])
    return {"matrix":k,"family":pair[0]["family"],"proposed":len(ix),
            "accepted":len(ix) if keep else 0,"before":list(before),"after_trial":list(after),
            "both_nonworse":bool(all(a<=b+.005 for a,b in zip(after,before))),
            "joint_loss_change":float(sum(after)-sum(before))}

def run(g05,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tokenizer_bytes=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    tr,dv=dataset(tok,"TRAIN"),dataset(tok,"DEV")
    assert not {r["q"] for v in tr.values() for r in v}&{r["q"] for v in dv.values() for r in v}
    rng=np.random.default_rng(SEED)
    mm,gg=pairs(tr["math"]),pairs(tr["general"])
    assert len(mm)>=12 and len(gg)>=12
    order=[p for pair in zip([mm[int(i)] for i in rng.permutation(len(mm))[:12]],
                               [gg[int(i)] for i in rng.permutation(len(gg))[:12]]) for p in pair]
    keys=[f"model.layers.{i}.{op}.weight" for i in range(12) for op in ("self_attn.q_proj","mlp.down_proj")]
    assert len(keys)==len(order)==TRIALS
    out.mkdir(parents=True,exist_ok=False)
    def reload(b):return restore(b[:6],b[6],b[7])
    st,*rest=reload(blobs)
    before=pair_audit(st,dv,tok)
    del st,rest
    print("G07A_BASE",json.dumps(compact(before),sort_keys=True),flush=True)
    arms={}
    for mode in ("SINGLE_ACCEPT","PAIRED_ACCEPT"):
        st,*rest=reload(blobs);bank=rest[-2]
        assert len(bank)==84
        rows=[trial(st,bank,k,p,mode) for k,p in zip(keys,order)]
        new=extra_pack(bank)
        vals=(*blobs[:6],new,blobs[7])
        storage=rate(vals[:6],vals[6],vals[7],G05,tokenizer_bytes)
        assert storage["within_known_weight_cap"]
        for k,v in zip(PIN,vals):(out/(mode+"_"+k)).write_bytes(v)
        del st,rest,bank
        st,*rest=reload(vals)
        post=pair_audit(st,dv,tok);del st,rest
        info={"trials":len(rows),"proposed":sum(r["proposed"] for r in rows),
              "accepted":sum(r["accepted"] for r in rows),
              "accepted_matrices":sum(r["accepted"]>0 for r in rows),
              "accepted_with_pair_regression":sum(r["accepted"]>0 and not r["both_nonworse"] for r in rows),
              "train_joint_loss_change_accepted":sum(r["joint_loss_change"] for r in rows if r["accepted"]>0),
              "after_physical_reload":post,"storage":storage,"sixth_sha256":digest(new),
              "records":rows}
        arms[mode]=info
        print("G07A_ARM",mode,json.dumps({**{k:v for k,v in info.items() if k!="records"},
                                         "after_physical_reload":compact(post)},sort_keys=True),flush=True)
    report={"stage":"G07A_MATCHED_PAIRED_DISCRETE_ACCEPTANCE","geometry":G05,
            "baseline":before,"arms":arms,"matched_trials":TRIALS,
            "actual_gradient_forwards_per_arm":2*TRIALS,
            "actual_trial_loss_forwards_per_arm":2*TRIALS,
            "candidate_accepted":False,"FP32_teacher_calls":0,"output_gate":0,
            "limitations":["Pinned G05, not unfinished G06. Both arms use same paired gradient proposal, only acceptance rule differs.",
                           "Only 24 sixth-bank matrices updated, first-token TRAIN loss and fixed 0.005 tolerance; no full-model discrete convergence claim.",
                           "DEV short counterfactual prompts; pair acceptance may still overfit. No historical 616 replay, CPU or full-runtime certification."]}
    (out/"G07A_REPORT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("G07A_ASSESS",json.dumps({"stage":report["stage"],"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
