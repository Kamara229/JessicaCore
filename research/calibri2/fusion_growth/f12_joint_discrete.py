"""F12: extend F04 discrete optimization to the real integrated mixed-rate model.

Matched CE+contrast+region training with/without discrete code swaps, fixed
96 TRAIN rows per arm from F11's authored curriculum. Start from F06, not
uncompleted F10/F11. All source codebooks receive gradients; only the selected
operator's already-paid indices may change. No reference teacher.
"""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f03_rate_train_core import decode,encode,k_for
from research.calibri2.fusion_growth.f06_integrated_recovery import build,forward,evaluate,OLD_PREFIX
from research.calibri2.fusion_growth.f07_own_error_curriculum import preference_loss
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,open_bundle,HDR
from research.calibri2.fusion_growth.f11_broad_recovery import corpus,pack,FAMILIES
from v176.operators import clear

PARENT_SHA="7dcc391ce6001f88f6e7c12498c4f371b83431b049e4db2cb65d7f0b89ab722a"
RATE="K16_embed_K8_body"
STEPS=96
SEED=20261201
ARMS=("CENTROIDS","CENTROIDS_AND_CODES")

def propose(source,gradient,k):
    g=gradient.reshape(-1,8).astype(np.float32)
    old=source.codes.detach().cpu().numpy().copy()
    scales=source.base_scales.detach().cpu().numpy()
    assert len(g)==len(old) and np.all(old<k)
    assert np.isfinite(g).all() and np.all(scales>0)
    norms=np.einsum("ij,ij->i",g,g)
    # Bounded candidate search; same local trust penalty as F04.
    n=min(4096,len(g))
    idx=np.argsort(-norms,kind="stable")[:n]
    g=g[idx]*scales[idx//(source.shape[1]//8),None]
    centers=source.centroids.detach().cpu().numpy()[:k].astype(np.float16).astype(np.float32)
    delta=centers[None,:,:]-centers[old[idx],None,:]
    score=np.einsum("ij,ikj->ik",g,delta)+.05*np.linalg.norm(g,axis=1)[:,None]*np.linalg.norm(delta,axis=2)
    best=score.argmin(1);gain=-score[np.arange(n),best]
    valid=np.flatnonzero((gain>0)&(best!=old[idx]))
    sel=valid[np.argsort(-gain[valid],kind="stable")[:256]]
    new=old.copy();new[idx[sel]]=best[sel]
    assert np.all(new<k)
    return old,new,len(sel)

def objective(z,row,i):
    t=row["g"][i]
    l=(.25 if t==2 else 1.)*F.cross_entropy(z,torch.tensor([t]))/max(1,len(row["g"])-1)
    if i==row["div"]:
        l=l+.5*preference_loss(z,t,row["bad"][i])
    if t!=2:
        l=l+.15*preference_loss(z,t,2)
    return l

def train_row(state,row,target):
    grad=None;total=0.
    source=state[1][target] if target else None
    for i,t in enumerate(row["g"]):
        observed={}
        def capture(_m,_args,v):
            if id(v) not in observed:
                v.retain_grad();observed[id(v)]=v
        h=source.register_forward_hook(capture) if source is not None and i==row["div"] else None
        try:
            z=forward(state,row["p"]+row["g"][:i],row["features"])
            l=objective(z,row,i)
            assert torch.isfinite(l)
            l.backward();total+=float(l.detach())
            if h:
                gradients=[v.grad.detach().cpu().numpy().copy() for v in observed.values() if v.grad is not None]
                assert gradients,"missing actual operator gradient"
                grad=sum(gradients)
        finally:
            if h:h.remove()
        observed.clear()
    return total,grad

def train_score(state,row):
    total=0.
    for i in range(len(row["g"])):
        z=forward(state,row["p"]+row["g"][:i],row["features"]).detach()
        total+=float(objective(z,row,i))
    return total

def subset(rows,n):
    return [r for k in FAMILIES for r in [x for x in rows if x["kind"]==k][:n]]

def run(f01,f06,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    parent=(f06/"F06_JOINT.bundle").read_bytes();assert digest(parent)==PARENT_SHA
    archive,side=decode(parent)
    original=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(original[k])==v
    assert all(side[k]==original[k] for k in NAMES[1:3])
    legacy=(f08/"legacy_extras.bundle").read_bytes();assert len(legacy)==137712
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    tr,dv=subset(corpus("TRAIN"),12),subset(corpus("DEV"),6)
    assert len(tr)==STEPS and len(dv)==48
    assert not set(x["q"] for x in tr)&set(x["q"] for x in dv)
    train,dev=pack(tok,tr),pack(tok,dv)
    order=np.random.default_rng(SEED).permutation(STEPS)
    out.mkdir(parents=True,exist_ok=False)
    results={}
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/NAMES[0]).write_bytes(original[NAMES[0]])
        for k,v in side.items():(root/k).write_bytes(v)
        state=build(root,archive);before=evaluate(state,dev,tok);del state
        print("F12_BASELINE",json.dumps(before),flush=True)
        for arm in ARMS:
            for k,v in side.items():(root/k).write_bytes(v)
            state=build(root,archive)
            model,sources,bridge,full,expansion,child,region=state
            keys=sorted(sources)
            initial={k:s.codes.detach().numpy().copy() for k,s in sources.items()}
            core=[s.centroids for s in sources.values()]
            opt=torch.optim.Adam([{"params":core,"lr":.00015},{"params":region,"lr":.0003},
                                  {"params":[full.gates],"lr":.0001}])
            # 17 is coprime to 85? No: gcd=17. Use 19 to visit all 85
            # operators before repeating, including the tied embedding.
            seen=set();gates=set();regions=set();attempted=accepted=0;targets=set();history=[]
            for i,j in enumerate(order):
                row=train[int(j)]
                # First 85 steps cover every paid operator, then 11 revisit.
                target=keys[(i*19)%len(keys)] if arm==ARMS[1] else None
                opt.zero_grad(set_to_none=True)
                loss,gradient=train_row(state,row,target)
                for k,s in sources.items():
                    if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(k)
                if full.gates.grad is not None:gates.update(k for k,x in enumerate(full.gates.grad) if float(x.abs())>0)
                for k,p in child.named_parameters():
                    if p.grad is not None and float(p.grad.abs().sum())>0:regions.add(k)
                torch.nn.utils.clip_grad_norm_(core+region+[full.gates],1.,error_if_nonfinite=True)
                opt.step()
                with torch.no_grad():
                    for k,s in sources.items():s.centroids[k_for(RATE,k):].zero_()
                swaps=0
                if target:
                    targets.add(target)
                    s=sources[target]
                    old,new,n=propose(s,gradient,k_for(RATE,target));attempted+=n
                    if n:
                        pre=train_score(state,row)
                        with torch.no_grad():s.codes.copy_(torch.from_numpy(new))
                        post=train_score(state,row)
                        if np.isfinite(post) and post<pre-1e-7:
                            accepted+=n;swaps=n
                        else:
                            with torch.no_grad():s.codes.copy_(torch.from_numpy(old))
                        clear(sources)
                history.append({"step":i+1,"loss":loss,"target":target,"accepted":swaps})
                if (i+1)%24==0:print("F12_PROGRESS",arm,i+1,loss,attempted,accepted,flush=True)
            assert len(seen)==85 and len(gates)==12 and regions
            if arm==ARMS[1]:assert len(targets)==85
            final_changes=sum(int(np.count_nonzero(s.codes.numpy()!=initial[k])) for k,s in sources.items())
            if arm==ARMS[0]:assert final_changes==0
            new_side=dict(side)
            new_side[NAMES[3]]=save({OLD_PREFIX+k:v.detach().cpu().half().contiguous() for k,v in child.state_dict().items()})
            gate_key=next(iter(load_file(str(root/NAMES[4]))))
            new_side[NAMES[4]]=save({gate_key:full.gates.detach().cpu().half().contiguous()})
            fp=encode(archive.blob,sources,RATE,new_side);restored,rs=decode(fp);assert rs==new_side
            compact,head,_=split_head(new_side,8)
            cp=encode(archive.blob,sources,RATE,compact)
            qp=HDR.pack(b"FG09",len(cp),len(head),len(legacy))+cp+head+legacy
            qcore,qside,qlegacy=open_bundle(qp)
            assert qcore.blob==restored.blob and qlegacy==legacy
            (out/(arm+"_FP16.bundle")).write_bytes(fp);(out/(arm+"_Q8.bundle")).write_bytes(qp)
            del opt,state,model,sources,bridge,full,expansion,child,region,core
            for k,v in rs.items():(root/k).write_bytes(v)
            state=build(root,restored);fscore=evaluate(state,dev,tok);del state
            for k,v in qside.items():(root/k).write_bytes(v)
            state=build(root,qcore);qscore=evaluate(state,dev,tok);del state
            results[arm]={"FP16_plus_legacy":metrics(len(fp)+len(legacy)),
                          "Q8_plus_legacy":metrics(len(qp)),"FP16_sha256":digest(fp),"Q8_sha256":digest(qp),
                          "after_FP16":fscore,"after_Q8":qscore,"optimizer_steps":STEPS,
                          "connected_core_codebooks":len(seen),"connected_GT_gates":sorted(gates),
                          "connected_region_tensors":sorted(regions),
                          "operators_with_proposals":len(targets),"proposed_swaps":attempted,
                          "accepted_swaps":accepted,"final_changed_indices":final_changes,
                          "training_history":history}
            print("F12_ARM",arm,json.dumps({k:v for k,v in results[arm].items() if k!="training_history"},sort_keys=True),flush=True)
    report={"stage":"F12_INTEGRATED_DISCRETE_CODES","parent_sha256":PARENT_SHA,"depth":12,
            "before":before,"arms":results,"train_rows":len(tr),"dev_rows":len(dv),
            "data_sha256":digest(json.dumps([tr,dv],ensure_ascii=False).encode()),
            "FP32_teacher_calls":0,"candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Matched objectives and data, only discrete index optimization differs; extra candidate scoring compute explicitly not matched.",
                           "CPU integrated experiment, not an integrated GPU speed claim. Shared F11 authored diagnostic subset, not independent benchmark.",
                           "All known legacy bytes included but legacy decoder not active; no J7, optimized CPU, tokenizer or deployment certification."]}
    (out/"F12_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F12_ASSESS",json.dumps({**report,"arms":{a:{k:v for k,v in z.items() if k!="training_history"} for a,z in results.items()}},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("f01","f06","f08","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f06,a.f08,a.out)
