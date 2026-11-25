"""F04: matched real MPS core training, codebook-only vs trainable discrete codes.
An independent K16 experiment while F03 runs; not a promoted F03 child.
All four existing sidecars are charged and preserved, not attached.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,unpack,metrics,digest
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f03_rate_train_core import make,reduced,encode,decode,examples,evaluate
from direct_codebook_v59 import CB59,unpack5
from v176.operators import clear

SEED=20261126
STEPS=96
K=16
ARMS=("CENTROIDS","CENTROIDS_AND_CODES")

def corpus():
    train=[]
    for i in range(48):
        a,b=13+i*2,3+i%9
        train.extend([(f"Add {a} and {b}. Give only the result.",str(a+b)),
                      (f"Найди разность чисел {a} и {b}. Только число.",str(a-b)),
                      (f"What is {i%8+2} times {i%7+3}?",str((i%8+2)*(i%7+3))),
                      (f"Повтори точно: число {a}.",f"число {a}")])
    dev=[]
    for i in range(8):
        a,b=211+7*i,21+i
        dev.extend([(f"Find the sum of {a} and {b}.",str(a+b)),
                    (f"Вычти {b} из {a}.",str(a-b)),
                    (f"Calculate the product of {11+i} and 4.",str((11+i)*4)),
                    (f"Скопируй: число {a}.",f"число {a}")])
    assert len(train)==192 and len(dev)==32
    # Some simple multiplication prompts repeat; repeated exposures are
    # intentional, not counted as unique prompts.
    assert not set(p for p,a in train)&set(p for p,a in dev)
    return train,dev

def ce(model,sources,row,device):
    prefix,answer,_=row
    ids=torch.tensor([prefix+answer],dtype=torch.long,device=device)
    clear(sources)
    logits=model(input_ids=ids,use_cache=False).logits[0,len(prefix)-1:-1]
    assert logits.shape[0]==len(answer)
    return F.cross_entropy(logits.float(),torch.tensor(answer,device=device))

def candidate(src,grad):
    # Use real task gradient of the reconstructed matrix, not teacher weights.
    # Only legal K16 indices; select groups with highest actual gradient norm.
    g=grad.reshape(-1,8).astype(np.float32)
    old=src.codes.detach().cpu().numpy().copy()
    scale=src.base_scales.detach().cpu().numpy()
    groups=src.shape[1]//8
    assert len(g)==len(old)
    norms=np.einsum("ij,ij->i",g,g)
    n=min(4096,len(g))
    idx=np.argsort(-norms,kind="stable")[:n]
    g=g[idx]*scale[idx//groups,None]
    c=src.centroids.detach().cpu().numpy()[:K].astype(np.float16).astype(np.float32)
    delta=c[None,:,:]-c[old[idx],None,:]
    norm=np.linalg.norm(g,axis=1)
    score=np.einsum("ij,ikj->ik",g,delta)
    # Mild local trust penalty: no original FP32 reference is used.
    score+=.05*norm[:,None]*np.linalg.norm(delta,axis=2)
    best=score.argmin(axis=1)
    gain=-score[np.arange(n),best]
    valid=np.flatnonzero((gain>0)&(best!=old[idx]))
    ranked=valid[np.argsort(-gain[valid],kind="stable")[:256]]
    new=old.copy();new[idx[ranked]]=best[ranked]
    return old,new,len(ranked)

def init(archive):
    model,sources=make(archive)
    with torch.no_grad():
        for s in sources.values():
            c,i=reduced(s.centroids.detach().numpy(),s.codes.numpy(),K)
            s.centroids.copy_(torch.from_numpy(c));s.codes.copy_(torch.from_numpy(i))
    return model,sources

def run(root,out):
    assert torch.backends.mps.is_available() and torch.backends.mps.is_built()
    import os
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.manual_seed(SEED);torch.set_num_threads(2)
    files=unpack((root/"F01.xz.bundle").read_bytes())
    assert set(files)==set(NAMES)
    for k,v in PIN.items():assert digest(files[k])==v
    raw=raw_core(files[NAMES[0]]);archive=CB59(raw)
    side={k:files[k] for k in NAMES[1:]}
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    tr,dv=corpus();train=examples(tok,tr);dev=examples(tok,dv)
    order=np.random.default_rng(SEED).permutation(len(train))
    out.mkdir(parents=True,exist_ok=False)
    device=torch.device("mps")
    results={}
    for arm in ARMS:
        model,sources=init(archive)
        keys=sorted(sources)
        initial={k:s.codes.detach().numpy().copy() for k,s in sources.items()}
        initial_sha=digest(b"".join(initial[k].tobytes()+sources[k].centroids.detach().numpy().tobytes() for k in keys))
        if results:assert initial_sha==results[ARMS[0]]["initial_state_sha256"]
        model.to(device)
        assert all(s.centroids.device.type=="mps" and s.codes.device.type=="mps" for s in sources.values())
        opt=torch.optim.Adam([s.centroids for s in sources.values()],lr=.0008)
        seen=set();curve=[];attempted=accepted=0
        torch.mps.synchronize();start=time.perf_counter()
        for step in range(STEPS):
            rows=[train[int(order[step*2+j])] for j in range(2)]
            # Rotate proposals across embedding and all 12 layers. Every
            # fourth update, same sample for gradient and acceptance.
            target=keys[((step//4)*17)%len(keys)]
            use_codes=arm==ARMS[1] and (step+1)%4==0
            captured=[]
            def hook(_m,_a,v):
                v.retain_grad();captured.append(v)
            handle=sources[target].register_forward_hook(hook) if use_codes else None
            opt.zero_grad(set_to_none=True)
            try:
                l=ce(model,sources,rows[0],device)
                (l/2).backward()
                if use_codes:
                    assert captured and captured[0].grad is not None
                    grad=captured[0].grad.detach().cpu().numpy().copy()*2
            finally:
                if handle:handle.remove()
            captured.clear();clear(sources)
            l2=ce(model,sources,rows[1],device)
            (l2/2).backward();clear(sources)
            for k,s in sources.items():
                g=s.centroids.grad
                assert g is not None and g.device.type=="mps"
            # Batch check gradient presence without 85 device synchronizations.
            norms=torch.stack([s.centroids.grad.abs().sum() for s in sources.values()]).detach().cpu().numpy()
            assert np.isfinite(norms).all()
            seen.update(k for k,n in zip(sources,norms) if n>0)
            torch.nn.utils.clip_grad_norm_([s.centroids for s in sources.values()],1.)
            opt.step()
            with torch.no_grad():
                for s in sources.values():s.centroids[K:].zero_()
                if use_codes:
                    s=sources[target]
                    old,new,n=candidate(s,grad);attempted+=n
                    if n:
                        before=float(ce(model,sources,rows[0],device))
                        s.codes.copy_(torch.as_tensor(new,device=device,dtype=torch.long))
                        after=float(ce(model,sources,rows[0],device))
                        if np.isfinite(after) and after<before-1e-7:accepted+=n
                        else:s.codes.copy_(torch.as_tensor(old,device=device,dtype=torch.long))
                    clear(sources)
            curve.append((float(l.detach())+float(l2.detach()))/2)
            if (step+1)%24==0:
                print("F04_PROGRESS",arm,step+1,json.dumps({"loss":curve[-1],"attempted":attempted,"accepted":accepted}),flush=True)
        torch.mps.synchronize();elapsed=time.perf_counter()-start
        assert len(seen)==85
        clear(sources);del opt
        model.to("cpu")
        final_changes=sum(int(np.count_nonzero(s.codes.numpy()!=initial[k])) for k,s in sources.items())
        if arm==ARMS[0]:assert final_changes==0
        packet=encode(raw,sources,"K16_all",side)
        restored,restored_side=decode(packet)
        assert restored_side==side
        for k,s in sources.items():
            h=restored.handles[k]
            assert np.array_equal(unpack5(h.packed.tobytes(),s.codes.numel()),s.codes.numpy())
            assert np.array_equal(h.centroids,s.centroids.detach().numpy().astype(np.float16).astype(np.float32))
        (out/(arm+".bundle")).write_bytes(packet)
        del model,sources
        torch.mps.empty_cache()
        model,sources=make(restored)
        score=evaluate(model,sources,dev,tok)
        del model,sources
        results[arm]={**metrics(len(packet)),"initial_state_sha256":initial_sha,
                      "checkpoint_sha256":digest(packet),"MPS_seconds":elapsed,
                      "optimizer_steps":STEPS,"gradient_connected_codebooks":len(seen),
                      "train_exposures":len(train),"unique_train_prompts":len(set(p for p,a in tr)),
                      "proposed_code_swaps":attempted,"accepted_code_swaps":accepted,
                      "final_changed_code_indices":final_changes,
                      "mean_first24_train_NLL":sum(curve[:24])/24,
                      "mean_last24_train_NLL":sum(curve[-24:])/24,
                      "cpu_physical_reload_DEV":score}
        print("F04_ARM",arm,json.dumps(results[arm],sort_keys=True),flush=True)
    report={"stage":"F04_MPS_DISCRETE_CORE_TRAIN","device":"mps","CPU_fallback":False,
            "depth":12,"arms":results,"FP32_teacher_calls":0,"output_gate":0,
            "candidate_accepted":False,"all_in_certified":False,
            "train_dev_sha256":digest(json.dumps([tr,dv],ensure_ascii=False).encode()),
            "limitations":["MPS backward/Adam for all 85 codebooks; discrete proposal selection on CPU explicitly, not hidden fallback.",
                           "Same K16 initial core from F01, independent of unfinished F03; four charged sidecars not active.",
                           "Internal 32 arithmetic/copy examples, not independent full-system quality, J7, or CPU speed acceptance.",
                           "Archive restores K32 ABI; runtime compact kernel not implemented."]}
    (out/"F04_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F04_ASSESS",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
