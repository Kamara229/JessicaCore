"""F36: true full-vocabulary two-future-token prediction from one causal prefill.
Existing source-span decoder untouched. Small experimental rank8 residual
for position t+2, shared original tied native head, no answer fed to slot2.
Matched ordinary full-answer CE vs CE plus auxiliary future-token CE.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load,save
from research.calibri2.fusion_growth.f35_joint_parallel_binding import data
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,digest,metrics
from research.calibri2.fusion_growth.f03_rate_train_core import decode,encode,k_for
from research.calibri2.fusion_growth.f06_integrated_recovery import forward,OLD_PREFIX,scoped_reduced_native_head_gain
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,open_bundle,HDR
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f20_state_diagnostic import F10_SHA
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from v176.operators import clear

SEED=20261220
RATE="K16_embed_K8_body"
ARMS=("AR_CE","AR_CE_PLUS_MTP")
STEPS=48

class FutureSlot(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Linear(320,8,bias=False)
        self.up=nn.Linear(8,320,bias=False)
        nn.init.normal_(self.down.weight,std=.02);nn.init.zeros_(self.up.weight)
    def forward(self,h):
        return h+self.up(F.gelu(self.down(F.layer_norm(h,(320,)))))

def arithmetic(tok,split,n):
    tr=split=="TRAIN"
    rng=np.random.default_rng(SEED+(0 if tr else 1))
    pairs=[(a,b,c,d) for a in range(1,5) for b in range(1,5)
           for c in range(1,5) for d in range(1,5) if a+b<=9 and c+d<=9]
    rows=[]
    for i in rng.permutation(len(pairs)):
        a,b,c,d=pairs[int(i)];y=f"{a+b}{c+d}"
        q=(f"Реши два примера: {a}+{b} и {c}+{d}. Запиши обе цифры подряд, без пробелов."
           if tr else f"Сначала сложи {a} и {b}, затем сложи {c} и {d}. Соедини два результата в одно число без пробела.")
        p,g=native(tok,q,y)
        if len(g)!=3 or g[-1]!=2 or tok.decode(g[:-1],skip_special_tokens=True).strip()!=y:continue
        # Exclude any answer sequence already present contiguously in prompt.
        if any(p[j:j+2]==g[:2] for j in range(len(p)-1)):continue
        rows.append({"q":q,"p":p,"g":g,"split":split,"kind":"computed",
                     "features":prompt_features(tok,q)})
        if len(rows)==n:break
    assert len(rows)==n and len({tuple(r["g"][:2]) for r in rows})>=8
    return rows

def dataset(tok):
    ds=data(tok);result={}
    for split,copy in (("TRAIN",ds["TRAIN"][:24]),("DEV",ds["DEV_LEXICAL"][:12])):
        result[split]=[{**r,"split":split,"kind":"copy"} for r in copy]+arithmetic(tok,split,len(copy))
    assert len(result["TRAIN"])==48 and len(result["DEV"])==24
    assert not {r["q"] for r in result["TRAIN"]}&{r["q"] for r in result["DEV"]}
    return result

def prefill(state,row,aux=None):
    if aux is None:return forward(state,row["p"],row["features"]),None
    model,sources,_,_,_,child,_=state;cap=[]
    h=model.lm_head.register_forward_pre_hook(lambda _m,args:cap.append(args[0][:,-1:,:].float()))
    try:z=forward(state,row["p"],row["features"])
    finally:h.remove()
    assert len(cap)==1 and cap[0].shape==(1,1,320) and cap[0].requires_grad
    assert model.lm_head.source is sources["model.embed_tokens.weight"]
    future=aux(cap[0])
    # Same physically tied full 49152 readout and existing learned correction.
    # Separate readout operation, but no second decoder/core execution.
    with scoped_reduced_native_head_gain(child.language_head):
        z2=(model.lm_head(future).float()+child.language_head(future).float())[:,0,:]
    clear(sources)
    assert z.shape==z2.shape==(1,49152) and torch.isfinite(z2).all()
    return z,z2

def audit(state,rows,aux=None):
    records=[]
    for r in rows:
        z,z2=prefill(state,r,aux);first=int(z.detach().argmax())
        second=int(z2.detach().argmax()) if z2 is not None else None
        # Standard greedy decode. Parallel arm separately tries the first two
        # predictions jointly, then resumes normal greedy without forced EOS.
        modes={"AR":[first]}
        if aux is not None:modes["MTP"]=[first] if first==2 else [first,second]
        res={}
        for mode,ids in modes.items():
            calls=1
            for _ in range(8-len(ids)):
                if ids[-1]==2:break
                v=forward(state,r["p"]+ids,r["features"]).detach()
                ids.append(int(v.argmax()));calls+=1
            res[mode]={"ids":ids,"exact":ids==r["g"],"EOS":ids[-1]==2,"core_calls":calls}
        records.append({"q":r["q"],"kind":r["kind"],"gold":r["g"],
                        "first":first,"future_second":second,"modes":res})
    result={"n":len(records),"first_correct":sum(r["first"]==r["gold"][0] for r in records),
            "future_second_correct":sum(r["future_second"]==r["gold"][1] for r in records) if aux is not None else None,
            "parallel_two_correct":sum([r["first"],r["future_second"]]==r["gold"][:2] for r in records) if aux is not None else None,
            "first_histogram":dict(Counter(r["first"] for r in records)),"records":records}
    for m in records[0]["modes"]:
        result[m]={"exact":sum(r["modes"][m]["exact"] for r in records),
                   "EOS":sum(r["modes"][m]["EOS"] for r in records),
                   "core_calls":sum(r["modes"][m]["core_calls"] for r in records),
                   "exact_by_kind":{k:sum(r["modes"][m]["exact"] and r["kind"]==k for r in records) for k in ("copy","computed")}}
    return result

def brief(x):return {k:v for k,v in x.items() if k!="records"}

def run(f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(SEED)
    raw=(f10/"F10_FP16.bundle").read_bytes();assert digest(raw)==F10_SHA
    archive,side=decode(raw)
    legacy=(f08/"legacy_extras.bundle").read_bytes();assert len(legacy)==137712
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    ds=dataset(tok);out.mkdir(parents=True,exist_ok=False)
    state=direct(archive,side)
    before={k:audit(state,rows) for k,rows in ds.items()};del state
    print("F36_BEFORE",json.dumps({k:brief(v) for k,v in before.items()},sort_keys=True),flush=True)
    order=np.random.default_rng(SEED).permutation(STEPS);results={}
    for arm in ARMS:
        state=direct(archive,side);_,sources,_,full,_,child,region=state
        torch.manual_seed(SEED)
        aux=FutureSlot() if arm==ARMS[1] else None
        core=[s.centroids for s in sources.values()]
        groups=[{"params":core,"lr":.00015},{"params":region,"lr":.0003},{"params":[full.gates],"lr":.0001}]
        params=core+region+[full.gates]
        if aux is not None:
            groups.append({"params":list(aux.parameters()),"lr":.001})
            params=params+list(aux.parameters())
            assert sum(p.numel() for p in aux.parameters())==5120
        opt=torch.optim.Adam(groups);seen=set();gt=set();curve=[];aux_calls=0
        for step,i in enumerate(order):
            r=ds["TRAIN"][int(i)];assert r["split"]=="TRAIN"
            opt.zero_grad(set_to_none=True)
            z,z2=prefill(state,r,aux)
            loss=F.cross_entropy(z,torch.tensor([r["g"][0]]))/3
            if z2 is not None:
                # Future target from prompt ONLY. No gold first token input.
                loss=loss+.5*F.cross_entropy(z2,torch.tensor([r["g"][1]]))
                aux_calls+=1
            assert torch.isfinite(loss);loss.backward();v=float(loss.detach())
            for pos in (1,2):
                z=forward(state,r["p"]+r["g"][:pos],r["features"])
                l=F.cross_entropy(z,torch.tensor([r["g"][pos]]))/3
                assert torch.isfinite(l);l.backward();v+=float(l.detach())
            for k,s in sources.items():
                if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(k)
            if full.gates.grad is not None:gt.update(j for j,v in enumerate(full.gates.grad) if float(v.abs())>0)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for k,s in sources.items():s.centroids[k_for(RATE,k):].zero_()
            curve.append(v)
            if (step+1)%12==0:print("F36_PROGRESS",arm,step+1,v,len(seen),len(gt),flush=True)
        assert len(seen)==85 and len(gt)==12 and aux_calls==(STEPS if aux is not None else 0)
        ns=dict(side)
        original=load(side[NAMES[3]])
        changed={OLD_PREFIX+k:v.detach().cpu().half().contiguous() for k,v in child.state_dict().items()}
        assert set(changed)==set(original)
        assert all(torch.equal(v,original[k]) for k,v in changed.items() if ".nra3_neural_weights." in k)
        ns[NAMES[3]]=save(changed)
        gate_key=next(iter(load(side[NAMES[4]])))
        ns[NAMES[4]]=save({gate_key:full.gates.detach().cpu().half().contiguous()})
        compact,head,_=split_head(ns,8);cp=encode(archive.blob,sources,RATE,compact)
        packet=HDR.pack(b"FG09",len(cp),len(head),len(legacy))+cp+head+legacy
        a,s,l=open_bundle(packet);assert l==legacy
        (out/(arm+"_Q8.bundle")).write_bytes(packet)
        extra=b""
        if aux is not None:
            extra=save({k:v.detach().cpu().half().contiguous() for k,v in aux.state_dict().items()})
            (out/(arm+"_FUTURE_SLOT_FP16.safetensors")).write_bytes(extra)
        assert len(packet)+len(extra)<=3183292
        del opt,state,sources,full,child,region,core,params,groups,aux
        restored=direct(a,s);future=None
        if extra:
            future=FutureSlot();future.load_state_dict({k:v.float() for k,v in load(extra).items()},strict=True)
        after={k:audit(restored,rows,future) for k,rows in ds.items()};del restored
        results[arm]={"after":after,"steps":STEPS,"auxiliary_future_targets":aux_calls,
                      "core_training_forwards":STEPS*3,"connected_core":len(seen),"connected_GT":len(gt),
                      "new_research_parameters":5120 if extra else 0,
                      "aux_bytes":len(extra),"aux_sha256":digest(extra) if extra else None,
                      "known_payload":metrics(len(packet)+len(extra)),"core_packet_sha256":digest(packet),
                      "first12_objective":float(np.mean(curve[:12])),"last12_objective":float(np.mean(curve[-12:]))}
        print("F36_ARM",arm,json.dumps({**results[arm],"after":{k:brief(v) for k,v in after.items()}},sort_keys=True),flush=True)
    report={"stage":"F36_FULL_VOCAB_NATIVE_MTP","before":before,"arms":results,
            "manifest_sha256":digest(json.dumps({k:[(r["q"],r["g"]) for r in rows] for k,rows in ds.items()},ensure_ascii=False).encode()),
            "candidate_accepted":False,"output_gate":0,"FP32_teacher_calls":0,
            "limitations":["Two future positions, original full vocab, no source-token mask, second prediction has no first gold input.",
                           "Treatment adds 5120 temporary rank8 research parameters and extra objective/readout compute, not compute-matched; both arms same 144 causal training core forwards.",
                           "FP16 aux separately serialized and fully charged without changing original legacy head or its runtime. No permanent architecture approval.",
                           "Only short authored copy and computed digit-pair tasks; no universal generalization, protected replay, CPU speed, or all-in certification."]}
    (out/"F36_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("F36_ASSESS",json.dumps({"stage":report["stage"],"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f10,a.f08,a.out)
