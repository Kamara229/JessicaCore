"""G09: two native future tokens from one actual 133.701M prefill.
Matched autoregressive control, interleaved copy/computed TRAIN, shared
tied vocabulary, rank32 auxiliary residual. No gold first token supplied
to the future slot. Compare independent greedy outputs and decoder calls.
Pinned G05 while G06/G08 run; no deployment or protected fitting.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load,save
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import (
    payload,extra_pack,digest,open_bundle,RATE,k_for)
from research.calibri2.fusion_growth.g02_attention_discrete import indexed_banks,forward
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore,region_split,rate
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.f36_native_multitoken import dataset,prefill,audit,brief

SEED=20270101
RANK=32
EPOCHS=2
ARMS=("AR_CONTROL","AR_PLUS_MTP")

class FutureSlot(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Linear(320,RANK,bias=False)
        self.up=nn.Linear(RANK,320,bias=False)
        nn.init.normal_(self.down.weight,std=.02);nn.init.zeros_(self.up.weight)
    def forward(self,h):
        return h+self.up(F.gelu(self.down(F.layer_norm(h,(320,)))))

def run(g05,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    ar,sd,legacy=open_bundle(blobs[0])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tokenizer_bytes=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    ds=dataset(tok)
    assert all(len(r["g"])==3 and r["g"][-1]==2 for rows in ds.values() for r in rows)
    rng=np.random.default_rng(SEED)
    copy=[r for r in ds["TRAIN"] if r["kind"]=="copy"]
    math=[r for r in ds["TRAIN"] if r["kind"]=="computed"]
    assert len(copy)==len(math)==24
    schedule=[]
    for _ in range(EPOCHS):
        schedule.extend(r for pair in zip([copy[int(i)] for i in rng.permutation(24)],
                                          [math[int(i)] for i in rng.permutation(24)]) for r in pair)
    assert len(schedule)==96
    out.mkdir(parents=True,exist_ok=False)
    def reload(b):return restore(b[:6],b[6],b[7])
    st,*rest=reload(blobs)
    before=audit(st,ds["DEV"])
    del st,rest
    print("G09_BASE",json.dumps(brief(before),sort_keys=True),flush=True)
    results={}
    for arm in ARMS:
        st,second,added,fourth,reg,old,fifth,sixth,reg2=reload(blobs)
        _,sources,_,full,_,child,region=st
        ext=indexed_banks(second,added)|{"fourth/"+k:v for k,v in fourth.items()}|{"fifth/"+k:v for k,v in fifth.items()}|{"sixth/"+k:v for k,v in sixth.items()}
        assert len(ext)==324
        core=[s.centroids for s in sources.values()]+[s.centroids for s in ext.values()]
        torch.manual_seed(SEED)
        aux=FutureSlot() if arm==ARMS[1] else None
        groups=[{"params":core,"lr":.00015},{"params":region,"lr":.0003},{"params":[full.gates],"lr":.0001}]
        params=core+region+[full.gates]
        if aux is not None:
            groups.append({"params":list(aux.parameters()),"lr":.001})
            params+=list(aux.parameters())
        assert len(params)==len({id(p) for p in params})
        opt=torch.optim.Adam(groups);seen=set();gates=set();aux_grad=set();curve=[];calls=0
        for i,r in enumerate(schedule):
            assert r["split"]=="TRAIN"
            opt.zero_grad(set_to_none=True)
            z,z2=prefill(st,r,aux)
            loss=F.cross_entropy(z,torch.tensor([r["g"][0]]))/3
            if z2 is not None:
                loss+=.5*F.cross_entropy(z2,torch.tensor([r["g"][1]]))
                calls+=1
            assert torch.isfinite(loss)
            loss.backward();v=float(loss.detach())
            for pos in (1,2):
                z=forward(st,r["p"]+r["g"][:pos],r["features"])
                l=F.cross_entropy(z,torch.tensor([r["g"][pos]]))/3
                assert torch.isfinite(l);l.backward();v+=float(l.detach())
            for k,s in list(sources.items())+list(ext.items()):
                if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(k)
            if full.gates.grad is not None:gates.update(j for j,vv in enumerate(full.gates.grad) if float(vv.abs())>0)
            if aux is not None:
                aux_grad.update(k for k,p in aux.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for k,s in sources.items():s.centroids[k_for(RATE,k):].zero_()
                for s in ext.values():s.centroids[8:].zero_()
            curve.append(v)
            if (i+1)%16==0:print("G09_PROGRESS",arm,i+1,v,len(seen),len(gates),flush=True)
        assert len(seen)==409 and len(gates)==12
        assert calls==(96 if aux is not None else 0)
        if aux is not None:assert aux_grad=={"down.weight","up.weight"}
        _,packet,_,_=payload(ar,sd,st,legacy)
        rb,rb2=region_split(st,reg,old,reg2)
        vals=(packet,extra_pack(second),extra_pack(added),extra_pack(fourth),rb,extra_pack(fifth),extra_pack(sixth),rb2)
        ab=save({k:v.detach().half().contiguous() for k,v in aux.state_dict().items()}) if aux is not None else b""
        storage=rate(vals[:6],vals[6],vals[7],G05,tokenizer_bytes)
        storage["aux_bytes"]=len(ab)
        storage["with_aux_and_tokenizer_bpp"]=8*(storage["bytes"]+len(ab)+tokenizer_bytes)/G05
        assert storage["with_aux_and_tokenizer_bpp"]<=.7
        for k,v in zip(PIN,vals):(out/(arm+"_"+k)).write_bytes(v)
        if ab:(out/(arm+"_FUTURE_SLOT.safetensors")).write_bytes(ab)
        del st,second,added,fourth,reg,old,fifth,sixth,reg2,ext,opt,params,groups,core,region,child,full,sources,aux
        st,*rest=reload(vals)
        future=None
        if ab:
            future=FutureSlot();future.load_state_dict({k:v.float() for k,v in load(ab).items()},strict=True)
        after={k:audit(st,rows,future) for k,rows in ds.items()}
        del st,rest
        info={"after":after,"connected_codebooks":len(seen),"connected_GT":len(gates),
              "aux_connected":sorted(aux_grad),"steps":96,"native_core_training_calls":288,
              "future_slot_training_calls":calls,"first16_objective":float(np.mean(curve[:16])),
              "last16_objective":float(np.mean(curve[-16:])),"storage":storage,
              "sha256":dict(zip(PIN,[digest(v) for v in vals])),"aux_sha256":digest(ab) if ab else None}
        results[arm]=info
        print("G09_ARM",arm,json.dumps({**info,"after":{k:brief(v) for k,v in after.items()}},sort_keys=True),flush=True)
    report={"stage":"G09_SCALED_NATIVE_PARALLEL_TWO_TOKEN","geometry":G05,"depth":12,
            "TRAIN_unique":48,"DEV_unique":24,"before":before,"arms":results,
            "candidate_accepted":False,"FP32_teacher_calls":0,"output_gate":0,
            "limitations":["Pinned G05, not unfinished G06/G08; reused F36 diagnostic corpus, not independent blind QA.",
                           "Full 49152 vocabulary and same tied head; two positions from one prefix, no gold first token to future slot.",
                           "Both arms 288 core training forwards; MTP adds readout/loss compute and 20480 FP16 parameters, not equal total FLOPs.",
                           "MTP greedy mode unverified proposals, decoder-call savings alone not speed or quality gain. No protected replay, historical CPU certification or executable runtime budget."]}
    (out/"G09_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G09_ASSESS",json.dumps({"stage":report["stage"],"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
