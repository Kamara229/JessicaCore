"""F16: diagnose native generation collapse and train a balanced readout.
Matched full-vocabulary CE versus CE+online hardest wrong token, including EOS.
Frozen F10 core and all earlier regions; cache only actual causal hidden/logit
states from the model itself. No external teacher or protected J7 training.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load,save
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,digest,metrics
from research.calibri2.fusion_growth.f03_rate_train_core import decode,encode
from research.calibri2.fusion_growth.f06_integrated_recovery import forward,OLD_PREFIX
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,open_bundle,HDR
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features

PARENT="694d031e340825c4d9667f99a674f95b69c4ade29c962e178d01a465bad49018"
RATE="K16_embed_K8_body"
GAIN=.25
SEED=20261204
EPOCHS=8
ARMS=("BALANCED_CE","BALANCED_HARD_NEGATIVE")

def corpus(split):
    tr=split=="TRAIN"
    words=("лес","дом","море","снег","свет","поле","река","ветер") if tr else ("сад","луна","гора","дождь","мост","камень","облако","парус")
    rows=[]
    for i,w in enumerate(words):
        a=(31 if tr else 71)+3*i;b=2+i
        for kind,q,y in [
            ("copy",f"Повтори только слово: {w}." if tr else f"Напиши без изменений слово «{w}».",w),
            ("number",f"Запиши число {a}." if tr else f"Какое число здесь написано: {a}?",str(a)),
            ("arithmetic",f"Сколько будет {a} + {b}?" if tr else f"Вычисли сумму {a} и {b}.",str(a+b)),
            ("logic",f"Верно ли, что {a} {'>' if i%2 else '<'} {b}? Ответь да или нет." if tr else f"Число {a} {'больше' if i%2 else 'меньше'} {b}? Напиши да или нет.","да" if i%2 else "нет")]:
            rows.append({"split":split,"kind":kind,"q":q,"answer":y})
    assert len(rows)==32 and len({r["q"] for r in rows})==32
    return rows

def pack(tok,rows):
    out=[]
    for r in rows:
        p,g=native(tok,r["q"],r["answer"])
        out.append({**r,"p":p,"g":g,"features":prompt_features(tok,r["q"])})
    return out

def cache_train(state,rows):
    model=state[0];head=state[5].language_head
    entries=[];max_error=0.;calls=0
    for r in rows:
        assert r["split"]=="TRAIN"
        h=[];z=[]
        for i,t in enumerate(r["g"]):
            captured=[]
            def hook(_m,args,base):
                captured.append((args[0][:,-1:,:].detach().clone(),base[:,-1,:].detach().float().clone()))
            handle=model.lm_head.register_forward_hook(hook)
            try:actual=forward(state,r["p"]+r["g"][:i],r["features"]).detach()
            finally:handle.remove()
            assert len(captured)==1
            x,b=captured[0];pred=b+GAIN*head(x)[:,0,:]
            err=float((actual-pred.detach()).abs().max())
            assert torch.allclose(actual,pred.detach(),atol=2e-5,rtol=1e-6)
            max_error=max(max_error,err);calls+=1;h.append(x);z.append(b)
        entries.append((torch.cat(h),torch.cat(z),torch.tensor(r["g"],dtype=torch.long)))
    return entries,{"actual_decoder_forwards":calls,"max_cache_logit_error":max_error}

def audit(state,rows,tok):
    hist=Counter();full=first=eos=empty=repeat=0;loss=[];ranks=[];records=[];by=Counter()
    for r in rows:
        nll=[]
        for i,t in enumerate(r["g"]):
            z=forward(state,r["p"]+r["g"][:i],r["features"]).detach()
            nll.append(float(F.cross_entropy(z,torch.tensor([t]))))
            if i==0:ranks.append(1+int((z[0]>z[0,t]).sum()))
        loss.append(sum(nll)/len(nll))
        own=[]
        for _ in range(8):
            z=forward(state,r["p"]+own,r["features"]).detach()
            own.append(int(z.argmax()))
            if own[-1]==2:break
        text=tok.decode(own,skip_special_tokens=True).strip()
        ok=own==r["g"];full+=int(ok);by[r["kind"]]+=int(ok)
        first+=int(own[0]==r["g"][0]);eos+=int(own[-1]==2);empty+=int(not text)
        repeat+=int(any(own[j]==own[j-1]==own[j-2] for j in range(2,len(own))))
        hist[own[0]]+=1
        records.append({"q":r["q"],"answer":r["answer"],"actual":text,"ids":own,"exact":ok})
    return {"n":len(rows),"NLL":sum(loss)/len(loss),"first_correct":first,
            "exact_native_EOS":full,"EOS":eos,"empty":empty,"triple_repetition":repeat,
            "distinct_first_tokens":len(hist),"first_token_histogram":dict(hist),
            "median_gold_first_rank":float(np.median(ranks)),"exact_by_family":dict(by),"records":records}

def brief(a):return {k:v for k,v in a.items() if k!="records"}

def run(f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(SEED)
    packet=(f10/"F10_FP16.bundle").read_bytes();assert digest(packet)==PARENT
    archive,side=decode(packet)
    legacy=(f08/"legacy_extras.bundle").read_bytes();assert len(legacy)==137712
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    train,dev=pack(tok,corpus("TRAIN")),pack(tok,corpus("DEV"))
    assert not set(r["q"] for r in train)&set(r["q"] for r in dev)
    out.mkdir(parents=True,exist_ok=False)
    state=direct(archive,side)
    before={"TRAIN":audit(state,train,tok),"DEV":audit(state,dev,tok)}
    print("F16_BEFORE",json.dumps({k:brief(v) for k,v in before.items()}),flush=True)
    cached,witness=cache_train(state,train)
    del state
    results={}
    for arm in ARMS:
        state=direct(archive,side);head=state[5].language_head
        params=list(head.parameters())
        assert sum(p.numel() for p in params)==247040
        opt=torch.optim.Adam(params,lr=.003)
        order=np.random.default_rng(SEED);curve=[]
        for epoch in range(EPOCHS):
            for ix in order.permutation(len(cached)):
                h,z,g=cached[int(ix)]
                opt.zero_grad(set_to_none=True)
                pred=z+GAIN*head(h)[:,0,:]
                ce=F.cross_entropy(pred,g)
                l=ce
                if arm==ARMS[1]:
                    neg=pred.detach().clone();neg.scatter_(1,g[:,None],-float("inf"))
                    bad=neg.argmax(1)
                    gap=pred.gather(1,g[:,None])-pred.gather(1,bad[:,None])
                    l=l+.5*F.softplus(1.-gap).mean()
                assert torch.isfinite(l)
                l.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
                curve.append(float(ce.detach()))
        original=load(side[NAMES[3]])
        changed={OLD_PREFIX+k:v.detach().cpu().half().contiguous() for k,v in state[5].state_dict().items()}
        assert set(changed)==set(original)
        assert all(torch.equal(changed[k],v) for k,v in original.items() if ".language_head." not in k)
        ns=dict(side);ns[NAMES[3]]=save(changed)
        fp=encode(archive.blob,state[1],RATE,ns);ar,rs=decode(fp)
        assert ar.blob==archive.blob and rs==ns
        compact,h,_=split_head(ns,8);cp=encode(archive.blob,state[1],RATE,compact)
        qp=HDR.pack(b"FG09",len(cp),len(h),len(legacy))+cp+h+legacy
        qa,qs,ql=open_bundle(qp);assert qa.blob==archive.blob and ql==legacy
        del opt,state,head,params
        (out/(arm+"_FP16.bundle")).write_bytes(fp);(out/(arm+"_Q8.bundle")).write_bytes(qp)
        final=direct(ar,rs);fp16={"TRAIN":audit(final,train,tok),"DEV":audit(final,dev,tok)};del final
        final=direct(qa,qs);q8={"TRAIN":audit(final,train,tok),"DEV":audit(final,dev,tok)};del final
        results[arm]={"FP16":fp16,"Q8":q8,"Q8_plus_legacy":metrics(len(qp)),
                      "FP16_plus_legacy":metrics(len(fp)+len(legacy)),
                      "Q8_sha256":digest(qp),"FP16_sha256":digest(fp),"updates":len(curve),
                      "first32_train_CE":sum(curve[:32])/32,"last32_train_CE":sum(curve[-32:])/32}
        print("F16_ARM",arm,json.dumps({**results[arm],"FP16":{k:brief(v) for k,v in fp16.items()},"Q8":{k:brief(v) for k,v in q8.items()}},sort_keys=True),flush=True)
    report={"stage":"F16_NATIVE_CALIBRATION","parent_sha256":PARENT,"depth":12,
            "before":before,"cache_witness":witness,"arms":results,
            "head_parameters":247040,"FP32_teacher_calls":0,"candidate_accepted":False,
            "output_gate":0,"all_in_certified":False,
            "limitations":["Only head trained, core/GT/other regions unchanged. Equal CE on all native content/EOS tokens; hardest wrong target computed from current logits, not a teacher.",
                           "32 authored TRAIN repeated eight times, 32 authored DEV; deliberate small fit/generalization diagnosis, not independent capability benchmark.",
                           "Real free generation after FP16/Q8 reload; legacy extras paid but inactive, no J7/CPU/runtime certification."]}
    (out/"F16_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("F16_ASSESS",json.dumps({"stage":report["stage"],"cache_witness":witness,"candidate_accepted":False}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f10,a.f08,a.out)
