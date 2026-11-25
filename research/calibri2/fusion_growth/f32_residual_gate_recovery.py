"""F32: matched joint training after reducing the existing middle residual.
F31 shows the child adds a near-common large vector, reducing relative
signal. Change only initialization of its existing three tanh gates.
No permanent hook, new weights, loss change, or protected training.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load,save
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,digest,metrics
from research.calibri2.fusion_growth.f03_rate_train_core import decode,encode,k_for
from research.calibri2.fusion_growth.f06_integrated_recovery import forward,OLD_PREFIX
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,open_bundle,HDR
from research.calibri2.fusion_growth.f20_state_diagnostic import F10_SHA
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features

RATE="K16_embed_K8_body"
ARMS=("ORIGINAL_GATE_INIT","REDUCED_GATE_INIT")
SEED=20261211
EPOCHS=1
VOCAB=49152

def groups(tok,split):
    tr=split=="TRAIN"
    n=12 if tr else 8
    rng=np.random.default_rng(SEED+(0 if tr else 1))
    names=("Анна","Борис","Вера","Глеб","Дина","Егор","Жанна","Илья") if tr else ("Лида","Марк","Нина","Олег","Рита","Тимур","Ульяна","Фёдор")
    objects=("мяч","лист","зонт","куб","ключ","пакет","шар","фонарь") if tr else ("билет","жетон","чашка","часы","лампа","камень","книга","блокнот")
    result=[]
    for i in range(n):
        family=("owner","position")[i%2]
        words=names if family=="owner" else objects
        chosen=[words[j] for j in rng.choice(len(words),4,replace=False)]
        values=list(map(int,rng.permutation(np.arange(1,5))))
        if family=="owner":
            context="; ".join(f"{word}: {v}" for word,v in zip(chosen,values))
            q=lambda word: (f"Список кодов: {context}. Какой код у {word}? Ответь цифрой." if tr else f"В записи ({context}) указаны коды. Назови цифру, которая соответствует имени {word}.")
        else:
            values=[1,2,3,4]
            context=", ".join(chosen)
            q=lambda word: (f"Предметы по порядку: {context}. На каком месте {word}? Ответь цифрой." if tr else f"Ряд предметов [{context}], счёт начинается с единицы. Укажи номер позиции предмета «{word}».")
        rows=[]
        for word,y in zip(chosen,values):
            text=q(word)
            p,g=native(tok,text,str(y))
            assert len(g)==2 and g[-1]==2
            rows.append({"q":text,"p":p,"g":g,"label":y,"kind":family,"split":split,"features":prompt_features(tok,text)})
        result.append(sorted(rows,key=lambda x:x["label"]))
    assert len({r["q"] for group in result for r in group})==4*n
    assert len({r["g"][0] for group in result for r in group})==4
    return result

def get_hidden(state,row):
    cap=[]
    def hook(_m,args):cap.append(args[0][0,-1].float())
    h=state[0].lm_head.register_forward_pre_hook(hook)
    try:z=forward(state,row["p"],row["features"])
    finally:h.remove()
    assert len(cap)==1 and cap[0].shape==(320,) and cap[0].requires_grad
    assert z.shape==(1,VOCAB) and torch.isfinite(z).all()
    return z,F.normalize(cap[0],dim=0,eps=1e-8)

def audit(state,rows,tok):
    results=[];nll=[];first=0;exact=0;eos=0;hist=Counter();by=Counter()
    hs=[]
    for row in rows:
        z,h=get_hidden(state,row);hs.append(h.detach().cpu().numpy().copy())
        costs=[float(F.cross_entropy(z.detach(),torch.tensor([row["g"][0]])))]
        z2=forward(state,row["p"]+row["g"][:1],row["features"]).detach()
        costs.append(float(F.cross_entropy(z2,torch.tensor([2]))))
        nll.append(sum(costs)/2)
        ids=[int(z.detach().argmax())];hist[ids[0]]+=1
        for _ in range(5):
            if ids[-1]==2:break
            ids.append(int(forward(state,row["p"]+ids,row["features"]).detach().argmax()))
        ok=ids==row["g"];first+=int(ids[0]==row["g"][0]);exact+=int(ok);eos+=int(ids[-1]==2)
        by[row["kind"]]+=int(ok)
        results.append({"q":row["q"],"expected":row["g"],"generated":ids,"exact":ok})
    h=np.stack(hs);labels=np.array([r["label"] for r in rows])
    d=h@h.T;mask=~np.eye(len(h),dtype=bool)
    same=(labels[:,None]==labels[None,:])&mask;diff=labels[:,None]!=labels[None,:]
    return {"n":len(rows),"first_correct":first,"exact_native_EOS":exact,"EOS":eos,
            "NLL":sum(nll)/len(nll),"first_token_histogram":dict(hist),
            "same_answer_cosine":float(d[same].mean()),"different_answer_cosine":float(d[diff].mean()),
            "exact_by_family":dict(by),"records":results}
def brief(x):return {k:v for k,v in x.items() if k!="records"}

def run(f10,f08,out):
    torch.set_num_threads(2);torch.manual_seed(SEED)
    packet=(f10/"F10_FP16.bundle").read_bytes();assert digest(packet)==F10_SHA
    archive,side=decode(packet)
    legacy=(f08/"legacy_extras.bundle").read_bytes();assert len(legacy)==137712
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tr,dv=groups(tok,"TRAIN"),groups(tok,"DEV")
    train=[r for g in tr for r in g];dev=[r for g in dv for r in g]
    assert len(train)==48 and len(dev)==32 and not set(r["q"] for r in train)&set(r["q"] for r in dev)
    # Fixed triplet order: same-answer positive from another context in
    # same family; counterfactual negative differs only in requested entity.
    schedule=[]
    for ep in range(EPOCHS):
        for i in range(len(tr)):
            for label in range(4):
                pos=(i+2+2*ep)%len(tr);neg=(label+ep+1)%4
                a,p,n=tr[i][label],tr[pos][label],tr[i][neg]
                assert a["q"]!=p["q"] and a["kind"]==p["kind"]==n["kind"]
                assert a["label"]==p["label"]!=n["label"]
                schedule.append((a,p,n))
    assert len(schedule)==48
    manifest=[[(r["q"],r["label"]) for r in t] for t in schedule]
    out.mkdir(parents=True,exist_ok=False)
    state=direct(archive,side)
    before={"TRAIN":audit(state,train,tok),"DEV":audit(state,dev,tok)};del state
    print("F32_BEFORE",json.dumps({k:brief(v) for k,v in before.items()}),flush=True)
    results={}
    for arm in ARMS:
        state=direct(archive,side)
        _,sources,_,full,_,child,region=state
        gate=child.child.child.midlayer_gates
        assert gate.shape==(3,) and gate.requires_grad
        original_gate=gate.detach().clone()
        if arm==ARMS[1]:
            with torch.no_grad():
                gate.copy_(torch.atanh(.1*torch.tanh(original_gate)))
            assert torch.allclose(gate.tanh(),.1*original_gate.tanh(),atol=1e-7,rtol=1e-6)
        initial_gate=gate.detach().tolist()
        initial_effective_gate=gate.detach().tanh().tolist()
        core=[s.centroids for s in sources.values()]
        params=core+region+[full.gates]
        opt=torch.optim.Adam([{"params":core,"lr":.00015},{"params":region,"lr":.0003},{"params":[full.gates],"lr":.0001}])
        order=np.random.default_rng(SEED).permutation(len(schedule))
        seen=set();gates=set();curve=[]
        for step,ix in enumerate(order):
            opt.zero_grad(set_to_none=True)
            trip=schedule[int(ix)]
            assert all(r["split"]=="TRAIN" for r in trip)
            reps=[];ce=[]
            for r in trip:
                z,h=get_hidden(state,r);reps.append(h)
                ce.append(F.cross_entropy(z,torch.tensor([r["g"][0]])))
            a,p,n=reps
            contrast=F.relu(.2+(a*n).sum()-(a*p).sum())
            first_loss=torch.stack(ce).mean()*.5
            loss=first_loss
            assert torch.isfinite(loss)
            loss.backward()
            # Equal full-vocabulary EOS CE in both arms. No EOS mask.
            for r in trip:
                z=forward(state,r["p"]+r["g"][:1],r["features"])
                (F.cross_entropy(z,torch.tensor([2]))/6).backward()
            for name,s in sources.items():
                if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(name)
            if full.gates.grad is not None:gates.update(j for j,v in enumerate(full.gates.grad) if float(v.abs())>0)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for name,s in sources.items():s.centroids[k_for(RATE,name):].zero_()
            curve.append(float(contrast.detach()))
            if (step+1)%24==0:print("F32_PROGRESS",arm,step+1,float(loss.detach()),curve[-1],flush=True)
        assert len(seen)==85 and len(gates)==12
        final_effective_gate=gate.detach().tanh().tolist()
        original=load(side[NAMES[3]])
        ns=dict(side)
        changed={OLD_PREFIX+k:v.detach().cpu().half().contiguous() for k,v in child.state_dict().items()}
        assert set(changed)==set(original)
        assert all(torch.equal(v,original[k]) for k,v in changed.items() if ".nra3_neural_weights." in k)
        ns[NAMES[3]]=save(changed)
        gate_key=next(iter(load(side[NAMES[4]])))
        ns[NAMES[4]]=save({gate_key:full.gates.detach().cpu().half().contiguous()})
        fp=encode(archive.blob,sources,RATE,ns);a,s=decode(fp);assert s==ns
        compact,head,_=split_head(ns,8)
        cp=encode(archive.blob,sources,RATE,compact)
        qp=HDR.pack(b"FG09",len(cp),len(head),len(legacy))+cp+head+legacy
        qa,qs,ql=open_bundle(qp);assert qa.blob==a.blob and ql==legacy
        del opt,state,params,core,region,sources,full,child
        (out/(arm+"_FP16.bundle")).write_bytes(fp);(out/(arm+"_Q8.bundle")).write_bytes(qp)
        fpstate=direct(a,s);fp_a={"TRAIN":audit(fpstate,train,tok),"DEV":audit(fpstate,dev,tok)};del fpstate
        qstate=direct(qa,qs);q_a={"TRAIN":audit(qstate,train,tok),"DEV":audit(qstate,dev,tok)};del qstate
        results[arm]={"FP16":fp_a,"Q8":q_a,"steps":len(schedule),"initial_gate":initial_gate,"initial_effective_gate":initial_effective_gate,"final_effective_gate":final_effective_gate,"connected_core":len(seen),"connected_GT":len(gates),
                      "Q8_plus_legacy":metrics(len(qp)),"Q8_sha256":digest(qp),"FP16_sha256":digest(fp),
                      "first24_triplet_loss":sum(curve[:24])/24,"last24_triplet_loss":sum(curve[-24:])/24}
        print("F32_ARM",arm,json.dumps({**results[arm],"FP16":{k:brief(v) for k,v in fp_a.items()},"Q8":{k:brief(v) for k,v in q_a.items()}},sort_keys=True),flush=True)
    report={"stage":"F32_RESIDUAL_GATE_RECOVERY","parent_sha256":F10_SHA,"depth":12,
            "schedule_sha256":digest(json.dumps(manifest,ensure_ascii=False).encode()),
            "before":before,"arms":results,"new_parameters":0,"FP32_teacher_calls":0,
            "candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Small authored two-family diagnostic, not proof of broad reasoning. Identical TRAIN triplets and full vocabulary CE plus EOS; only existing midlayer tanh-gate initialization differs (0.1x effective gain). State contrast logged but not optimized in either arm.",
                           "Uses unaccepted F10 parent. No permanent runtime override: trained gates serialized in original tensor layout. No protected J7 data or routing changes.",
                           "Full free generation after physical reload, not masked four-class output. Native quality, full retention and historical CPU gates remain pending."]}
    (out/"F32_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("F32_ASSESS",json.dumps({"stage":report["stage"],"candidate_accepted":False}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f10","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f10,a.f08,a.out)
