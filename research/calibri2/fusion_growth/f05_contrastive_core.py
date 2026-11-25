"""F05: train correct answers AND discriminate/correct plausible wrong ones.
Matched positive-only vs contrastive objective on the physical F03 mixed-rate
core (K16 tied embedding, K8 body). No reference teacher. No sidecar activation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from research.calibri2.fusion_growth.f01_physical_budget import metrics,digest,PIN,NAMES
from research.calibri2.fusion_growth.f03_rate_train_core import make,encode,decode,k_for
from v176.operators import clear

PARENT_SHA="9e710de6db41d6bb2a99776080f045c537d4af2f6e971529ba6527bf993828d4"
RATE="K16_embed_K8_body"
SEED=20261127
STEPS=192
ARMS=("CE","CE_CONTRAST")
LAMBDA=.5
MARGIN=.5

def rows(split):
    # New operands, independently computed answers, explicit wrong rationale.
    # Holdout uses different prompt templates and disjoint operand ranges.
    n=48 if split=="TRAIN" else 12
    offset=0 if split=="TRAIN" else 200
    data=[]
    for i in range(n):
        a=24+3*i+offset;b=3+i%9
        percent=(10,20,25,50)[i%4]
        value=(4+i+offset)*20
        math=[
          (f"Сколько будет {a} + {b}?",f"Найди сумму чисел {a} и {b}.",a+b,a-b,"Вычитание вместо сложения."),
          (f"Вычисли {a} - {b}.",f"Уменьши {a} на {b}.",a-b,a+b,"Сложение вместо вычитания."),
          (f"Найди {percent}% от {value}.",f"Чему равны {percent} процентов числа {value}?",value*percent//100,value-percent,"Проценты нельзя вычитать как обычное число."),
          (f"Реши: x + {b} = {a}.",f"Какое число плюс {b} равно {a}?",a-b,a+b,"При переносе слагаемого знак меняется.")
        ]
        for kind,(p,q,y,w,why) in enumerate(math):
            assert y!=w and (kind!=2 or value*percent%100==0)
            prompt=p if split=="TRAIN" else q
            data.append({"prompt":prompt+" Ответь только числом.","good":str(y),"bad":str(w),"kind":kind,"why":why})
    assert len({x["prompt"] for x in data})==len(data)
    return data

def packs(tok,raw):
    out=[]
    for r in raw:
        p=r["prompt"]
        positive=" "+r["good"]
        negative=" "+r["bad"]
        correction=f" Ошибка. {r['why']} Правильный ответ: {r['good']}."
        q=f"{p}\nПредложенный ответ: {r['bad']}. Проверь и исправь."
        prompt=tok.encode(p+"\nОтвет:",add_special_tokens=False)
        good=tok.encode(positive,add_special_tokens=False)+[tok.eos_token_id]
        bad=tok.encode(negative,add_special_tokens=False)+[tok.eos_token_id]
        fixp=tok.encode(q+"\nПроверка:",add_special_tokens=False)
        fix=tok.encode(correction,add_special_tokens=False)+[tok.eos_token_id]
        assert good!=bad and max(len(prompt)+len(good),len(prompt)+len(bad),len(fixp)+len(fix))<160
        # Train on the first divergent native token with the exact same prefix.
        # No loss on negative continuation as a positive label.
        div=next(i for i,(a,b) in enumerate(zip(good,bad)) if a!=b)
        out.append(dict(r,p=prompt,g=good,b=bad,fixp=fixp,fix=fix,div=div))
    return out

def nll(model,sources,p,a):
    clear(sources)
    ids=torch.tensor([p+a],dtype=torch.long)
    logits=model(input_ids=ids,use_cache=False).logits[0,len(p)-1:-1].float()
    return F.cross_entropy(logits,torch.tensor(a)),logits

def main_loss(model,sources,row,arm):
    l,logits=nll(model,sources,row["p"],row["g"])
    i=row["div"]
    gap=logits[i,row["g"][i]]-logits[i,row["b"][i]]
    # Both arms receive exactly the same correct solution and correction
    # targets, same wrong answer as context in the correction prompt.
    # The treatment alone adds explicit negative preference supervision.
    fix,_=nll(model,sources,row["fixp"],row["fix"])
    return (l+fix)/2+(LAMBDA*F.softplus(MARGIN-gap) if arm=="CE_CONTRAST" else 0),gap

@torch.no_grad()
def evaluate(model,sources,rows,tok):
    nlls=[];gaps=[];fixes=[];correct=0;finish=0
    for row in rows:
        l,lg=nll(model,sources,row["p"],row["g"]);nlls.append(float(l))
        i=row["div"];gaps.append(float(lg[i,row["g"][i]]-lg[i,row["b"][i]]))
        f,_=nll(model,sources,row["fixp"],row["fix"]);fixes.append(float(f))
        p=row["p"].copy();gen=[]
        clear(sources)
        for _ in range(8):
            z=model(input_ids=torch.tensor([p]),use_cache=False).logits[0,-1]
            t=int(z.argmax());gen.append(t);p.append(t)
            if t==tok.eos_token_id:break
        finish+=int(gen[-1]==tok.eos_token_id)
        correct+=int(gen[-1]==tok.eos_token_id and tok.decode(gen,skip_special_tokens=True).strip()==row["good"])
        clear(sources)
    return {"n":len(rows),"answer_NLL":sum(nlls)/len(rows),"correction_NLL":sum(fixes)/len(rows),
            "correct_over_wrong_first_divergent":sum(x>0 for x in gaps),"mean_correct_wrong_gap":sum(gaps)/len(rows),
            "greedy_exact_EOS":correct,"EOS":finish,
            "pair_gaps":gaps}

def run(root,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    packet=(root/"K16_embed_K8_body.bundle").read_bytes()
    assert digest(packet)==PARENT_SHA
    archive,side=decode(packet)
    for k in NAMES[1:]:
        assert k in side
        if k in PIN:assert digest(side[k])==PIN[k]
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    tr,dv=rows("TRAIN"),rows("DEV")
    assert not set(x["prompt"] for x in tr)&set(x["prompt"] for x in dv)
    train,dev=packs(tok,tr),packs(tok,dv)
    assert len(train)==STEPS and len(dev)==48
    out.mkdir(parents=True,exist_ok=False)
    model,sources=make(archive);before=evaluate(model,sources,dev,tok);del model,sources
    print("F05_BEFORE",json.dumps(before),flush=True)
    order=np.random.default_rng(SEED).permutation(len(train))
    results={}
    for arm in ARMS:
        model,sources=make(archive)
        opt=torch.optim.Adam([s.centroids for s in sources.values()],lr=.0006)
        seen=set();curve=[]
        for step,index in enumerate(order):
            opt.zero_grad(set_to_none=True)
            l,gap=main_loss(model,sources,train[int(index)],arm)
            assert bool(torch.isfinite(l))
            l.backward();clear(sources)
            for k,s in sources.items():
                g=s.centroids.grad
                assert g is not None and bool(torch.isfinite(g).all())
                if float(g.abs().sum())>0:seen.add(k)
            torch.nn.utils.clip_grad_norm_([s.centroids for s in sources.values()],1.)
            opt.step()
            with torch.no_grad():
                for k,s in sources.items():s.centroids[k_for(RATE,k):].zero_()
            curve.append(float(l.detach()))
            if (step+1)%48==0:
                print("F05_PROGRESS",arm,step+1,curve[-1],flush=True)
        assert len(seen)==85
        packed=encode(archive.blob,sources,RATE,side)
        restored,rs=decode(packed);assert rs==side
        for k,s in sources.items():
            assert np.array_equal(restored.handles[k].centroids,s.centroids.detach().numpy().astype(np.float16).astype(np.float32))
            from direct_codebook_v59 import unpack5
            assert np.array_equal(unpack5(restored.handles[k].packed.tobytes(),s.codes.numel()),s.codes.numpy())
        (out/(arm+".bundle")).write_bytes(packed)
        del opt,model,sources
        model,sources=make(restored);after=evaluate(model,sources,dev,tok);del model,sources
        results[arm]={**metrics(len(packed)),"sha256":digest(packed),"optimizer_steps":STEPS,
                      "positive_and_correction_targets_per_step":2,"negative_preference_weight":LAMBDA if arm==ARMS[1] else 0,
                      "gradient_connected_codebooks":len(seen),"after_physical_reload":after,
                      "first24_training_objective":sum(curve[:24])/24,"last24_training_objective":sum(curve[-24:])/24}
        print("F05_ARM",arm,json.dumps(results[arm],sort_keys=True),flush=True)
    report={"stage":"F05_CONTRASTIVE_CORE","parent_sha256":PARENT_SHA,"depth":12,
            "before":before,"arms":results,"train_rows":len(train),"dev_rows":len(dev),
            "data_sha256":digest(json.dumps([tr,dv],ensure_ascii=False).encode()),
            "FP32_teacher_calls":0,"candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Internal synthetic arithmetic and correction prompts only; no independent generalization or free correction assessment.",
                           "Both arms get same positive and correction targets; only treatment adds first-divergence preference loss.",
                           "Core only: existing sidecars accounted and unchanged, not attached. No J7/CPU acceptance.",
                           "No own-generated error mining yet; negative examples deliberately authored."]}
    (out/"F05_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F05_ASSESS",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
