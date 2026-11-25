"""F11: broad joint recovery, independent of the running math-only F10.

Real F06 physical parent, 384 TRAIN examples in eight families, 96 held-out
authored diagnostics. Correct answers, plausible contrast, and explicit repairs.
No protected J7 training, no FP32 teacher. All results remain research-only.
"""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f03_rate_train_core import decode,encode,k_for
from research.calibri2.fusion_growth.f06_integrated_recovery import build,evaluate,OLD_PREFIX
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native,supervised
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,open_bundle,HDR
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features

PARENT_SHA="7dcc391ce6001f88f6e7c12498c4f371b83431b049e4db2cb65d7f0b89ab722a"
RATE="K16_embed_K8_body"
SEED=20261130
STEPS=384
TRAIN_NAMES=("Анна","Борис","Вера","Глеб","Дина","Егор","Жанна","Илья")
DEV_NAMES=("Лидия","Марк","Нина","Олег","Рита","Степан","Тамара","Фёдор")
TRAIN_OBJECTS=("ключ","мяч","лист","зонт","шар","куб","пакет","фонарь")
DEV_OBJECTS=("камень","карандаш","блокнот","конверт","часы","билет","жетон","рюкзак")
TRAIN_NOUNS=(("стол","m"),("дом","m"),("книга","f"),("сумка","f"),("окно","n"),("письмо","n"))
DEV_NOUNS=(("мост","m"),("сад","m"),("чашка","f"),("лампа","f"),("зеркало","n"),("кольцо","n"))
TRAIN_ADJ=(("новый","новая","новое"),("старый","старая","старое"),("белый","белая","белое"),("красный","красная","красное"),
           ("синий","синяя","синее"),("большой","большая","большое"),("маленький","маленькая","маленькое"),("жёлтый","жёлтая","жёлтое"))
DEV_ADJ=(("зелёный","зелёная","зелёное"),("чёрный","чёрная","чёрное"))
FAMILIES=("arithmetic","comparison","temporal","spatial","negation","semantic_role","context","agreement")

def corpus(split):
    assert split in ("TRAIN","DEV")
    train=split=="TRAIN"
    n=48 if train else 12
    names=TRAIN_NAMES if train else DEV_NAMES
    objects=TRAIN_OBJECTS if train else DEV_OBJECTS
    nouns=TRAIN_NOUNS if train else DEV_NOUNS
    adjs=TRAIN_ADJ if train else DEV_ADJ
    out=[]
    for i in range(n):
        a=(101 if train else 9001)+19*i
        b=3+i%11
        x=names[i%8];y=names[(i//8+i+1)%8]
        if x==y:y=names[(i%8+1)%8]
        obj=objects[(i*3+i//8)%8]
        # Numeric identifiers make each source distinct without changing the relation.
        ident=f"Ситуация {a}. "
        def add(kind,q,g,w):
            g,w=str(g),str(w)
            assert g!=w
            out.append({"split":split,"id":f"F11_{split}_{kind}_{i:03d}",
                        "kind":kind,"q":q,"answer":g,"wrong":w})
        add("arithmetic",
            (f"Вычисли {a} - {b}." if i%2 else f"Вычисли {a} + {b}.") if train else
            (f"Уменьши {a} на {b}." if i%2 else f"Прибавь {b} к {a}."),
            a-b if i%2 else a+b,a+b if i%2 else a-b)
        low,high=a,a+b
        u,v=(low,high) if i%2 else (high,low)
        add("comparison",f"Какое число больше: {u} или {v}?" if train else
            f"Назови большее из двух чисел {u} и {v}.",high,low)
        first,second=(x,y) if i%2 else (y,x)
        ask_first=bool((i//2)%2)
        add("temporal",ident+(
            f"{first} пришёл раньше, чем {second}. Кто пришёл {'раньше' if ask_first else 'позже'}?" if train else
            f"Сначала появился {first}, затем {second}. Кто был {'первым' if ask_first else 'вторым'}?"),
            first if ask_first else second,second if ask_first else first)
        left,right=(x,y) if i%2 else (y,x)
        ask_left=bool((i//2)%2)
        add("spatial",ident+(
            f"{left} стоит слева от {right}. Кто {'слева' if ask_left else 'справа'}?" if train else
            f"{right} расположен справа от {left}. Назови того, кто находится {'левее' if ask_left else 'правее'}."),
            left if ask_left else right,right if ask_left else left)
        has=bool(i%2)
        add("negation",ident+(
            f"У {x} {'есть' if has else 'нет'} предмета: {obj}. Есть ли у {x} этот предмет? Ответь да или нет." if train else
            f"{x}: предмет {obj} {'имеется' if has else 'отсутствует'}. Верно ли, что предмет имеется? Ответь да или нет."),
            "да" if has else "нет","нет" if has else "да")
        giver,receiver=(x,y) if i%2 else (y,x)
        ask_giver=bool((i//2)%2)
        add("semantic_role",ident+(
            f"{giver} передал предмет ({obj}) человеку по имени {receiver}. Кто {'передал' if ask_giver else 'получил'} предмет?" if train else
            f"{receiver} получил предмет ({obj}) от человека по имени {giver}. Назови {'отправителя' if ask_giver else 'получателя'}."),
            giver if ask_giver else receiver,receiver if ask_giver else giver)
        own_a,own_b=(a,a+b) if i%2 else (a+b,a)
        ask_a=bool((i//2)%2)
        add("context",(
            f"Код человека {x} — {own_a}, код человека {y} — {own_b}. Какой код у {x if ask_a else y}?" if train else
            f"Запись: {y} = {own_b}; {x} = {own_a}. Назови код, записанный для {x if ask_a else y}."),
            own_a if ask_a else own_b,own_b if ask_a else own_a)
        noun,gender=nouns[i%6];forms=adjs[(i//6)%len(adjs)]
        ix={"m":0,"f":1,"n":2}[gender];g=forms[ix];w=forms[(ix+1)%3]
        choices=(g,w) if (i//2)%2 else (w,g)
        add("agreement",
            f"Выбери согласованное прилагательное для слова «{noun}»: {choices[0]} или {choices[1]}." if train else
            f"Какое из слов «{choices[0]}», «{choices[1]}» согласуется с существительным «{noun}»?",g,w)
    assert len(out)==8*n and len({r["q"] for r in out})==len(out)
    assert Counter(r["kind"] for r in out)=={k:n for k in FAMILIES}
    return out

def pack(tok,rows):
    result=[]
    for r in rows:
        p,g=native(tok,r["q"],r["answer"])
        p2,bad=native(tok,r["q"],r["wrong"])
        assert p==p2 and g!=bad and len(p)+max(len(g),len(bad))+8<160
        div=next(j for j,(x,y) in enumerate(zip(g,bad)) if x!=y)
        result.append({**r,"p":p,"g":g,"bad":bad,"div":div,
                       "features":prompt_features(tok,r["q"])})
    return result

def audit(state,dev,tok):
    return {"all":evaluate(state,dev,tok),
            "by_family":{k:evaluate(state,[r for r in dev if r["kind"]==k],tok) for k in FAMILIES}}

def run(f01,f06,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    raw=(f06/"F06_JOINT.bundle").read_bytes();assert digest(raw)==PARENT_SHA
    archive,side=decode(raw)
    original=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(original[k])==v
    assert all(side[k]==original[k] for k in NAMES[1:3])
    legacy=(f08/"legacy_extras.bundle").read_bytes();assert len(legacy)==137712
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    tr,dv=corpus("TRAIN"),corpus("DEV")
    assert not set(r["q"] for r in tr)&set(r["q"] for r in dv)
    train,dev=pack(tok,tr),pack(tok,dv)
    assert len(train)==STEPS and len(dev)==96
    out.mkdir(parents=True,exist_ok=False)
    (out/"F11_DATA_MANIFEST.json").write_text(json.dumps({"TRAIN":tr,"DEV":dv},ensure_ascii=False,indent=2))
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/NAMES[0]).write_bytes(original[NAMES[0]])
        for k,v in side.items():(root/k).write_bytes(v)
        state=build(root,archive)
        before=audit(state,dev,tok)
        print("F11_BEFORE",json.dumps(before,sort_keys=True),flush=True)
        model,sources,bridge,full,expansion,child,region=state
        core=[s.centroids for s in sources.values()]
        opt=torch.optim.Adam([{"params":core,"lr":.00015},{"params":region,"lr":.0003},
                              {"params":[full.gates],"lr":.0001}])
        order=np.random.default_rng(SEED).permutation(STEPS)
        seen=set();regions=set();gates=set();repairs=0;curve=[]
        for i,j in enumerate(order):
            row=train[int(j)]
            opt.zero_grad(set_to_none=True)
            l=supervised(state,tok,row["q"],row["answer"],row["bad"])
            # One repair per eight rows, chosen by fixed TRAIN order, not DEV.
            if i%8==0:
                q=row["q"]+" Предложенный ответ: "+row["wrong"]+". Исправь ошибку."
                l+=supervised(state,tok,q,"Верный ответ: "+row["answer"]+".")
                repairs+=1
            for k,s in sources.items():
                if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(k)
            if full.gates.grad is not None:gates.update(k for k,x in enumerate(full.gates.grad) if float(x.abs())>0)
            for k,p in child.named_parameters():
                if p.grad is not None and float(p.grad.abs().sum())>0:regions.add(k)
            torch.nn.utils.clip_grad_norm_(core+region+[full.gates],1.,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for k,s in sources.items():s.centroids[k_for(RATE,k):].zero_()
            curve.append(l)
            if (i+1)%64==0:print("F11_PROGRESS",i+1,l,len(seen),len(gates),repairs,flush=True)
        assert len(seen)==85 and len(gates)==12 and regions
        new_side=dict(side)
        new_side[NAMES[3]]=save({OLD_PREFIX+k:v.detach().cpu().half().contiguous() for k,v in child.state_dict().items()})
        gate_key=next(iter(load_file(str(root/NAMES[4]))))
        new_side[NAMES[4]]=save({gate_key:full.gates.detach().cpu().half().contiguous()})
        full_packet=encode(archive.blob,sources,RATE,new_side)
        restored,rs=decode(full_packet);assert rs==new_side
        (out/"F11_FP16.bundle").write_bytes(full_packet)
        compact,head,_=split_head(new_side,8)
        cp=encode(archive.blob,sources,RATE,compact)
        packed=HDR.pack(b"FG09",len(cp),len(head),len(legacy))+cp+head+legacy
        qcore,qside,qlegacy=open_bundle(packed)
        assert qcore.blob==restored.blob and qlegacy==legacy
        (out/"F11_Q8.bundle").write_bytes(packed)
        del opt,state,model,sources,bridge,full,expansion,child,region,core
        for k,v in rs.items():(root/k).write_bytes(v)
        final=build(root,restored);fp16=audit(final,dev,tok);del final
        for k,v in qside.items():(root/k).write_bytes(v)
        final=build(root,qcore);q8=audit(final,dev,tok);del final
    report={"stage":"F11_BROAD_RECOVERY","parent_sha256":PARENT_SHA,"depth":12,
            "before":before,"after_FP16":fp16,"after_Q8":q8,
            "FP16_plus_legacy":metrics(len(full_packet)+len(legacy)),
            "Q8_plus_legacy":metrics(len(packed)),"Q8_sha256":digest(packed),
            "FP16_sha256":digest(full_packet),"steps":STEPS,"repair_updates":repairs,
            "train_families":dict(Counter(r["kind"] for r in tr)),
            "data_sha256":digest((out/"F11_DATA_MANIFEST.json").read_bytes()),
            "connected_core_codebooks":len(seen),"connected_GT_gates":sorted(gates),
            "connected_region_tensors":sorted(regions),"FP32_teacher_calls":0,
            "candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Independent broad recovery from F06; not a continuation of running F10.",
                           "Authored templated data, held-out prompts and different lexical items but not an independent benchmark.",
                           "Single training arm; no causal attribution to particular objective. Original older NRA3 and legacy donor weights remain frozen.",
                           "Legacy files included but not active. Tokenizer/code, J7, CPU and functional plateau not certified."]}
    (out/"F11_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F11_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("f01","f06","f08","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f06,a.f08,a.out)
