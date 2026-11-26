"""G16: actual joint Russian language recovery on 133.701M G05.
Balanced elementary copying, context-grounded answers, and complete
sentences. Two examples per Adam update, full native answer CE + EOS,
opposite first-target contrast when verified distinct. No external teacher,
answer rules at runtime, or training on G14. Physical reload after epochs.
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import (
    payload,extra_pack,digest,open_bundle,native,RATE,k_for)
from research.calibri2.fusion_growth.g02_attention_discrete import indexed_banks,forward
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore,region_split,rate
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g14_russian_ladder import suite as ladder
from research.calibri2.fusion_growth.f21_native_cpu_integration import prompt_features
from research.calibri2.fusion_growth.g13_open_135m_benchmark import norm
from v176.operators import clear

SEED=20270108
FAMILIES=("copy","context","sentence")
EPOCHS=2
TRAIN_N=144
DEV_N=48

# Separate lexical pools, no G14 exact prompt reuse. Original authored text.
WORDS={"TRAIN":["арбуз","берег","ветка","гриб","дуб","ёж","жук","заяц","игла","камень","лодка","море","нос","облако","пень","ручей",
                "сова","туман","улей","фонарь","хлеб","цветок","чашка","шар"],
       "DEV":["белка","гора","зонт","книга","лист","мост","река","стол"]}
COLORS={"TRAIN":[("белый","белая"),("чёрный","чёрная"),("зелёный","зелёная"),("жёлтый","жёлтая"),
                 ("синий","синяя"),("красный","красная"),("серый","серая"),("коричневый","коричневая")],
        "DEV":[("розовый","розовая"),("фиолетовый","фиолетовая"),("оранжевый","оранжевая"),("голубой","голубая")]}
NAMES={"TRAIN":["Аня","Борис","Вера","Глеб","Дина","Егор","Женя","Зоя"],"DEV":["Лада","Максим","Ника","Роман"]}
OBJECTS={"TRAIN":[("шапка","шапку"),("ложка","ложку"),("сумка","сумку"),("тетрадь","тетрадь"),("ваза","вазу"),("лампа","лампу")],
         "DEV":[("куртка","куртку"),("тарелка","тарелку"),("коробка","коробку"),("ручка","ручку")]}
SENTENCES={
"TRAIN":[
("Что делает птица в небе? Ответь предложением.","Птица летит в небе."),
("Что делает рыба в реке?","Рыба плавает в реке."),
("Где растёт трава?","Трава растёт на земле."),
("Какого цвета листья летом? Ответь предложением.","Летом листья зелёные."),
("Что происходит с водой при замерзании?","Вода превращается в лёд."),
("Для чего нужны часы?","Часы показывают время."),
("Для чего используют ножницы?","Ножницами режут бумагу."),
("Зачем люди открывают окно?","Окно открывают для свежего воздуха."),
("Что человек делает с книгой?","Человек читает книгу."),
("Зачем нужен зонтик во время дождя?","Зонтик защищает от дождя."),
("Чем питается корова?","Корова ест траву."),
("Как передвигается лошадь?","Лошадь ходит и бегает."),
("Для чего нужен холодильник?","Холодильник сохраняет продукты холодными."),
("Для чего нужна подушка?","На подушке удобно спать."),
("Почему светит лампа?","Лампа получает электрическую энергию."),
("Что случается с листом осенью?","Осенью лист желтеет и падает."),
("Зачем нужно чистить зубы?","Чистка зубов помогает сохранить их здоровье."),
("Для чего нужен карандаш?","Карандашом пишут и рисуют."),
("Что слышно во время грозы?","Во время грозы слышен гром."),
("Что нужно сделать, чтобы согреть суп?","Суп нужно нагреть."),
("Как вежливо попросить стакан воды?","Дайте, пожалуйста, стакан воды."),
("Как поблагодарить за помощь?","Большое спасибо за помощь."),
("Что сказать при встрече утром?","Доброе утро!"),
("Что пожелать человеку перед сном?","Спокойной ночи!"),
("Скажи, что тебе нравится читать.","Мне нравится читать книги."),
("Откажись вежливо от чашки кофе.","Нет, спасибо, я не хочу кофе."),
("Извинись за опоздание.","Извините, пожалуйста, за опоздание."),
("Попроси передать соль.","Передайте, пожалуйста, соль."),
("Сообщи, что автобус уже приехал.","Автобус уже приехал."),
("Сообщи, что сегодня холодно.","Сегодня на улице холодно."),
("Составь предложение о собачке и мяче.","Собачка играет с мячом."),
("Составь предложение о девочке и цветах.","Девочка собирает цветы.")],
"DEV":[
("Зачем человеку нужен дождевик?","Дождевик защищает от дождя."),
("Что делают весной птицы в гнезде?","Птицы откладывают яйца в гнезде."),
("Какую пользу приносит мыло?","Мыло помогает очищать кожу."),
("Для чего используют полотенце?","Полотенцем вытирают руки."),
("Что происходит с кипящей водой?","Вода превращается в пар."),
("Зачем зимой отапливают дом?","Отопление согревает дом."),
("Попроси вежливо закрыть дверь.","Закройте, пожалуйста, дверь."),
("Поблагодари человека за подарок.","Большое спасибо за подарок."),
("Сообщи, что поезд задерживается.","Поезд задерживается."),
("Пожелай другу удачного дня.","Желаю тебе удачного дня!"),
("Составь предложение о мальчике и велосипеде.","Мальчик едет на велосипеде."),
("Составь предложение о бабушке и пироге.","Бабушка печёт пирог.")]}

def data(tok,split):
    tr=split=="TRAIN";result={k:[] for k in FAMILIES}
    def add(k,q,a):
        p,g=native(tok,q,a);assert tok.decode(g[:-1],skip_special_tokens=True).strip()==a
        assert len(p)+32<160
        result[k].append({"split":split,"family":k,"q":q,"answer":a,"p":p,"g":g,
                          "features":prompt_features(tok,q)})
    ws=WORDS[split];ns=NAMES[split];cs=COLORS[split];os=OBJECTS[split]
    for i,w in enumerate(ws):
        add("copy",f"Запиши слово «{w}» без изменений." if tr else f"Напечатай в точности слово: {w}.",w)
        add("copy",f"Повтори два слова: {w} {ws[(i+1)%len(ws)]}." if tr else f"Перепиши без изменений: {w} {ws[(i+1)%len(ws)]}.",
            w+" "+ws[(i+1)%len(ws)])
    n=48 if tr else 16
    for i in range(n//2):
        name=ns[i%len(ns)];obj,acc=os[(i%len(ns)+i//len(ns))%len(os)];color=cs[(i+i//len(ns))%len(cs)][1]
        add("context",f"{name} купил(а) предмет: {obj}. Что купил(а) {name}?" if tr else f"{name} выбрал(а) предмет: {obj}. Какой предмет выбрал(а) {name}?",acc)
        add("context",f"У персонажа по имени {name} есть {color} {obj}. Какого цвета {obj}?" if tr else f"У {name} предмет {obj} имеет цвет: {color}. Назови цвет.",color)
    # Split-specific explicit sentence targets plus context-dependent full
    # sentences; enough complete sentences rather than only digits/yes/no.
    for q,a in SENTENCES[split]:add("sentence",q,a)
    for i in range(n-len(SENTENCES[split])):
        name=ns[i%len(ns)];obj,acc=os[i%len(os)];place=("на столе","в шкафу","на полке","в сумке")[i%4] if tr else ("у окна","на стуле","в ящике","у двери")[i%4]
        add("sentence",f"{name} оставил(а) {acc} {place}. Ответь полным предложением: где находится {obj}?" if tr else f"В комнате {obj} находится {place}. Скажи полным предложением, где она находится.",
            obj[0].upper()+obj[1:]+" находится "+place+".")
    assert all(len(v)==n for v in result.values())
    assert len({r["q"] for v in result.values() for r in v})==3*n
    return result

def audit(st,tok,rows,tag):
    records=[]
    for r in rows:
        own=[];first_rank=None;nll=[];cap=min(32,max(12,len(r["g"])+5))
        for _ in range(cap):
            z=forward(st,r["p"]+own,r["features"]).detach()
            if not own:first_rank=1+int((z[0]>z[0,r["g"][0]]).sum())
            own.append(int(z.argmax()))
            if own[-1]==2:break
        text=tok.decode(own,skip_special_tokens=True).strip()
        for j,t in enumerate(r["g"]):
            z=forward(st,r["p"]+r["g"][:j],r["features"]).detach()
            nll.append(float(F.cross_entropy(z,torch.tensor([t]))))
        rec={"q":r["q"],"gold":r["answer"],"family":r["family"],"answer":text,"ids":own,
             "exact":norm(text)==norm(r["answer"]),"eos":own[-1]==2,"first_rank":first_rank,
             "NLL":float(np.mean(nll))}
        records.append(rec)
        print("G16_ROW",tag,json.dumps(rec,ensure_ascii=False),flush=True)
    result={}
    for k in FAMILIES:
        rr=[r for r in records if r["family"]==k]
        result[k]={"n":len(rr),"exact":sum(r["exact"] for r in rr),
                   "exact_EOS":sum(r["exact"] and r["eos"] for r in rr),"EOS":sum(r["eos"] for r in rr),
                   "unique_answers":len({r["answer"] for r in rr}),
                   "dominant_answer":Counter(r["answer"] for r in rr).most_common(1)[0],
                   "median_target_rank":statistics.median(r["first_rank"] for r in rr),
                   "mean_NLL":float(np.mean([r["NLL"] for r in rr]))}
    print("G16_AUDIT",tag,json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
    return {"summary":result,"records":records}

def train_epoch(st,ext,train,epoch,rng):
    _,sources,_,full,_,child,region=st
    core=[s.centroids for s in sources.values()]+[s.centroids for s in ext.values()]
    head=list(child.language_head.parameters())
    headids={id(p) for p in head};other=[p for p in region if id(p) not in headids]
    assert headids <= {id(p) for p in region}
    params=core+other+head+[full.gates]
    assert len(params)==len({id(p) for p in params})
    opt=torch.optim.Adam([{"params":core,"lr":.00015},{"params":other,"lr":.0003},
                          {"params":head,"lr":.0012},{"params":[full.gates],"lr":.0001}])
    # Pair within a family and rotate families every step: prevent long
    # runs of one response type. Equal example weighting, EOS included.
    schedules={k:[train[k][int(i)] for i in rng.permutation(len(train[k]))] for k in FAMILIES}
    steps=0;calls=0;contrast=0;curve=[];seen=set();gt=set();region_seen=set()
    for j in range(0,48,2):
        for k in FAMILIES:
            pair=schedules[k][j:j+2];assert len(pair)==2
            opt.zero_grad(set_to_none=True);value=0.
            for i,r in enumerate(pair):
                assert r["split"]=="TRAIN"
                g=r["g"];wrong=pair[1-i]["g"][0]
                for pos,t in enumerate(g):
                    z=forward(st,r["p"]+g[:pos],r["features"])
                    # Content mean plus independently weighted EOS, so
                    # long answers are not trained less on termination.
                    w=.7 if t==2 else 1./max(1,len(g)-1)
                    loss=w*F.cross_entropy(z,torch.tensor([t]))
                    if pos==0 and t!=wrong:
                        loss=loss+.25*F.softplus(.5-(z[0,t]-z[0,wrong]))
                        contrast+=1
                    loss=loss/2
                    assert torch.isfinite(loss);loss.backward()
                    value+=float(loss.detach());calls+=1
            for name,s in list(sources.items())+list(ext.items()):
                if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(name)
            if full.gates.grad is not None:gt.update(i for i,v in enumerate(full.gates.grad) if float(v.abs())>0)
            region_seen.update(i for i,p in enumerate(region) if p.grad is not None and float(p.grad.abs().sum())>0)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for name,s in sources.items():s.centroids[k_for(RATE,name):].zero_()
                for s in ext.values():s.centroids[8:].zero_()
            clear(sources);curve.append(value);steps+=1
            if steps%8==0:print("G16_PROGRESS",epoch,steps,calls,value,len(seen),len(gt),flush=True)
    assert steps==72 and len(seen)==409 and len(gt)==12
    return {"epoch":epoch,"steps":steps,"native_training_calls":calls,"contrast_terms":contrast,
            "connected_codebooks":len(seen),"connected_GT":len(gt),"connected_region_tensors":len(region_seen),
            "first8_objective":float(np.mean(curve[:8])),"last8_objective":float(np.mean(curve[-8:]))}

def run(g05,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2);rng=np.random.default_rng(SEED)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    ar,sd,legacy=open_bundle(blobs[0])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tb=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    tr=data(tok,"TRAIN");dv=data(tok,"DEV")
    train=[r for rows in tr.values() for r in rows];dev=[r for rows in dv.values() for r in rows]
    assert len(train)==TRAIN_N and len(dev)==DEV_N
    assert not {r["q"] for r in train}&({r["q"] for r in dev}|{r["q"] for r in ladder()})
    out.mkdir(parents=True,exist_ok=False)
    (out/"G16_MANIFEST.json").write_text(json.dumps({s:[{k:r[k] for k in ("family","q","answer","g")} for r in rows] for s,rows in (("TRAIN",train),("DEV",dev))},ensure_ascii=False,indent=2))
    def reload(vals):return restore(vals[:6],vals[6],vals[7])
    st,*rest=reload(blobs);before=audit(st,tok,dev,"BEFORE");del st,rest
    epochs=[];vals=blobs
    for epoch in range(1,EPOCHS+1):
        st,second,added,fourth,reg,old,fifth,sixth,reg2=reload(vals)
        ext=indexed_banks(second,added)|{"fourth/"+k:v for k,v in fourth.items()}|{"fifth/"+k:v for k,v in fifth.items()}|{"sixth/"+k:v for k,v in sixth.items()}
        info=train_epoch(st,ext,tr,epoch,rng)
        _,packet,_,_=payload(ar,sd,st,legacy);rb,rb2=region_split(st,reg,old,reg2)
        vals=(packet,extra_pack(second),extra_pack(added),extra_pack(fourth),rb,extra_pack(fifth),extra_pack(sixth),rb2)
        storage=rate(vals[:6],vals[6],vals[7],G05,tb)
        assert storage["weights_plus_tokenizer_bpp"]<=.7
        for k,v in zip(PIN,vals):(out/(f"EPOCH{epoch}_"+k)).write_bytes(v)
        del st,second,added,fourth,reg,old,fifth,sixth,reg2,ext
        st,*rest=reload(vals);after=audit(st,tok,dev,f"EPOCH{epoch}");del st,rest
        rec={"training":info,"after_physical_reload":after,"storage":storage,
             "sha256":dict(zip(PIN,[digest(v) for v in vals]))}
        epochs.append(rec)
        print("G16_EPOCH",epoch,json.dumps({**rec,"after_physical_reload":after["summary"]},ensure_ascii=False,sort_keys=True),flush=True)
        (out/"G16_REPORT.json").write_text(json.dumps({"stage":"G16_RUSSIAN_CONNECTED_LANGUAGE","geometry":G05,
           "TRAIN_unique":TRAIN_N,"DEV_unique":DEV_N,"before":before,"epochs":epochs,"candidate_accepted":False,
           "teacher_calls":0,"limitations":["Single intervention, not compute-matched control; cannot attribute change to one objective.",
                "Authored small corpus with lexical split and explicit sentence targets; exact match is conservative for open sentences.",
                "No G14 training, no protected replay or runtime certification. Two fixed epochs, physical Q8 reload between epochs; no DEV-based selection.",
                "Head LR, pair contrast and EOS weight are experimental decisions. No permanent router or rule-based answer generator."]},ensure_ascii=False,indent=2))
    print("G16_ASSESS",json.dumps({"epochs_completed":len(epochs),"candidate_accepted":False}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
