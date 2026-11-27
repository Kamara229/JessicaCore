"""G31a Russian language foundation pilot, authored compositional data only.
No model/teacher inference, no updates, native tokenizer audit.
Lexical entities and template order are partitioned before generation.
"""
import argparse,json,hashlib,collections
from pathlib import Path
from transformers import AutoTokenizer

TRAIN_NAMES=("Анна","Марина","Вера","Ольга","Ирина","Елена","Светлана","Татьяна","Наталья","Дарья","Полина","Софья")
DEV_NAMES=("Алёна","Кира","Валерия","Лидия")
# nominative, accusative, prepositional location, instrumental
TRAIN_OBJECTS=(
("книга","книгу","книге","книгой"),("тетрадь","тетрадь","тетради","тетрадью"),
("чашка","чашку","чашке","чашкой"),("письмо","письмо","письме","письмом"),
("ключ","ключ","ключе","ключом"),("платок","платок","платке","платком"),
("карта","карту","карте","картой"),("газета","газету","газете","газетой"),
("ручка","ручку","ручке","ручкой"),("блокнот","блокнот","блокноте","блокнотом"),
("зонт","зонт","зонте","зонтом"),("коробка","коробку","коробке","коробкой"),
("сумка","сумку","сумке","сумкой"),("альбом","альбом","альбоме","альбомом"),
("пакет","пакет","пакете","пакетом"),("линейка","линейку","линейке","линейкой"))
DEV_OBJECTS=(("открытка","открытку","открытке","открыткой"),("карандаш","карандаш","карандаше","карандашом"),("конверт","конверт","конверте","конвертом"),("папка","папку","папке","папкой"))
# Destination accusative phrase, static location.
TRAIN_PLACES=(("на стол","на столе"),("на полку","на полке"),("в шкаф","в шкафу"),("в ящик","в ящике"),("на скамью","на скамье"),("в корзину","в корзине"),("на стул","на стуле"),("в комод","в комоде"))
DEV_PLACES=(("на подоконник","на подоконнике"),("в рюкзак","в рюкзаке"),("на тумбу","на тумбе"))
# Natural fixed sentences, not mechanically derived inflections.
TRAIN_SCENES=(
("Дождь закончился","На небе показалось солнце","солнце"),("Ветер стих","В саду стало тихо","тихо"),
("Вода закипела","Анна заварила чай","чай"),("Молоко остыло","Марина убрала его в холодильник","холодильник"),
("Погас свет","Вера зажгла свечу","свечу"),("Автобус остановился","Ольга вышла на улицу","улицу"),
("Снег растаял","На дорожке появились лужи","лужи"),("Дверь открылась","Ирина вошла в комнату","комнату"),
("Будильник прозвенел","Елена встала с кровати","кровати"),("Урок закончился","Ученики вышли в коридор","коридор"),
("Хлеб испёкся","Светлана вынула его из печи","печи"),("Наступил вечер","Татьяна включила лампу","лампу"),
("Телефон зазвонил","Наталья ответила на звонок","звонок"),("Пришла весна","На деревьях распустились листья","листья"),
("Собака проголодалась","Дарья наполнила её миску","миску"),("Поезд прибыл","Полина встретила подругу","подругу"),
("Окно запотело","Софья проветрила комнату","комнату"),("Начался дождь","Прохожие раскрыли зонты","зонты"),
("Река замёрзла","Берега покрыл снег","снег"),("Мастер закончил работу","Он убрал инструменты","инструменты"))
DEV_SCENES=(
("Солнце село","На улице стемнело","стемнело"),("Часы пробили полдень","Кира пошла обедать","обедать"),
("Послышался стук","Алёна открыла дверь","дверь"),("Кошка уснула","Валерия укрыла её пледом","пледом"),
("Книга упала","Лидия подняла её с пола","пола"),("Зажёгся зелёный свет","Пешеходы перешли дорогу","дорогу"))

def generate(split):
    tr=split=="TRAIN"
    names=TRAIN_NAMES if tr else DEV_NAMES
    objects=TRAIN_OBJECTS if tr else DEV_OBJECTS
    places=TRAIN_PLACES if tr else DEV_PLACES
    scenes=TRAIN_SCENES if tr else DEV_SCENES
    rows=[]
    def add(family,q,a,group):
        rows.append({"split":split,"family":family,"q":q,"answer":a,"group":group,"origin":"authored_compositional_pilot"})
    # Rotate names and places so the data is not a full repetitive cross product.
    for i,(nom,acc,prep,ins) in enumerate(objects):
        for j,n in enumerate(names):
            dest,loc=places[(i+j)%len(places)]
            grp=f"{n}/{nom}"
            add("sentence",f"Составь предложение: {n}, положила, {acc}, {dest}.",f"{n} положила {acc} {dest}.",grp)
            add("context",f"{n} положила {acc} {dest}. Где теперь находится этот предмет? Ответь полным предложением.",f"Предмет находится {loc}.",grp)
            add("grammar",f"Дополни фразу, изменив слово в скобках: {n} воспользовалась ({nom}).",f"{n} воспользовалась {ins}.",grp)
            add("copy",f"Повтори предложение без изменений: {n} нашла {acc} и положила {dest}.",f"{n} нашла {acc} и положила {dest}.",grp)
            add("question",f"Ответь полным предложением. Кто положил {acc} {dest}, если это сделала {n}?",f"{n} положила {acc} {dest}.",grp)
            add("word",f"Напиши форму творительного падежа слова «{nom}» без других слов.",ins,f"word/{nom}")
    # Deduplicate identical word instructions from rotated names.
    seen=set();result=[]
    for r in rows:
        if r["q"] not in seen:result.append(r);seen.add(r["q"])
    rows=result
    for a,b,last in scenes:
        grp=a
        add("two_sentences",f"Запиши два предложения: «{a}» и «{b}».",f"{a}. {b}.",grp)
        add("punctuation",f"Расставь знаки препинания и заглавные буквы: {a.lower()} {b.lower()}",f"{a}. {b}.",grp)
        add("copy",f"Повтори без изменений: {a}. {b}.",f"{a}. {b}.",grp)
    return rows

def audit(tok,rows):
    out=[];bad=[]
    for r in rows:
        chat=tok.apply_chat_template([{"role":"user","content":r["q"]}],tokenize=False,add_generation_prompt=True)
        p=tok.encode(chat,add_special_tokens=False)
        ids=tok.encode(chat+r["answer"],add_special_tokens=False)
        g=ids[len(p):]+[2]
        if ids[:len(p)]!=p or len(p)+len(g)>=160 or len(g)<2 or 2 in g[:-1] or tok.decode(g[:-1],skip_special_tokens=True).strip()!=r["answer"]:
            bad.append({"q":r["q"],"family":r["family"],"length":len(p)+len(g)});continue
        out.append({**r,"p":p,"g":g})
    return out,bad

def stats(rows):
    tokens=[t for r in rows for t in r["g"]]
    return {"rows":len(rows),"target_tokens":len(tokens),"unique_target_tokens":len(set(tokens)),"families":dict(collections.Counter(r["family"] for r in rows)),"mean_answer_tokens":len(tokens)/len(rows),"mean_answer_words":sum(len(r["answer"].split()) for r in rows)/len(rows),"first_token_counts":collections.Counter(r["g"][0] for r in rows).most_common(8)}

def run(f08,out):
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    tr0,dv0=generate("TRAIN"),generate("DEV")
    assert not {r["q"] for r in tr0}&{r["q"] for r in dv0}
    assert not {r["group"] for r in tr0}&{r["group"] for r in dv0}
    tr,badtr=audit(tok,tr0);dv,baddv=audit(tok,dv0)
    assert len(tr)>=500 and len(dv)>=30
    assert len({r["q"] for r in tr})==len(tr) and len({r["q"] for r in dv})==len(dv)
    out.mkdir(parents=True,exist_ok=False)
    digests={}
    for name,rows in (("TRAIN",tr),("DEV",dv)):
        data="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode()
        (out/(name+".jsonl")).write_bytes(data);digests[name]=hashlib.sha256(data).hexdigest()
    report={"TRAIN":stats(tr),"DEV":stats(dv),"rejected_train":badtr,"rejected_dev":baddv,"sha256":digests,"limitations":["Small authored combinatorial pilot; shared task templates across splits, not independent sealed evaluation.","No broad natural-language corpus yet; narrow feminine-name and everyday-object domain.","No model training or quality claim; future curriculum must include diverse authentic licensed text and protected replay."]}
    (out/"G31A_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G31A_REPORT",json.dumps({k:v for k,v in report.items() if not k.startswith("rejected")},ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--f08",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();run(a.f08,a.out)
