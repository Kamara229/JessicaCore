"""G14: Russian-only capability ladder, read-only, actual G05 vs open 135M.
Eight independently interpretable levels x 16 authored prompts.
No teaching, output selection, or updates. Probe literal generation,
context, knowledge, compositional rules and longer multi-step answers.
Keep raw outputs, exact/acceptable answer scores, first-token likelihood
and output collapse statistics. Difficulty is an experimental ordering,
not a calibrated IQ scale.
"""
import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import open_bundle,digest,native
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g02_attention_discrete import forward
from research.calibri2.fusion_growth.f21_native_cpu_integration import prompt_features,strict_chat_source_ids
from research.calibri2.fusion_growth.g13_open_135m_benchmark import REPO,REV,norm

LEVELS=[
("L0","Буквальное воспроизведение",[
("Напиши цифру 0.","0"),("Напиши цифру 3.","3"),("Напиши цифру 8.","8"),("Напиши букву а.","а"),
("Напиши букву м.","м"),("Напиши букву я.","я"),("Повтори: да.","да"),("Повтори: нет.","нет"),
("Повтори: дом.","дом"),("Повтори: кот.","кот"),("Повтори: лес.","лес"),("Повтори: вода.","вода"),
("Повтори: солнце.","солнце"),("Напиши число 27.","27"),("Повтори: красный шар.","красный шар"),("Повтори: тихий сад.","тихий сад")]),
("L1","Базовые понятия и простые ассоциации",[
("Как говорит кошка?","мяу"),("Какого цвета обычный снег?","белый|белого|белого цвета"),("Что больше: 2 или 3?","3"),("Что меньше: 4 или 6?","4"),
("Сколько будет 1 + 1?","2"),("Сколько будет 2 − 1?","1"),("Какое число идёт сразу после 4?","5"),("Какое число идёт перед 8?","7"),
("У человека обычно сколько глаз?","2|два"),("Как называется детёныш кошки?","котёнок"),("Лимон обычно кислый или сладкий?","кислый"),("Ночь тёмная или светлая?","тёмная"),
("Какое из слов обозначает животное: кот или стол?","кот"),("Какое из слов обозначает растение: сосна или камень?","сосна"),("Какое слово противоположно слову «высокий»?","низкий"),("Зимой обычно холодно или жарко?","холодно")]),
("L2","Извлечение явного факта из вопроса",[
("У Лены красная кружка. Какого цвета кружка?","красная|красного цвета"),("У Олега синяя кружка. Какого цвета кружка?","синяя|синего цвета"),
("Ключ лежит в сумке. Где ключ?","в сумке|сумка"),("Ключ лежит на полке. Где ключ?","на полке|полка"),
("У Тани две груши. Сколько груш у Тани?","2|две|два"),("У Ромы пять груш. Сколько груш у Ромы?","5|пять"),
("Мила читает книгу. Что читает Мила?","книгу|книга"),("Мила читает журнал. Что читает Мила?","журнал"),
("Серый кот спит. Кто спит?","кот|серый кот"),("Белая собака спит. Кто спит?","собака|белая собака"),
("Брат приехал в понедельник. В какой день он приехал?","понедельник|в понедельник"),("Брат приехал во вторник. В какой день он приехал?","вторник|во вторник"),
("Мяч принадлежит Ире. Кто владелец мяча?","Ира"),("Мяч принадлежит Ване. Кто владелец мяча?","Ваня"),
("На двери написано 51. Какое число на двери?","51"),("На двери написано 63. Какое число на двери?","63")]),
("L3","Простые операции, отрицания, инструкции",[
("Вычисли 3 + 4.","7"),("Вычисли 3 + 5.","8"),("Вычисли 9 − 4.","5"),("Вычисли 9 − 2.","7"),
("У Пети есть ручка. Есть ли у Пети ручка?","да"),("У Пети нет ручки. Есть ли у Пети ручка?","нет"),
("В коробке нет мяча. Есть ли в коробке мяч?","нет"),("В коробке есть мяч. Есть ли в коробке мяч?","да"),
("Назови первое слово: река гора поле.","река"),("Назови последнее слово: река гора поле.","поле"),
("Назови второе слово: окно дверь стена.","дверь"),("Назови первое слово: окно дверь стена.","окно"),
("Напиши число 7, затем число 2 без пробела.","72"),("Напиши число 2, затем число 7 без пробела.","27"),
("Что больше: 12 или 19?","19"),("Что меньше: 12 или 19?","12")]),
("L4","Факты, грамматика и отношения",[
("Столица России?","Москва"),("Сколько месяцев в году?","12|двенадцать"),("Сколько часов в сутках?","24|двадцать четыре"),
("Как называется спутник Земли?","Луна"),("Какая часть речи у слова «бежать»?","глагол"),("Напиши множественное число слова «стол».","столы"),
("Напиши единственное число слова «книги».","книга"),("Скажи антоним к слову «день».","ночь"),
("Игорь передал чашку Лизе. У кого теперь чашка?","Лиза|у Лизы"),("Лиза передала чашку Игорю. У кого теперь чашка?","Игорь|у Игоря"),
("Оля стоит левее Саши. Кто справа?","Саша"),("Оля стоит правее Саши. Кто справа?","Оля"),
("Вычисли 6 × 4.","24"),("Вычисли 28 ÷ 4.","7"),("Продолжи ряд: 1, 3, 5, 7, ...","9"),("Продолжи ряд: 3, 6, 9, 12, ...","15")]),
("L5","Композиция двух отношений и правил",[
("Ира старше Лены, Лена старше Оли. Кто старше всех?","Ира"),("Ира младше Лены, Лена младше Оли. Кто старше всех?","Оля"),
("Миша дал ключ Нине, Нина дала ключ Паше. У кого ключ?","Паша|у Паши"),("Миша дал ключ Паше, Паша дал ключ Нине. У кого ключ?","Нина|у Нины"),
("Все вороны — птицы. Кеша — ворона. Кеша птица?","да"),("Ни одна ворона не рыба. Кеша — ворона. Кеша рыба?","нет"),
("У Коли 5 яблок. Он купил 3 и съел 2. Сколько осталось?","6|шесть"),("У Коли 5 яблок. Он купил 2 и съел 3. Сколько осталось?","4|четыре"),
("Сначала умножь 3 на 4, затем прибавь 2.","14"),("Сначала прибавь 4 к 3, затем умножь результат на 2.","14"),
("Вычисли 2 × (4 + 3).","14"),("Вычисли 2 × 4 + 3.","11"),
("Назови следующий день после послезавтра, если сегодня понедельник.","четверг"),("Назови день перед вчерашним, если сегодня пятница.","среда"),
("Из слов кот, дуб, пёс выбери только животных через запятую.","кот, пёс|кот, пес"),("Из слов дуб, кот, ель выбери только деревья через запятую.","дуб, ель")]),
("L6","Многошаговое рассуждение и новые правила",[
("Вычисли (18 − 6) ÷ 3 + 2.","6"),("Вычисли 4 × (9 − 5) − 3.","13"),
("Если А означает 2, Б означает 5, чему равно А + Б?","7"),("Если А означает 2, Б означает 5, чему равно А × Б?","10"),
("Правило: красный означает 1, синий означает 2. Запиши код для «синий красный синий» без пробелов.","212"),
("Правило: красный означает 2, синий означает 1. Запиши код для «синий красный синий» без пробелов.","121"),
("Есть три коробки: А левее Б, Б левее В. Какая коробка посередине?","Б"),
("Есть три коробки: В левее А, А левее Б. Какая коробка посередине?","А"),
("У Ани 12 монет, у Веры вдвое меньше, у Оли на 3 больше, чем у Веры. Сколько у Оли?","9"),
("В магазин привезли 3 коробки по 8 яблок. Продали 7 яблок. Сколько осталось?","17"),
("Все ламы травоядные. Ни одно травоядное не хищник. Может ли лама быть хищником согласно этим условиям?","нет"),
("Все лумы — зеты. Все зеты — рапы. Лум является рапом?","да"),
("Отсортируй числа 9, 2, 6 по возрастанию через запятую.","2, 6, 9"),
("Отсортируй числа 9, 2, 6 по убыванию через запятую.","9, 6, 2"),
("Сколько составляет 25 процентов от 80?","20"),("Реши уравнение: 3x + 2 = 17. Чему равен x?","5")]),
("L7","Связная генерация и объяснение",[
("Объясни одним предложением, почему лёд тает в тёплой комнате.",""),
("Объясни одним предложением, зачем зимой надевают тёплую одежду.",""),
("Почему ночью на улице обычно темнее, чем днём?",""),
("Расскажи в двух предложениях, чем кошка отличается от собаки.",""),
("Объясни, чем круг отличается от квадрата.",""),
("Объясни, почему 4 + 3 = 7.",""),
("Объясни, как проверить результат вычитания сложением.",""),
("Придумай одно предложение со словами «снег» и «дерево».",""),
("Напиши короткое поздравление с днём рождения.",""),
("Напиши два последовательных действия для приготовления чая.",""),
("Составь короткий вежливый ответ на приглашение, если не можешь прийти.",""),
("Объясни, зачем нужно мыть руки перед едой.",""),
("Расскажи короткую историю о коте, который нашёл потерянный мяч.",""),
("Напиши одним предложением, что делать, если потерял ключи дома.",""),
("Объясни, почему после дождя на земле появляются лужи.",""),
("В двух предложениях расскажи, как беречь воду дома.","")])]

# Human review rubrics for open answers, not substring-based correctness.
RUBRICS=[
"Связь тепла и таяния льда","Сохранение тепла тела","Отсутствие прямого солнечного света",
"Два связанных предложения, корректное различие","Круг без углов, квадрат с четырьмя сторонами/углами",
"Корректное объяснение сложения","Обратная операция: разность плюс вычитаемое",
"Осмысленное предложение с обоими словами","Связное поздравление","Два корректных последовательных действия",
"Вежливый отказ","Гигиена и удаление загрязнений/микробов","Связная история с котом и мячом",
"Осмысленное безопасное действие","Скопление дождевой воды","Два связанных предложения о разумном расходовании воды"]
CAPS={"L0":12,"L1":16,"L2":20,"L3":20,"L4":24,"L5":32,"L6":40,"L7":80}

def suite():
    rows=[]
    for lid,title,items in LEVELS:
        assert len(items)==16
        for i,(q,a) in enumerate(items):
            rows.append({"id":f"{lid}/{i+1:02}","level":lid,"title":title,"q":q+(" Ответь только кратким ответом." if lid!="L7" else ""),
                         "accepted":a.split("|") if a else [],"rubric":RUBRICS[i] if lid=="L7" else None})
    assert len(rows)==128 and len({r["q"] for r in rows})==128
    return rows

def normalize(x):
    return re.sub(r"\s*,\s*",",",norm(x))

def score(row,text):
    if not row["accepted"]:return None
    return normalize(text) in {normalize(a) for a in row["accepted"]}

def calibri(st,tok,row):
    p=strict_chat_source_ids(tok,row["q"]);feat=prompt_features(tok,row["q"])
    cap=CAPS[row["level"]];ids=[];first=None;t=time.perf_counter()
    for _ in range(cap):
        z=forward(st,p+ids,feat).detach()
        if first is None:first=z.clone()
        ids.append(int(z.argmax()))
        if ids[-1]==2:break
    elapsed=time.perf_counter()-t
    text=tok.decode(ids,skip_special_tokens=True).strip()
    result={"text":text,"ids":ids,"seconds":elapsed,"eos":ids[-1]==2,
            "accepted_exact":score(row,text),"first_token_id":ids[0]}
    if row["accepted"]:
        candidates=[]
        for a in row["accepted"]:
            try:pp,g=native(tok,row["q"],a)
            except AssertionError:continue
            assert p==pp
            gold=g[0];r=1+int((first[0]>first[0,gold]).sum())
            candidates.append({"answer":a,"first_token_rank":r,"first_token_probability":float(first.softmax(-1)[0,gold])})
        result["target_first_token"]=candidates
    return result

def smol(model,tok,row):
    p=tok.apply_chat_template([{"role":"user","content":row["q"]}],tokenize=True,add_generation_prompt=True,return_tensors="pt")
    t=time.perf_counter()
    with torch.inference_mode():
        seq=model.generate(p,attention_mask=torch.ones_like(p),max_new_tokens=CAPS[row["level"]],do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    seconds=time.perf_counter()-t;ids=seq[0,p.shape[-1]:].tolist()
    eos=model.generation_config.eos_token_id;stop=set(eos if isinstance(eos,list) else [eos])
    text=tok.decode(ids,skip_special_tokens=True).strip()
    return {"text":text,"ids":ids,"seconds":seconds,"eos":ids[-1] in stop,
            "accepted_exact":score(row,text),"first_token_id":ids[0]}

def summarize(records,mode):
    result={}
    for lid,title,_ in LEVELS:
        rr=[r["answers"][mode] for r in records if r["level"]==lid]
        cnt=Counter(normalize(x["text"]) for x in rr)
        first=Counter(x["first_token_id"] for x in rr)
        result[lid]={"title":title,"n":len(rr),"accepted_exact":sum(x["accepted_exact"] is True for x in rr) if lid!="L7" else None,
                     "EOS":sum(x["eos"] for x in rr),"empty":sum(not x["text"] for x in rr),
                     "unique_answers":len(cnt),"most_common_answers":cnt.most_common(5),
                     "dominant_first_token_count":first.most_common(1)[0][1],
                     "tokens":sum(len(x["ids"]) for x in rr),
                     "seconds":sum(x["seconds"] for x in rr)}
        if mode=="CALIBRI_G05" and lid!="L7":
            ranks=[min(z["first_token_rank"] for z in x["target_first_token"]) for x in rr if x["target_first_token"]]
            result[lid]["target_first_token_median_rank"]=statistics.median(ranks) if ranks else None
            result[lid]["target_first_token_top10"]=sum(r<=10 for r in ranks)
    return result

def run(g05,f08,out):
    torch.manual_seed(20270106);torch.set_num_threads(2)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    ctok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    root=Path(snapshot_download(REPO,revision=REV,allow_patterns=["config.json","generation_config.json","model.safetensors","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.json","merges.txt"]))
    stok=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").eval()
    st,*rest=restore(blobs[:6],blobs[6],blobs[7])
    out.mkdir(parents=True,exist_ok=False)
    rows=suite();records=[]
    (out/"G14_SUITE.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    # No timing claims: baseline native forward for rank diagnostics, not
    # packed fast path. No long-run cache retained; evaluation only.
    for i,row in enumerate(rows):
        a=calibri(st,ctok,row);b=smol(model,stok,row)
        rec={**row,"answers":{"CALIBRI_G05":a,"SMOL_135M":b}}
        records.append(rec)
        print("G14_ROW",json.dumps({"id":row["id"],"q":row["q"],"accepted":row["accepted"],
             "CALIBRI":a["text"],"SMOL":b["text"],"C_EOS":a["eos"],"S_EOS":b["eos"]},ensure_ascii=False),flush=True)
        # Keep incremental raw records in case a later question fails.
        with (out/"G14_RAW.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
        if (i+1)%16==0:print("G14_LEVEL_PROGRESS",row["level"],i+1,flush=True)
    result={m:summarize(records,m) for m in ("CALIBRI_G05","SMOL_135M")}
    for m,v in result.items():print("G14_ARM",m,json.dumps(v,ensure_ascii=False,sort_keys=True),flush=True)
    report={"stage":"G14_RUSSIAN_CAPABILITY_LADDER","logical_core_entries":G05,
            "n":128,"scored_closed":112,"open_manual_review":16,"language":"ru","levels":result,"records":records,
            "training_updates":0,"teacher_calls":0,"candidate_accepted":False,
            "limitations":["Pinned G05 current comparison baseline; G06 and G08 not promoted because generation/EOS regressed. No training on this test.",
                           "Levels are authored difficulty groups, not calibrated; 16 questions per level, no stopping after failure. Some simple facts or prompts may overlap past skills.",
                           "Closed scoring uses exact normalized accepted variants only; correct longer answers can be missed. Open answers need rubric-based human review; never count as zero automatically.",
                           "Longer answers capped by level. CALIBRI native research forward vs SmolLM2 BF16 KV; times diagnostic only, no direct CPU certification.",
                           "Target first-token rank is a diagnostic, not semantic understanding or answer accuracy. No historical 616 or universal capability certification."]}
    (out/"G14_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G14_ASSESS",json.dumps({"stage":report["stage"],"n":128,"candidate_accepted":False}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
