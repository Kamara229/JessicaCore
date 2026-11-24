"""GPT-authored full-sequence curriculum. FP32 baseline never used in training.
28 new deterministic Russian tasks + 14 fixed historical legal TRAIN replay.
Historical parent may include prior distillation; this is not a clean-room model.
"""
import hashlib
import json
from collections import Counter
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as z40
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS

CASES={
"sentence_formulation":[
("Составь короткое предложение: девочка рисует круг.","Девочка рисует круг."),
("Составь короткое предложение: ветер качает траву.","Ветер качает траву."),
("Составь короткое предложение: голубь клюёт зерно.","Голубь клюёт зерно."),
("Составь короткое предложение: мастер чинит замок.","Мастер чинит замок.")],
"case_agreement":[
("Исправь только ошибку в слове: Мы подошли к новая двери.","Мы подошли к новой двери."),
("Исправь только ошибку: Кот сидит под высокий столом.","Кот сидит под высоким столом."),
("Исправь только ошибку: Я горжусь младшая сестрой.","Я горжусь младшей сестрой."),
("Исправь только ошибку: Письмо лежит в белый конверте.","Письмо лежит в белом конверте.")],
"agent_action_object":[
("Вере книгу дала Нина. Кто дал книгу Вере? Назови имя.","Нина."),
("Дедушке очки принёс Юра. Кто принёс очки? Назови имя.","Юра."),
("Тане шарф связала Маша. Кто связал шарф? Назови имя.","Маша."),
("Илье билет купил Борис. Кто купил билет? Назови имя.","Борис.")],
"time_causality":[
("Лена сначала вымыла чашку, затем налила чай. Что она сделала сначала?","Лена вымыла чашку."),
("Павел закрыл окно после того, как выключил свет. Что он сделал позже?","Павел закрыл окно."),
("Перед звонком Оля записала номер. Что она сделала раньше?","Оля записала номер."),
("Сначала снег растаял, потом высохла дорожка. Что произошло вторым?","Дорожка высохла.")],
"negation_scope":[
("Антон не читал, а писал письмо. Что делал Антон?","Антон писал письмо."),
("Лиса не спала, а бежала. Что делала лиса?","Лиса бежала."),
("Врач не звонил, а осматривал пациента. Что делал врач?","Врач осматривал пациента."),
("Ира не шила, а вязала. Что делала Ира?","Ира вязала.")],
"arithmetic_units":[
("У Нади 8 карандашей. Она получила ещё 7. Сколько стало? Ответь числом.","15"),
("На ветке 17 птиц. 9 улетели. Сколько осталось? Ответь числом.","8"),
("В каждом из 4 ящиков по 6 мячей. Сколько всего? Ответь числом.","24"),
("Раздели 28 груш поровну между 7 детьми. Сколько у каждого? Ответь числом.","4")],
"semantic_paraphrase":[
("Замени выражение «очень быстро» одним словом: Он очень быстро убежал.","Он стремительно убежал."),
("Упрости фразу без изменения смысла: Она оказала помощь другу.","Она помогла другу."),
("Упрости фразу: Он осуществил поиск ключа.","Он искал ключ."),
("Упрости фразу: Мы приняли решение уйти.","Мы решили уйти.")]
}
ORIGIN="original_newly_authored_examples_no_third_party_corpus"
def digest(rows):
    return hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def curriculum():
    assert set(CASES)==set(KINDS)
    new=[{"id":f"S119Q2I8Z55_GPT_{k}_{i+1:02d}","kind":k,"split":"TRAIN","origin":ORIGIN,
          "prompt":p,"completion":a} for k in KINDS for i,(p,a) in enumerate(CASES[k])]
    old=z40.rows("TRAIN")
    replay=[dict(r,origin=ORIGIN) for k in KINDS for r in [x for x in old if x["kind"]==k][:2]]
    known=z40.previous_spent_and_train_prompts()+z40.rows("DEV")+old
    norm=lambda r:" ".join(r["prompt"].casefold().split())
    assert len(new)==28 and len(replay)==14 and len(old)==84
    assert len({norm(r) for r in new})==28
    assert not {norm(r) for r in new}.intersection(norm(r) for r in known)
    assert Counter(r["kind"] for r in new)=={k:4 for k in KINDS}
    assert Counter(r["kind"] for r in replay)=={k:2 for k in KINDS}
    assert all(r["split"]=="TRAIN" for r in new+replay)
    # Fixed interleave: two new and one replay per domain, twice.
    train=[]
    for k in KINDS:
        a=[r for r in new if r["kind"]==k];b=[r for r in replay if r["kind"]==k]
        train+=a[:2]+b[:1]+a[2:]+b[1:]
    return train
