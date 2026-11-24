"""GPT-authored, source-only precommit for new 28 internal DEV.
No inference here. Freeze candidate checkpoint SHA before first scoring.
FP32 135M is a comparison-only arm, never a teacher or checkpoint selector.
Do not train, select or tune on these prompts or any resulting answers.
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as z40
from research.calibri_nra3 import s119q2i8z46_fresh28_onpolicy_bpe2_preregister as z46
from research.calibri_nra3.s119q2i8z55_gpt_curriculum import curriculum
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS

MAX_NATIVE_STEPS=48
ARMS=("Z10_A_PARENT","Z55_GPT_ONLY","FP32_135M_COMPARISON_ONLY")
CASES={
"sentence_formulation":[
("Составь простое предложение, используя слова «пчела», «летит», «к цветку».",("Пчела летит к цветку.","К цветку летит пчела.")),
("Составь простое предложение из слов «мальчик», «моет», «руки».",("Мальчик моет руки.","Руки моет мальчик.")),
("Составь предложение из слов «рыбак», «ловит», «рыбу».",("Рыбак ловит рыбу.","Рыбу ловит рыбак.")),
("Составь предложение из слов «солнце», «греет», «землю».",("Солнце греет землю.","Землю греет солнце."))],
"case_agreement":[
("Исправь согласование, сохрани остальные слова: Мы шли по узкий тропе.",("Мы шли по узкой тропе.",)),
("Исправь согласование, сохрани остальные слова: Собака лежит возле деревянный будки.",("Собака лежит возле деревянной будки.",)),
("Исправь падеж, сохрани остальные слова: Я доволен новый рисунком.",("Я доволен новым рисунком.",)),
("Исправь согласование, сохрани остальные слова: Ключ находится в маленькая сумке.",("Ключ находится в маленькой сумке.",))],
"agent_action_object":[
("Косте тетрадь принесла Лида. Кто принёс тетрадь? Ответь только именем.",("Лида",)),
("Наташе цветы подарил Артём. Кто получил цветы? Ответь только именем.",("Наташа",)),
("Роману письмо вручила Света. Кто вручил письмо? Ответь только именем.",("Света",)),
("Олегу зонт одолжила Катя. Кто получил зонт? Ответь только именем.",("Олег",))],
"time_causality":[
("Сначала Дима почистил ботинки, затем надел пальто. Что он сделал раньше?",("Дима почистил ботинки.","Почистил ботинки.","Сначала он почистил ботинки.")),
("Мила вышла из дома после того, как закрыла кран. Что она сделала позже?",("Мила вышла из дома.","Вышла из дома.","Она вышла из дома.")),
("Перед посадкой семян Глеб полил землю. Что Глеб сделал первым?",("Глеб полил землю.","Полил землю.","Он полил землю.")),
("Сначала чайник закипел, потом Саша заварил чай. Что было вторым?",("Саша заварил чай.","Заварил чай.","Он заварил чай."))],
"negation_scope":[
("Федя не пел, а играл на гитаре. Что он делал?",("Федя играл на гитаре.","Играл на гитаре.","Он играл на гитаре.")),
("Утка не летела, а плавала. Что она делала?",("Утка плавала.","Плавала.","Она плавала.")),
("Соня не смотрела фильм, а читала журнал. Что она делала?",("Соня читала журнал.","Читала журнал.","Она читала журнал.")),
("Петя не мыл пол, а вытирал пыль. Что он делал?",("Петя вытирал пыль.","Вытирал пыль.","Он вытирал пыль."))],
"arithmetic_units":[
("В вазе 9 тюльпанов, добавили ещё 6. Сколько стало? Ответь только цифрами.",("15",)),
("Было 21 печенье, съели 8. Сколько осталось? Ответь только цифрами.",("13",)),
("В 5 пакетах по 7 орехов. Сколько всего? Ответь только цифрами.",("35",)),
("36 карточек раздали поровну 6 ученикам. Сколько получил каждый? Ответь только цифрами.",("6",))],
"semantic_paraphrase":[
("Замени «производил ремонт» словом «ремонтировал»: Папа производил ремонт стула.",("Папа ремонтировал стул.",)),
("Замени «дал ответ» словом «ответил»: Ученик дал ответ учителю.",("Ученик ответил учителю.",)),
("Упрости выражение «сделала покупку»: Вика сделала покупку хлеба.",("Вика купила хлеб.",)),
("Упрости выражение «оказал поддержку»: Брат оказал поддержку сестре.",("Брат поддержал сестру.",))]}
def normalize(text):
    s=unicodedata.normalize("NFC",text).casefold().replace("ё","е")
    return " ".join(s.split()).rstrip(".!?").strip()
def rows():
    return [{"id":f"S119Q2I8Z56_{k}_{i+1:02d}","kind":k,"split":"DEV",
             "prompt":p,"accepted_answers":list(a)}
            for k in KINDS for i,(p,a) in enumerate(CASES[k])]
def score_text(row,text,*,eos,valid_utf8):
    # Accept only whole answer variants, never substring or extra contradictory text.
    match=normalize(text) in {normalize(a) for a in row["accepted_answers"]}
    return {"whole_text_match":match and valid_utf8,
            "whole_answer_and_EOS":match and valid_utf8 and eos,
            "EOS":eos,"valid_UTF8":valid_utf8}
def verify():
    current=rows()
    old=z40.previous_spent_and_train_prompts()+z40.rows("DEV")+z40.rows("TRAIN")+z46.rows("DEV")+curriculum()
    norm=lambda r:normalize(r["prompt"])
    assert set(CASES)==set(KINDS)
    assert len(current)==len({norm(r) for r in current})==28
    assert not {norm(r) for r in current}.intersection(norm(r) for r in old)
    assert Counter(r["kind"] for r in current)=={k:4 for k in KINDS}
    assert all(r["split"]=="DEV" and r["accepted_answers"] and
               all(normalize(a) for a in r["accepted_answers"]) for r in current)
    return {"stage":"Z56_GPT_FRESH28_FULLANSWER_PREREG",
            "sha256":hashlib.sha256(json.dumps(current,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "cases":28,"arms":ARMS,"max_native_steps":MAX_NATIVE_STEPS,
            "same_native_chat_prompt_and_greedy":True,
            "freeze_Z55_saved_checkpoint_SHA_before_scoring":True,
            "no_teacher_forcing_or_gold_in_generation":True,
            "primary":"whole_answer_and_EOS",
            "secondary":["whole_text_match","EOS","valid_UTF8","generated_native_length","per_kind_counts"],
            "predeclared_internal_candidate_gate":{"min_complete_correct":4,
                "min_complete_gain_vs_parent":2,"no_UTF8_regression_vs_parent":True},
            "FP32_report_only_never_gate_or_training":True,
            "author_internal_not_external_blind":True,
            "spent_after_first_score":True,"candidate_accepted":False,"output_gate":0}
