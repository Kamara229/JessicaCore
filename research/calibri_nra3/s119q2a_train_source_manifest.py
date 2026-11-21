"""S119q2a: ORIGINAL AUTHOR-CREATED tiny Russian TRAIN/source calibration manifest.

This is a deterministic, TEMPLATE-BASED DATA ENGINEERING SMOKE corpus.
It is NOT SynTagRus and never reads S118-B GOLD, spent EVAL128, J7 answers,
user dialogs, external websites, or any pretrained model output. Do NOT
claim its CALIBRATION rows represent independent broad generalization.
All completion labels are deliberately authored by explicit symbolic rules.
Only TRAIN rows may be used for gradient training in the following q2b
experiment; CALIBRATION rows are read-only sanity checks; final blind eval
must be collected independently and kept unopened until final evaluation.

No network, no model weights, no tokenizer, no inference, no deployment.
Output records are physical, deterministic, self-auditing and SHA-bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

STAGE = "CALIBRI_S119Q2A_AUTHOR_CREATED_TRAIN_SOURCE_PROVENANCE_ONLY"
VERSION = "S119Q2A_v1_20261122"
KINDS = (
    "roles_word_order", "negation", "transitive_comparison",
    "event_time", "arithmetic", "paraphrase",
)
SPLITS = ("TRAIN", "CALIBRATION")
TRAIN_MALE = (
    ("Артём","Артёма"),("Егор","Егора"),("Никита","Никиты"),
    ("Максим","Максима"),("Лев","Льва"),("Вадим","Вадима"),
    ("Павел","Павла"),("Данил","Данила"),("Фёдор","Фёдора"),
    ("Матвей","Матвея"),("Борис","Бориса"),("Степан","Степана"),
)
CAL_MALE = (("Юрий","Юрия"),("Руслан","Руслана"),("Глеб","Глеба"))
TRAIN_FEMALE = (
    ("Диана","Диане"),("Алина","Алине"),("Кира","Кире"),
    ("Мила","Миле"),("Полина","Полине"),("Рита","Рите"),
    ("Таня","Тане"),("Соня","Соне"),("Жанна","Жанне"),
    ("Вика","Вике"),("Лиза","Лизе"),("Тамара","Тамаре"),
)
CAL_FEMALE = (("Зоя","Зое"),("Вера","Вере"),("Алёна","Алёне"))
TRAIN_OBJECTS = (
    "книгу","папку","карту","газету","посылку","записку",
    "коробку","фотографию","тетрадь","ручку","открытку","квитанцию",
)
CAL_OBJECTS = ("схему","корзину","монету")
TRAIN_EVENTS = (
    "погас свет", "открылась дверь", "заиграла музыка",
    "зазвонил телефон", "начался урок", "закончился дождь",
    "включился экран", "поднялся занавес", "вышел поезд",
    "прозвенел будильник", "затихла сирена", "загорелась лампа",
)
CAL_EVENTS = ("запустился насос","закрылись ворота","стартовал таймер")
TRAIN_PARAPHRASE_OBJECTS = (
    "альбом","билет","телефон","планшет","подарок","атлас",
    "мяч","карандаш","цветок","рюкзак","зонт","плащ",
)
CAL_PARAPHRASE_OBJECTS = ("фонарь","календарь","браслет")
# S119n authored prompts are already spent as diagnostic examples; NEVER
# silently repurpose as new calibration or independent quality test material.
SPENT_S119N_PROMPTS = (
    "Сколько будет 4 плюс 3? Ответь одним числом.",
    "Книгу Лене передал Олег. Кто получил книгу?",
    "Миша не давал Саше книгу. Получил ли Саша книгу от Миши?",
    "Петя выше Коли, а Коля выше Маши. Кто самый высокий?",
    "Сначала Анна вышла из дома, затем приехал автобус. Что произошло раньше?",
    "Как иначе сказать: «Миша купил велосипед»?",
)
EXPECTED_COUNTS = {"TRAIN": 72, "CALIBRATION": 18}
EXPECTED_MANIFEST_SHA256 = "1a9bf7f4d1564d27d0ab752186ed97dfccc28f41ac9870761fe1e1374ea4a17e"  # first real CI #44954874290
EVIDENCE_ONLY = True


def demand(condition: bool, reason: str):
    if not condition:
        raise ValueError("S119q2a FAIL CLOSED: " + reason)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)+"\n").encode("utf8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def authored_row(kind, split, i):
    """Small deterministic authored labels for infrastructure smoke, not QA gold."""
    demand(kind in KINDS and split in SPLITS and
           type(i) is int and 0 <= i < (12 if split=="TRAIN" else 3),
           "only predeclared exact original generated source rows")
    males = TRAIN_MALE if split=="TRAIN" else CAL_MALE
    females = TRAIN_FEMALE if split=="TRAIN" else CAL_FEMALE
    objects = TRAIN_OBJECTS if split=="TRAIN" else CAL_OBJECTS
    events = TRAIN_EVENTS if split=="TRAIN" else CAL_EVENTS
    items = (TRAIN_PARAPHRASE_OBJECTS if split=="TRAIN"
             else CAL_PARAPHRASE_OBJECTS)
    male, male_gen = males[i % len(males)]
    female, female_dat = females[i % len(females)]
    obj = objects[i % len(objects)]
    other,other_gen = males[(i+1)%len(males)]
    third,third_gen = males[(i+2)%len(males)]
    if kind=="roles_word_order":
        if i%2==0:
            prompt=f"{male} передал {obj} {female_dat}. Кто получил {obj}?"
            answer=f"{female}."
            rule="agent_first_recipient"
        else:
            prompt=f"{obj.capitalize()} {female_dat} передал {male}. Кто передал {obj}?"
            answer=f"{male}."
            rule="object_front_sender"
    elif kind=="negation":
        negative=(i%2==0)
        particle="не " if negative else ""
        prompt=f"{male} {particle}передал {obj} {female_dat}. Состоялась ли передача?"
        answer="Нет." if negative else "Да."
        rule="negative_explicit" if negative else "positive_explicit"
    elif kind=="transitive_comparison":
        if i%2==0:
            prompt=(f"{male} выше {other_gen}, а {other} выше {third_gen}. "
                    "Кто выше всех?")
            rule="greater_chain"
        else:
            prompt=(f"{third} ниже {other_gen}, а {other} ниже {male_gen}. "
                    "Кто самый высокий?")
            rule="less_chain_reversed"
        answer=f"{male}."
    elif kind=="event_time":
        start=events[i%len(events)]
        end=events[(i+1)%len(events)]
        if i%2==0:
            prompt=(f"Сначала {start}, затем {end}. "
                    "Какое событие было раньше?")
            rule="first_then_second"
        else:
            prompt=(f"Событие «{end}» произошло после события «{start}». "
                    "Что произошло первым?")
            rule="second_after_first"
        answer=start.capitalize()+"."
    elif kind=="arithmetic":
        if i%2==0:
            a=(9+3*i) if split=="TRAIN" else (151+7*i)
            b=2+(i%3)
            prompt=(f"Было {a} коробок, {b} коробки отправили. "
                    "Сколько осталось?")
            answer=str(a-b)+"."
            rule="difference"
        else:
            a=(4+2*i) if split=="TRAIN" else (113+5*i)
            b=(3+i) if split=="TRAIN" else (42+i)
            prompt=(f"Было {a} билетов, добавили ещё {b}. "
                    "Сколько стало?")
            answer=str(a+b)+"."
            rule="sum"
    elif kind=="paraphrase":
        if i%2==0:
            prompt=f"Перефразируй: «{male} приобрёл {items[i%len(items)]}»."
            answer=f"{male} купил {items[i%len(items)]}."
            rule="purchase_synonym"
        else:
            prompt=f"Перефразируй: «{male} начал работу»."
            answer=f"{male} приступил к работе."
            rule="begin_synonym"
    else:
        raise AssertionError("unreachable")
    return {"id":f"{VERSION}_{split}_{kind}_{i:02d}",
            "kind":kind,"split":split,"rule":rule,"prompt":prompt,
            "completion":answer,
            "source_kind":"original_deterministic_author_created_synthetic",
            "rights_scope":"research_only_not_external_corpus",
            "input_provenance":"inline_rule_and_literal_source_banks",
            "ground_truth_kind":"generator_symbolic_rule_NOT_independent_QA"}


def build_rows():
    return [authored_row(kind,split,i)
            for split in SPLITS for kind in KINDS
            for i in range(12 if split=="TRAIN" else 3)]


def validate_rows(rows: list[dict]):
    """Cross-split partition, exact generator provenance, zero borrowed GOLD."""
    demand(type(rows) is list and len(rows)==90,
           "exactly 90 source smoke examples; not larger corpus")
    expected=build_rows()
    demand(rows == expected,
           "exact generator output must match source text, rule and answer")
    ids=set();prompts=set();by_split=Counter()
    spent={sha(s.encode("utf8")) for s in SPENT_S119N_PROMPTS}
    for row in rows:
        demand(set(row)==set(expected[0]) and
               all(type(row[x]) is str and row[x] for x in row),
               "all literal provenance fields required")
        demand(row["id"] not in ids,"duplicate identity")
        demand(row["prompt"] not in prompts,"cross-split repeated exact prompt")
        demand(sha(row["prompt"].encode("utf8")) not in spent,
               "reuse of previously spent S119n source prompt")
        demand(row["prompt"] == row["prompt"].strip() and
               row["completion"] == row["completion"].strip() and
               row["completion"].endswith("."),
               "noncanonical prompt/completion")
        ids.add(row["id"]);prompts.add(row["prompt"])
        by_split[row["split"]]+=1
    demand(dict(by_split)==EXPECTED_COUNTS,
           "fixed TRAIN/calibration distribution")
    # Compare complete morphological name forms and strip punctuation rather
    # than matching whitespace tokens (e.g. «Артём or Артём. are valid).
    import re
    train_roots = {form for item in (*TRAIN_MALE, *TRAIN_FEMALE)
                   for form in item}
    cal_roots = {form for item in (*CAL_MALE, *CAL_FEMALE)
                 for form in item}
    demand(not train_roots & cal_roots and
           not set(TRAIN_EVENTS)&set(CAL_EVENTS) and
           not set(TRAIN_PARAPHRASE_OBJECTS)&set(CAL_PARAPHRASE_OBJECTS),
           "distinct subject and event lexical banks across split")
    for row in rows:
        own_roots = train_roots if row["split"]=="TRAIN" else cal_roots
        other_roots = cal_roots if row["split"]=="TRAIN" else train_roots
        words=set(re.findall(r"[А-Яа-яЁё]+",row["prompt"]))
        demand(not words & other_roots,"wrong-split named entity leaked")
        if row["kind"] in ("roles_word_order","negation",
                           "transitive_comparison","paraphrase"):
            demand(bool(words & own_roots),
                   "role/source prompts must contain only authorized bank")
    return {"rows_total":len(rows),"split_counts":dict(by_split),
            "per_task_train":12,"per_task_calibration":3,
            "only_template_synthetic":True,
            "no_independent_quality_claim":True,
            "no_protected_original_616_J7_or_S118_B_GOLD_used":True,
            "no_spent_EVAL128_or_S119n_prompt_used":True,
            "no_model_or_tokenizer_loaded":True,
            "no_training_or_gradients":True,
            "third_party_source_loaded":False,
            "no_final_blind_test_created":True}


def build_manifest(rows):
    metrics=validate_rows(rows)
    digests={split:sha(b"".join(canonical_bytes(row)
                  for row in rows if row["split"]==split))
             for split in SPLITS}
    manifest={
        "stage":STAGE,
        "version":VERSION,
        "generator_source_file":"research/calibri_nra3/s119q2a_train_source_manifest.py",
        "record_format":"canonical sorted utf8 jsonl",
        "split_SHA256":digests,
        "synthetic_rows_SHA256":sha(b"".join(canonical_bytes(r) for r in rows)),
        "metrics":metrics,
        "not_frozen_final_independent_eval":True,
        "candidate_accepted":False,"output_gate":0,
        "next_step":"S119q2b legal train-only native-vocab onepass optimizer with matched local ablation",
    }
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--expected-source-sha",type=str,default=None)
    args=p.parse_args()
    demand(not args.out.exists(),"never overwrite previously generated source")
    rows=build_rows()
    manifest=build_manifest(rows)
    if args.expected_source_sha is not None:
        demand(args.expected_source_sha==manifest["synthetic_rows_SHA256"],
               "physically pinned externally predeclared full row digest")
    if EXPECTED_MANIFEST_SHA256 is not None:
        demand(manifest["synthetic_rows_SHA256"]==EXPECTED_MANIFEST_SHA256,
               "committed frozen full synthetic source SHA mismatch")
    args.out.mkdir(parents=True,exist_ok=False)
    for split in SPLITS:
        f=args.out/("S119Q2A_"+split+".jsonl")
        f.write_bytes(b"".join(canonical_bytes(r) for r in rows
                               if r["split"]==split))
    (args.out/"S119Q2A_MANIFEST.json").write_bytes(canonical_bytes(manifest))
    print("S119Q2A_SOURCE_ONLY "+json.dumps({
        "counts":manifest["metrics"]["split_counts"],
        "source_sha256":manifest["synthetic_rows_SHA256"],
        "calibration_is_independent_QA":False,
        "training_completed":False,
        "accepted":False,
    },ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
