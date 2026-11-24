"""Z104 fixed author-internal held-out transfer, after Z103 TRAIN metrics only.
No scoring or fitting here. All checkpoints immutable, no inference masks.
"""
import json
from collections import Counter
from research.calibri_nra3 import s119q2i8z99_word_fresh_prereg as prior
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer
KNOWN = [
  [
    "мост",
    "Как называется сооружение, соединяющее два берега над водой?",
    "Вставь существительное: Разводной ... подняли для прохода судна."
  ],
  [
    "река",
    "Как называется естественный водный поток, впадающий в море или озеро?",
    "Вставь существительное: Горная ... вышла из берегов."
  ],
  [
    "лес",
    "Как называют природную территорию, густо заросшую деревьями?",
    "Вставь существительное: Хвойный ... темнел на горизонте."
  ],
  [
    "дом",
    "Вставь существительное: Семья переехала в новый частный ...",
    "Как называется жилое здание из трёх букв, первая из которых д?"
  ],
  [
    "кот",
    "Вставь существительное: Рыжий ... замурлыкал на подоконнике.",
    "Назови животное: усатый самец кошки, слово из трёх букв."
  ],
  [
    "пёс",
    "Вставь существительное: Дворовый ... залаял на прохожего.",
    "Назови трёхбуквенное слово с буквой ё, означающее собаку."
  ],
  [
    "снег",
    "Вставь существительное: Белый ... растаял на солнце.",
    "Как называются осадки в виде ледяных снежинок?"
  ],
  [
    "дождь",
    "Вставь существительное: Проливной ... затопил улицу.",
    "Как называются жидкие атмосферные осадки в виде капель, падающих из облаков?"
  ],
  [
    "вода",
    "Вставь существительное: В стакане была питьевая ...",
    "Как называется жидкость, которая при замерзании превращается в лёд?"
  ],
  [
    "хлеб",
    "Вставь существительное: Пекарь вынул из печи ржаной ...",
    "Назови четырёхбуквенное название выпеченного продукта, который режут на ломти для бутербродов."
  ],
  [
    "мяч",
    "Вставь существительное: Вратарь поймал футбольный ...",
    "Назови трёхбуквенное слово для круглого спортивного снаряда."
  ],
  [
    "ключ",
    "Вставь существительное: Запасной ... подошёл к замочной скважине.",
    "Как называется предмет с бородкой, которым отпирают механический замок?"
  ]
]
OTHER = [
  [
    "свеча",
    "Вставь существительное: В подсвечнике горела восковая ..."
  ],
  [
    "зонт",
    "Вставь существительное: Перед ливнем прохожий раскрыл складной ..."
  ],
  [
    "мыло",
    "Как называется средство для мытья рук, которое пенится, слово из четырёх букв?"
  ],
  [
    "сыр",
    "Вставь существительное: Из молока изготовили твёрдый ..."
  ],
  [
    "шар",
    "Назови трёхбуквенное название геометрического тела, ограниченного сферой."
  ],
  [
    "луна",
    "Как называется естественный спутник Земли? Напиши нарицательное название без заглавной буквы."
  ],
  [
    "книга",
    "Вставь существительное: На полке лежала раскрытая ..."
  ],
  [
    "пила",
    "Как называется зубчатый инструмент для распиливания досок вручную?"
  ],
  [
    "ложка",
    "Вставь существительное: Для супа нужна столовая ..."
  ],
  [
    "рыба",
    "Как называется водное позвоночное животное с жабрами, слово из четырёх букв?"
  ],
  [
    "гора",
    "Вставь существительное: Над долиной возвышалась высокая ..."
  ],
  [
    "часы",
    "Как называется прибор для определения времени с циферблатом и стрелками?"
  ]
]
ARMS = ("Z73_BEFORE", "Z103_BASE_GROUP_MEAN_CE", "Z103_MATCHED_TERMINAL_SHARE_CE", "FP32_135M_COMPARISON_ONLY")
CAP = 16
PINS = {
    "run": 45486685533,
    "artifact": 14964954420,
    "artifact_zip_sha256": "ce9125f8483e5f26440a9d2f6b2ec91ded1f56c13e485bac6036b98cd93b72cb",
    "baseline_joint_sha256": "56056d675b50b627923a881aaacbb8c193f76d8335fcc25f89b3734cadd13d14",
    "matched_joint_sha256": "2be2b70d9ef516b27bd0afc94cc80bd9d3a9f0aa406fd306e4e65d935ec79057",
    "both_gt12_sha256": "79fd6297ebd350140f15d68bea9146c08e89b767ed6404c7ba311793616d44a6",
}
def rows():
    data=[]
    for i,(w,p,q) in enumerate(KNOWN,1):
        for j,prompt in enumerate((p,q),1):
            data.append(dict(id=f"Z104_known_{i:02d}_{j}", group="known_new_context", pair=str(i), kind="semantic_paraphrase", split="DEV", origin=prior.z55.ORIGIN, prompt=prompt+" Ответь одним словом строчными буквами.", completion=w))
    for i,(w,p) in enumerate(OTHER,1):
        data.append(dict(id=f"Z104_other_{i:02d}", group="other_target", pair="", kind="semantic_paraphrase", split="DEV", origin=prior.z55.ORIGIN, prompt=p+" Ответь одним словом строчными буквами.", completion=w))
    return data

def verify():
    prior.verify()
    z=prior
    data=rows()
    past=z.z40.previous_spent_and_train_prompts()+z.z40.rows("TRAIN")+z.z40.rows("DEV")+z.z46.rows("DEV")+z.z56.rows()+z.z59.curriculum()+z.z87.fresh()+z.z66.rows()+z.z72.rows()+z.z89.rows()+z.rows()
    norm=lambda r:z.z66.normalize(r["prompt"])
    assert len(data)==len({norm(r) for r in data})==36
    assert not {norm(r) for r in data}.intersection(norm(r) for r in past)
    known={r["completion"] for r in z.z87.fresh()}
    assert {r["completion"] for r in data if r["group"]=="known_new_context"}==known
    assert not ({r["completion"] for r in data if r["group"]=="other_target"} & (known | {r["completion"] for r in z.rows()}))
    assert Counter(r["group"] for r in data)=={"known_new_context":24,"other_target":12}
    assert Counter(r["pair"] for r in data if r["pair"])=={str(i):2 for i in range(1,13)}
    return {"stage":"Z104_TERMINAL_TRANSFER_PREREG","sha256":z.z55.digest(data),"arms":ARMS,"cases":36,"native_cap":CAP,"primary":"exact_word_and_natural_EOS_and_valid_UTF8","other_target_relative_to_Z87_and_Z99_only":True,"internal_author_DEV_not_external_blind":True,"known_exact_prompt_disjoint":True,"FP32_comparison_only":True,"no_model_calls":True,"no_scoring":True,"spent_after_first_score":True,"output_gate":0,"pins":PINS,"decision":"exploratory signal only: matched >= base + 4 complete, no decrease in UTF8/EOS or other-target exact; paired both-correct reported separately; no promotion"}

def native(tok):
    proof=verify()
    data=[verified_native_answer(tok,r) for r in rows()]
    assert all(len(r["prefix"])+CAP<160 and len(r["answer_native_with_eos"])<=CAP for r in data)
    return {**proof,"native_plan_SHA256":prior.z55.digest([{"id":r["id"],"prefix":r["prefix"],"targets":r["answer_native_with_eos"]} for r in data]),"max_native_prompt_length":max(len(r["prefix"]) for r in data),"max_native_answer_length":max(len(r["answer_native_with_eos"]) for r in data)}

def score(row,text,*,eos,valid_utf8):
    return {**prior.score(row,text,eos=eos,valid_utf8=valid_utf8),"pair":row["pair"]}

def summarize(records):
    assert len(records)==36 and {r["id"] for r in records}=={r["id"] for r in rows()}
    def group(items):
        return {"n":len(items),**{k:sum(r[k] for r in items) for k in ("text_correct","complete_correct","EOS","valid_UTF8")}}
    pairs=[[r for r in records if r["pair"]==str(i)] for i in range(1,13)]
    assert all(len(p)==2 for p in pairs)
    return {"overall":group(records),"by_group":{g:group([r for r in records if r["group"]==g]) for g in ("known_new_context","other_target")},"known_pairs_both_correct":sum(all(r["complete_correct"] for r in p) for p in pairs)}

if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    print("Z104_PREREG "+json.dumps(native(AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)),sort_keys=True),flush=True)
