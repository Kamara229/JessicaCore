"""Z99 fixed fresh internal DEV prior to Z97 outcome; exact word generation. No model calls or training. Z89 remains spent."""
import json
from collections import Counter
from research.calibri_nra3 import s119q2i8z55_gpt_curriculum as z55
from research.calibri_nra3 import s119q2i8z59_gpt_compositional_curriculum as z59
from research.calibri_nra3 import s119q2i8z87_short_word_curriculum as z87
from research.calibri_nra3 import s119q2i8z72_head_eval_prereg as z72
from research.calibri_nra3 import s119q2i8z66_gpt_paired_eval_prereg as z66
from research.calibri_nra3 import s119q2i8z56_gpt_fullanswer_prereg as z56
from research.calibri_nra3 import s119q2i8z46_fresh28_onpolicy_bpe2_preregister as z46
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as z40
from research.calibri_nra3 import s119q2i8z89_word_transfer_prereg as z89
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer
KNOWN=[
  [
    "мост",
    "Закончи фразу: Через ущелье построили подвесной ..."
  ],
  [
    "река",
    "Закончи фразу: От истока до устья течёт ..."
  ],
  [
    "лес",
    "Закончи фразу: Большой массив деревьев называют словом ... (именительный падеж)."
  ],
  [
    "дом",
    "Закончи фразу: На участке построили жилой ..."
  ],
  [
    "кот",
    "Закончи фразу: Самец домашней кошки — это ..."
  ],
  [
    "пёс",
    "Вставь трёхбуквенный синоним: верный ... охраняет двор."
  ],
  [
    "снег",
    "Закончи фразу: Снежинки падают, и землю покрывает ..."
  ],
  [
    "дождь",
    "Закончи фразу: Из тучи капает вода, идёт ..."
  ],
  [
    "вода",
    "Назови вещество H₂O обычным русским словом."
  ],
  [
    "хлеб",
    "Закончи фразу: В пекарне из муки выпекают ..."
  ],
  [
    "мяч",
    "Закончи фразу: Футболист ударил ногой по предмету, который называется ... (именительный падеж)."
  ],
  [
    "ключ",
    "Закончи фразу: Чтобы отпереть замок, в него вставляют ..."
  ]
]
OTHER=[
  [
    "вилка",
    "Закончи фразу: Столовый прибор с зубцами — это ..."
  ],
  [
    "окно",
    "Закончи фразу: Остеклённый проём в стене — это ..."
  ],
  [
    "стол",
    "Закончи фразу: Обеденную посуду ставят на ..."
  ],
  [
    "кран",
    "Закончи фразу: Чтобы перекрыть подачу воды из трубы, закрывают ..."
  ],
  [
    "дверь",
    "Закончи фразу: Чтобы войти в комнату, открывают ..."
  ],
  [
    "печь",
    "Закончи фразу: Дровяное устройство для отопления, которое топят, — это ..."
  ],
  [
    "мёд",
    "Закончи фразу: Сладкий продукт, который собирают пчёлы, — это ..."
  ],
  [
    "круг",
    "Назови плоскую фигуру, ограниченную окружностью."
  ],
  [
    "ночь",
    "Закончи фразу: После вечера наступает ..."
  ],
  [
    "звук",
    "Закончи фразу: Колебания воздуха, воспринимаемые слухом, — это ..."
  ],
  [
    "игла",
    "Назови тонкий заострённый инструмент с ушком для нитки."
  ],
  [
    "лист",
    "Закончи фразу: Один отдельный элемент листвы дерева — это ..."
  ]
]
ARMS=("Z73_BEFORE","Z96_MICRO","Z97_BATCH_REPLAY","FP32_135M_COMPARISON_ONLY")
CAP=16
def rows():
    return [{"id":f"Z99_{group}_{i+1:02d}","group":group,"kind":"semantic_paraphrase","split":"DEV","origin":z55.ORIGIN,
             "prompt":p+" Ответь одним словом строчными буквами.","completion":w}
            for group,items in (("shared_target_new_prompt",KNOWN),("other_target",OTHER)) for i,(w,p) in enumerate(items)]
def verify():
    data=rows();train=z59.curriculum()+z87.fresh()
    past=z40.previous_spent_and_train_prompts()+z40.rows("TRAIN")+z40.rows("DEV")+z46.rows("DEV")+z56.rows()+train+z66.rows()+z72.rows()+z89.rows()
    norm=lambda r:z66.normalize(r["prompt"])
    assert len(data)==len({norm(r) for r in data})==24
    assert not {norm(r) for r in data}.intersection(norm(r) for r in past)
    known={x["completion"] for x in z87.fresh()}
    assert {r["completion"] for r in data if r["group"]=="shared_target_new_prompt"}==known
    assert not known.intersection(r["completion"] for r in data if r["group"]=="other_target")
    assert Counter(r["group"] for r in data)=={"shared_target_new_prompt":12,"other_target":12}
    return {"stage":"Z99_WORD_FRESH_PREREG","sha256":z55.digest(data),"arms":ARMS,"cases":24,"native_cap":CAP,
        "primary":"exact_word_and_natural_EOS_and_valid_UTF8","other_target_relative_only_to_Z87":True,
        "known_exact_prompt_disjoint":True,"internal_author_DEV_not_external_blind":True,
        "FP32_comparison_only":True,"no_model_calls":True,"no_scoring":True,"spent_after_first_score":True,"output_gate":0}
def native(tok):
    proof=verify();data=[verified_native_answer(tok,r) for r in rows()]
    assert all(len(r["prefix"])+CAP<160 and len(r["answer_native_with_eos"])<=CAP for r in data)
    return {**proof,"native_plan_SHA256":z55.digest([{"id":r["id"],"prefix":r["prefix"],"targets":r["answer_native_with_eos"]} for r in data]),
        "max_native_prompt_length":max(len(r["prefix"]) for r in data),"max_native_answer_length":max(len(r["answer_native_with_eos"]) for r in data)}
def score(row,text,*,eos,valid_utf8):
    exact=bool(valid_utf8 and text.strip()==row["completion"])
    return {"id":row["id"],"group":row["group"],"text_correct":exact,"complete_correct":exact and bool(eos),
            "EOS":bool(eos),"valid_UTF8":bool(valid_utf8)}
def summarize(records):
    assert len(records)==24 and {r["id"] for r in records}=={r["id"] for r in rows()}
    def group(items):
        return {"n":len(items),**{k:sum(r[k] for r in items) for k in ("text_correct","complete_correct","EOS","valid_UTF8")}}
    return {"overall":group(records),"by_group":{g:group([r for r in records if r["group"]==g]) for g in ("shared_target_new_prompt","other_target")}}
if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    print("Z99_PREREG "+json.dumps(native(AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)),sort_keys=True),flush=True)
