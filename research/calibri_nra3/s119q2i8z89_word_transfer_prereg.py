"""Precommit internal word transfer DEV before Z88 outcomes.
No learning, model output, or FP32-generated labels.
'Other word' means absent from Z87 target set, not necessarily historical training.
"""
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
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer
KNOWN=[
("мост","С какого сооружения автомобили переезжают с одного берега на другой? Назови сооружение."),
("река","Как называется водоём, который имеет исток, течение и устье?"),
("лес","Как назвать одним словом место, где густо растут дикие деревья?"),
("дом","Квартира находится в жилом здании. Назови это здание обычным словом."),
("кот","Как называется взрослый самец кошки одним коротким словом?"),
("пёс","Назови короткий трёхбуквенный синоним слова «кобель»."),
("снег","Что образует белый покров на земле зимой после снегопада?"),
("дождь","Что идёт из облаков в виде капель воды, а не снежинок?"),
("вода","Что замерзает в лёд при охлаждении обычной питьевой жидкости? Назови жидкость."),
("хлеб","Назови пищевой продукт, который продаётся в виде батонов и буханок."),
("мяч","Какой круглый предмет игрок забрасывает в баскетбольную корзину?"),
("ключ","Как называется металлический предмет с зубцами для открывания дверного замка?")]
OTHER=[
("стул","Как называется предмет мебели со спинкой, на котором сидит один человек?"),
("зонт","Что раскрывают над головой, чтобы укрыться от дождя?"),
("сок","Как называется напиток, который выжимают из яблок?"),
("соль","Как называется белая приправа с химической формулой NaCl?"),
("лук","Как называется овощ с луковицей, при нарезке которого часто текут слёзы?"),
("сыр","Как называется молочный продукт, который часто нарезают ломтиками для бутерброда?"),
("часы","Как называется прибор с циферблатом для определения времени?"),
("шарф","Как называется длинная полоска ткани, которой обматывают шею в мороз?"),
("книга","Как называется печатное издание из сшитых или скреплённых страниц?"),
("ложка","Как называется столовый прибор с углублением, которым едят суп?"),
("рыба","Как одним общим словом назвать щуку, карася и окуня?"),
("пила","Как называется ручной инструмент с зубчатым полотном для распиливания дерева?")]
ARMS=("Z73_BEFORE","Z88_AFTER","FP32_135M_COMPARISON_ONLY")
CAP=16
def rows():
    return [{"id":f"Z89_{group}_{i+1:02d}","group":group,"kind":"semantic_paraphrase","split":"DEV","origin":z55.ORIGIN,
             "prompt":p+" Ответь одним словом строчными буквами.","completion":w}
            for group,items in (("shared_target_new_prompt",KNOWN),("other_target",OTHER)) for i,(w,p) in enumerate(items)]
def verify():
    data=rows();train=z59.curriculum()+z87.fresh()
    past=z40.previous_spent_and_train_prompts()+z40.rows("TRAIN")+z40.rows("DEV")+z46.rows("DEV")+z56.rows()+train+z66.rows()+z72.rows()
    norm=lambda r:z66.normalize(r["prompt"])
    assert len(data)==len({norm(r) for r in data})==24
    assert not {norm(r) for r in data}.intersection(norm(r) for r in past)
    known={x["completion"] for x in z87.fresh()}
    assert {r["completion"] for r in data if r["group"]=="shared_target_new_prompt"}==known
    assert not known.intersection(r["completion"] for r in data if r["group"]=="other_target")
    assert Counter(r["group"] for r in data)=={"shared_target_new_prompt":12,"other_target":12}
    return {"stage":"Z89_WORD_TRANSFER_PREREG","sha256":z55.digest(data),"arms":ARMS,"cases":24,"native_cap":CAP,
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
    print("Z89_PREREG "+json.dumps(native(AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)),sort_keys=True),flush=True)
