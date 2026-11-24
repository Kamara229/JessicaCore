"""Fresh GPT-authored internal DEV42 in 21 counterfactual pairs.
Freeze before Z64/Z65 results, score only after eligible checkpoint SHA pin.
FP32 comparative answers only. Never train or tune on these examples.
"""
import hashlib,json,re,unicodedata
from collections import Counter
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as z40
from research.calibri_nra3 import s119q2i8z46_fresh28_onpolicy_bpe2_preregister as z46
from research.calibri_nra3 import s119q2i8z55_gpt_curriculum as z55
from research.calibri_nra3 import s119q2i8z56_gpt_fullanswer_prereg as z56
from research.calibri_nra3 import s119q2i8z59_gpt_compositional_curriculum as z59
KINDS=z55.KINDS
MAX_NATIVE_STEPS=48
ARMS=("Z55_PARENT","Z65_PAIR_EOS","Z65_PAIR_EOS_UTF8","FP32_135M_COMPARISON_ONLY")
PAIRS={
"sentence_formulation":[
(("Напиши одно предложение из трёх слов в заданном порядке: лебедь, плывёт, медленно.",["Лебедь плывёт медленно."]),("Напиши одно предложение из трёх слов в заданном порядке: лебедь, плывёт, быстро.",["Лебедь плывёт быстро."])),
(("Собери предложение, сохраняя порядок слов: трактор, везёт, сено.",["Трактор везёт сено."]),("Собери предложение, сохраняя порядок слов: грузовик, везёт, сено.",["Грузовик везёт сено."])),
(("Запиши предложение в порядке: ёж, нашёл, яблоко.",["Ёж нашёл яблоко."]),("Запиши предложение в порядке: ёж, потерял, яблоко.",["Ёж потерял яблоко."]))],
"case_agreement":[
(("Вставь только пропущенное прилагательное от слова «яркий»: Мы любуемся ___ звездой.",["яркой"]),("Вставь только пропущенное прилагательное от слова «яркий»: Мы видим ___ звезду.",["яркую"])),
(("Напиши только нужную форму слова «старый»: Он подошёл к ___ мосту.",["старому"]),("Напиши только нужную форму слова «старый»: Он отошёл от ___ моста.",["старого"])),
(("Ответь одним словом: какую форму слова «летний» поставить в «под ___ солнцем»?",["летним"]),("Ответь одним словом: какую форму слова «летний» поставить в «к ___ солнцу»?",["летнему"]))],
"agent_action_object":[
(("Ане открытку вручила Вика. Кто получил открытку? Только имя.",["Аня"]),("Вике открытку вручила Аня. Кто получил открытку? Только имя.",["Вика"])),
(("Гоше книгу одолжил Матвей. Кто одолжил книгу? Только имя.",["Матвей"]),("Матвею книгу одолжил Гоша. Кто одолжил книгу? Только имя.",["Гоша"])),
(("Флоре рисунок показала Инга. Кто показал рисунок? Только имя.",["Инга"]),("Инге рисунок показала Флора. Кто показал рисунок? Только имя.",["Флора"]))],
"time_causality":[
(("Катя выключила плиту до того, как накрыла стол. Выключила плиту раньше? Ответь только да или нет.",["да"]),("Катя выключила плиту после того, как накрыла стол. Выключила плиту раньше? Ответь только да или нет.",["нет"])),
(("Сначала Аркадий достал зонт, затем вышел. Он вышел первым? Ответь только да или нет.",["нет"]),("Сначала Аркадий вышел, затем достал зонт. Он вышел первым? Ответь только да или нет.",["да"])),
(("Когда дождь закончился, Лариса открыла окно. Окно открыли позже окончания дождя? Ответь только да или нет.",["да"]),("До окончания дождя Лариса открыла окно. Окно открыли позже окончания дождя? Ответь только да или нет.",["нет"]))],
"negation_scope":[
(("Тимур не купил хлеб, зато купил сыр. Купил ли Тимур хлеб? Только да или нет.",["нет"]),("Тимур купил хлеб, зато не купил сыр. Купил ли Тимур хлеб? Только да или нет.",["да"])),
(("Жук не полз, а летел. Летел ли жук? Только да или нет.",["да"]),("Жук не летел, а полз. Летел ли жук? Только да или нет.",["нет"])),
(("Ася не открыла дверь, а закрыла её. Открыла ли Ася дверь? Только да или нет.",["нет"]),("Ася не закрыла дверь, а открыла её. Открыла ли Ася дверь? Только да или нет.",["да"]))],
"semantic_paraphrase":[
(("Замени оборот «испытывает страх» одним глаголом. Напиши только глагол.",["боится"]),("Замени оборот «испытывает удивление» одним глаголом. Напиши только глагол.",["удивляется"])),
(("Замени оборот «дал согласие» одним глаголом. Напиши только глагол.",["согласился"]),("Замени оборот «дал отказ» одним глаголом. Напиши только глагол.",["отказался"])),
(("Замени оборот «задала вопрос» одним глаголом. Напиши только глагол.",["спросила"]),("Замени оборот «дала совет» одним глаголом. Напиши только глагол.",["посоветовала"]))],
"arithmetic_units":[
(("На полке 14 книг, ещё 9 поставили. Ответь только итоговым числом.",["23"]),("На полке 14 книг, 9 убрали. Ответь только итоговым числом.",["5"])),
(("В 6 рядах по 4 стула. Сколько всего? Только число.",["24"]),("В 7 рядах по 4 стула. Сколько всего? Только число.",["28"])),
(("45 бусин поделили поровну между 5 детьми. Сколько каждому? Только число.",["9"]),("45 бусин поделили поровну между 9 детьми. Сколько каждому? Только число.",["5"]))]}
def normalize(text):
    # No substring matching or acceptance of extra contradictory content.
    s=unicodedata.normalize("NFC",text).casefold().replace("ё","е")
    s=" ".join(s.split()).strip()
    return s.rstrip(".!?").strip()
def rows():
    return [{"id":f"S119Q2I8Z66_{k}_{i+1}_{j+1}","pair_id":f"{k}_{i+1}","kind":k,
             "split":"DEV","prompt":p,"accepted_answers":a}
            for k in KINDS for i,pair in enumerate(PAIRS[k]) for j,(p,a) in enumerate(pair)]
def score(row,text,*,eos,valid_utf8):
    match=normalize(text) in {normalize(a) for a in row["accepted_answers"]}
    return {"text_correct":bool(match and valid_utf8),
            "complete_correct":bool(match and valid_utf8 and eos),
            "valid_UTF8":bool(valid_utf8),"EOS":bool(eos)}
def summarize(records):
    known={r["id"]:r for r in rows()}
    assert len(records)==42 and {r["id"] for r in records}==set(known)
    groups={}
    for r in records:
        pair=known[r["id"]]["pair_id"]
        groups.setdefault(pair,[]).append(r)
    assert len(groups)==21 and all(len(v)==2 for v in groups.values())
    return {**{k:sum(bool(r[k]) for r in records) for k in ("text_correct","complete_correct","valid_UTF8","EOS")},
            "both_members_text_correct":sum(all(r["text_correct"] for r in v) for v in groups.values()),
            "both_members_complete_correct":sum(all(r["complete_correct"] for r in v) for v in groups.values()),
            "pairs_same_normalized_output":sum(normalize(v[0]["generated_text"])==normalize(v[1]["generated_text"]) for v in groups.values()),
            "n":42,"pairs":21}
def verify():
    data=rows()
    past=z40.previous_spent_and_train_prompts()+z40.rows("TRAIN")+z40.rows("DEV")+z46.rows("DEV")+z56.rows()+z59.curriculum()
    assert set(PAIRS)==set(KINDS) and len(data)==42
    assert len({normalize(r["prompt"]) for r in data})==42
    assert not {normalize(r["prompt"]) for r in data}.intersection(normalize(r["prompt"]) for r in past)
    assert Counter(r["kind"] for r in data)=={k:6 for k in KINDS}
    for pairs in PAIRS.values():
        assert len(pairs)==3
        for a,b in pairs:
            assert not {normalize(x) for x in a[1]}.intersection(normalize(x) for x in b[1])
    assert all(r["split"]=="DEV" for r in data)
    return {"stage":"Z66_GPT_FRESH42_PAIRED_PREREG","sha256":hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "cases":42,"pairs":21,"arms":ARMS,"max_native_steps":MAX_NATIVE_STEPS,
            "primary":"both_members_complete_correct",
            "secondary":["complete_correct","text_correct","valid_UTF8","EOS","pairs_same_normalized_output"],
            "exploratory_gate_UTF8_vs_matched_control":{"min_UTF8_count_gain":4,"no_complete_answer_regression":True,"no_paired_complete_regression":True},
            "checkpoint_policy":"require both Z65 arms successfully completed and physically SHA-pinned before any scoring; otherwise do not run",
            "same_native_greedy_own_prefix":True,"no_inference_mask":True,
            "FP32_comparison_only_not_gate":True,"author_internal_not_external_blind":True,
            "known_exact_prompt_disjoint":True,"spent_after_first_score":True,
            "candidate_accepted":False,"output_gate":0}
if __name__=="__main__":
    print("Z66_PREREG",json.dumps(verify(),ensure_ascii=False,sort_keys=True),flush=True)
