"""GPT authored shared-target short-word training block.
Four distinct task families predict each of the same 12 target nouns.
Keep all 98 previous training rows as replay. No model-generated data.
"""
import json
from collections import Counter
from research.calibri_nra3 import s119q2i8z55_gpt_curriculum as z55
from research.calibri_nra3 import s119q2i8z59_gpt_compositional_curriculum as z59
from research.calibri_nra3 import s119q2i8z72_head_eval_prereg as z72
from research.calibri_nra3 import s119q2i8z66_gpt_paired_eval_prereg as z66
from research.calibri_nra3 import s119q2i8z56_gpt_fullanswer_prereg as z56
from research.calibri_nra3 import s119q2i8z46_fresh28_onpolicy_bpe2_preregister as z46
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as z40
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer
from research.calibri_nra3.s119q2i8z63_utf8_transition_objective import advance,byte_pieces
FAMILIES=("definition","cloze","contrast","plural")
KINDS=("semantic_paraphrase","sentence_formulation","agent_action_object","case_agreement")
# Explicit independently authored prompts. Contrast rows have matched distractor pairs.
WORDS=[
("мост","Как называется сооружение для перехода через реку?","Через овраг построили новый ___. Вставь существительное.","мосты","река"),
("река","Как называется естественный водный поток с руслом и берегами?","Между деревнями течёт широкая ___. Вставь существительное.","реки","мост"),
("лес","Как называется большая территория, покрытая деревьями?","За деревней шумит густой ___. Вставь существительное.","леса","дом"),
("дом","Как одним словом назвать здание, в котором живут люди?","Семья построила новый ___. Вставь существительное.","дома","лес"),
("кот","Как называют самца домашней кошки?","На печке мурлычет усатый ___. Вставь существительное.","коты","пёс"),
("пёс","Назови трёхбуквенное слово, обозначающее самца собаки.","У ворот залаял сторожевой ___. Вставь существительное из трёх букв.","псы","кот"),
("снег","Как называют белые ледяные хлопья, падающие зимой?","Зимой выпал пушистый ___. Вставь существительное.","снега","дождь"),
("дождь","Как называют осадки, состоящие из капель жидкой воды?","Из тучи пошёл сильный ___. Вставь существительное.","дожди","снег"),
("вода","Назови жидкость с химической формулой H₂O обычным словом.","Из крана течёт холодная ___. Вставь существительное.","воды","хлеб"),
("хлеб","Как называется продукт из муки, который пекут буханками?","Пекарь вынул из печи свежий ___. Вставь существительное.","хлеба","вода"),
("мяч","Как называется упругий шар для игры в футбол?","Футболист ударил ногой по предмету. Это был ___. Вставь существительное.","мячи","ключ"),
("ключ","Как называется предмет, которым отпирают замок?","Для замка нужен подходящий ___. Вставь существительное.","ключи","мяч")]
CONTRAST=[
"Что соединяет берега","Что течёт по руслу","Где растёт множество деревьев","Где живёт семья",
"Кто мурлычет","Кто лает","Что падает белыми хлопьями","Что падает каплями",
"Что наливают из крана","Что выпекают буханками","Что пинают в футболе","Чем отпирают замок"]
def fresh():
    result=[]
    for j,f in enumerate(FAMILIES):
        for i,(word,definition,cloze,plural,other) in enumerate(WORDS):
            a,b=(word,other) if i%2==0 else (other,word)
            q=(definition,cloze,f"{CONTRAST[i]}: {a} или {b}?",f"Поставь слово «{plural}» в форму именительного падежа единственного числа.")[j]
            result.append({"id":f"S119Q2I8Z87_GPT_{f}_{i+1:02d}","kind":KINDS[j],"family":f,"split":"TRAIN",
                "origin":z55.ORIGIN,"prompt":q+" Ответь одним словом строчными буквами.","completion":word})
    return result
def verify():
    rows=fresh();old=z59.curriculum()
    past=z40.previous_spent_and_train_prompts()+z40.rows("TRAIN")+z40.rows("DEV")+z46.rows("DEV")+z56.rows()+old+z66.rows()+z72.rows()
    norm=lambda r:z66.normalize(r["prompt"])
    assert len(rows)==48 and len(old)==98 and len({norm(r) for r in rows})==48
    assert not {norm(r) for r in rows}.intersection(norm(r) for r in past)
    assert Counter(r["completion"] for r in rows)=={w[0]:4 for w in WORDS}
    assert Counter(r["family"] for r in rows)=={f:12 for f in FAMILIES}
    assert all(r["split"]=="TRAIN" and r["origin"]==z55.ORIGIN for r in rows+old)
    assert z55.digest(old)=="fb68dde68942c9b4b686e88e64a7916d6152fd11e853de989495315fc6de93a5"
    return {"stage":"Z87_SHARED_TARGET_TRAIN_PREFLIGHT","new_TRAIN":48,"replay_TRAIN":98,"total_TRAIN":146,
        "target_words":12,"task_families":4,"new_SHA256":z55.digest(rows),"replay_SHA256":z55.digest(old),
        "known_exact_prompt_disjoint":True,"FP32_calls":0,"model_forwards":0,"optimizer_steps":0,
        "candidate_accepted":False,"output_gate":0}
def native(tok):
    m=verify();pieces=byte_pieces(tok);records=[]
    for row in fresh():
        n=verified_native_answer(tok,row);gold=n["answer_native_with_eos"];pending=b"";states=[]
        assert len(gold)>=2
        for g in gold:
            states.append(pending.hex())
            if g==tok.eos_token_id:assert not pending
            else:
                pending=advance(pending,pieces[g]);assert pending is not None
        records.append({"id":row["id"],"family":row["family"],"word":row["completion"],"target_ids":gold,"states":states})
    # Check that all contexts of each target have identical native tokenization.
    for w,*_ in WORDS:
        assert len({tuple(r["target_ids"]) for r in records if r["word"]==w})==1
    counts={str(t):dict(Counter(r["target_ids"][t-1] for r in records)) for t in (1,2)}
    return {**m,"native_target_positions":sum(len(r["target_ids"]) for r in records),
        "pending_positions":sum(bool(s) for r in records for s in r["states"]),
        "first_two_native_label_counts":counts,"native_plan_SHA256":z55.digest(records),
        "all_target_labels_covered_in_other_families":True,"per_case":records}
if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    result=native(tok)
    from pathlib import Path
    p=Path("results/z87/report.json");p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("Z87_NATIVE_PREFLIGHT "+json.dumps({k:v for k,v in result.items() if k!="per_case"},sort_keys=True),flush=True)
