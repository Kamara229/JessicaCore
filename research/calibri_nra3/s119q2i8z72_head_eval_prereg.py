"""Z72 one-time fresh internal DEV28 for head-only Z70.
Frozen before Z70 scores, no training labels, no FP32-derived data.
Separate from Z66 reserved for Z65.
"""
import json,hashlib
from collections import Counter
from research.calibri_nra3 import s119q2i8z40_new14dev_multiposition_entropy_prereg as z40
from research.calibri_nra3 import s119q2i8z46_fresh28_onpolicy_bpe2_preregister as z46
from research.calibri_nra3 import s119q2i8z56_gpt_fullanswer_prereg as z56
from research.calibri_nra3 import s119q2i8z59_gpt_compositional_curriculum as z59
from research.calibri_nra3 import s119q2i8z66_gpt_paired_eval_prereg as z66
from research.calibri_nra3.s119q2i8z57_fullanswer_eval import CHILD_SHA,GT_SHA
KINDS=z59.KINDS
ARMS=("Z55_PARENT","HEAD_UTF8_BASE","HEAD_UTF8_PENDING_X4","FP32_135M_COMPARISON_ONLY")
MAX_NATIVE_STEPS=48
PAIRS={
"sentence_formulation":[
(("Запиши три слова как предложение в том же порядке: воробей, клюёт, зерно.",["Воробей клюёт зерно."]),("Запиши три слова как предложение в том же порядке: воробей, ищет, зерно.",["Воробей ищет зерно."])),
(("Составь предложение без перестановки: ветер, качает, ветки.",["Ветер качает ветки."]),("Составь предложение без перестановки: ветер, качает, траву.",["Ветер качает траву."]))],
"case_agreement":[
(("Напиши только нужную форму «глубокий»: над ___ озером.",["глубоким"]),("Напиши только нужную форму «глубокий»: возле ___ озера.",["глубокого"])),
(("Одним словом заполни пропуск от «свежий»: купил ___ газету.",["свежую"]),("Одним словом заполни пропуск от «свежий»: доволен ___ газетой.",["свежей"]))],
"agent_action_object":[
(("Толе сумку вернула Лена. Кто вернул сумку? Только имя.",["Лена"]),("Лене сумку вернул Толя. Кто вернул сумку? Только имя.",["Толя"])),
(("Ире билет передала Люба. Кто получил билет? Только имя.",["Ира"]),("Любе билет передала Ира. Кто получил билет? Только имя.",["Люба"]))],
"time_causality":[
(("Борис позвонил врачу до обеда. Позвонил ли он раньше обеда? Только да или нет.",["да"]),("Борис позвонил врачу после обеда. Позвонил ли он раньше обеда? Только да или нет.",["нет"])),
(("Сначала Лев прочитал письмо, потом ответил. Он ответил первым? Только да или нет.",["нет"]),("Сначала Лев ответил, потом прочитал письмо. Он ответил первым? Только да или нет.",["да"]))],
"negation_scope":[
(("Гриша не рисовал, а лепил. Рисовал ли Гриша? Только да или нет.",["нет"]),("Гриша не лепил, а рисовал. Рисовал ли Гриша? Только да или нет.",["да"])),
(("Сова не охотилась, а отдыхала. Отдыхала ли сова? Только да или нет.",["да"]),("Сова не отдыхала, а охотилась. Отдыхала ли сова? Только да или нет.",["нет"]))],
"arithmetic_units":[
(("У Веры 18 марок. Ей подарили 7. Сколько теперь? Только число.",["25"]),("У Веры 18 марок. Она отдала 7. Сколько теперь? Только число.",["11"])),
(("Восемь коробок содержат по три карандаша. Сколько карандашей? Только число.",["24"]),("Восемь коробок содержат по пять карандашей. Сколько карандашей? Только число.",["40"]))],
"semantic_paraphrase":[
(("Замени «одержал победу» одним глаголом прошедшего времени. Только глагол.",["победил","выиграл"]),("Замени «потерпел поражение» одним глаголом прошедшего времени. Только глагол.",["проиграл"])),
(("Замени «проявляет заботу» одним глаголом настоящего времени. Только глагол.",["заботится"]),("Замени «испытывает волнение» одним глаголом настоящего времени. Только глагол.",["волнуется","переживает"]))]}
def rows():
    return [{"id":f"S119Q2I8Z72_{k}_{i+1}_{j+1}","pair_id":f"{k}_{i+1}","kind":k,"split":"DEV",
             "prompt":q,"accepted_answers":a} for k in KINDS for i,pair in enumerate(PAIRS[k]) for j,(q,a) in enumerate(pair)]
def verify():
    data=rows()
    past=z40.previous_spent_and_train_prompts()+z40.rows("TRAIN")+z40.rows("DEV")+z46.rows("DEV")+z56.rows()+z59.curriculum()+z66.rows()
    norm=lambda r:z66.normalize(r["prompt"])
    assert len(data)==len({norm(r) for r in data})==28
    assert set(PAIRS)==set(KINDS) and Counter(r["kind"] for r in data)=={k:4 for k in KINDS}
    assert not {norm(r) for r in data}.intersection(norm(r) for r in past)
    for pairs in PAIRS.values():
        for a,b in pairs:
            assert not {z66.normalize(x) for x in a[1]}.intersection(z66.normalize(x) for x in b[1])
    return {"stage":"Z72_HEAD_FRESH28_PREREG","sha256":hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "cases":28,"pairs":14,"arms":ARMS,"max_native_steps":48,"primary":"complete_correct",
            "secondary":["both_members_complete_correct","text_correct","valid_UTF8","EOS","nonempty_valid_UTF8"],
            "FP32_comparison_only":True,"no_checkpoint_selection":True,"known_exact_prompt_disjoint":True,
            "author_internal_not_external_blind":True,"no_inference_masks":True,"spent_after_first_score":True}
def score(row,text,*,eos,valid_utf8):
    return {**z66.score(row,text,eos=eos,valid_utf8=valid_utf8),
            "nonempty_valid_UTF8":bool(text.strip() and valid_utf8)}
def summarize(records):
    by_id={r["id"]:r for r in rows()}
    assert len(records)==28 and {r["id"] for r in records}==set(by_id)
    groups={}
    for r in records:groups.setdefault(by_id[r["id"]]["pair_id"],[]).append(r)
    assert len(groups)==14 and all(len(v)==2 for v in groups.values())
    return {**{k:sum(bool(r[k]) for r in records) for k in ("text_correct","complete_correct","valid_UTF8","EOS","nonempty_valid_UTF8")},
        "both_members_complete_correct":sum(all(r["complete_correct"] for r in v) for v in groups.values())}
def require(ok,msg):
    if not ok:raise ValueError(msg)
def verify_checkpoint_report(report,pins):
    require(pins["run_id"]==45438751425 and pins["head_sha"]=="fca63dae986f6b3fc4fd9ffd77622d66de9238fa" and pins["conclusion"]=="success","only successful corrected Z70 run")
    require(report["stage"]=="S119Q2I8Z70_GPT_JOINT98_PAIR_EOS_UTF8" and report["Z55_joint_SHA256"]==CHILD_SHA and report["Z55_GT12_SHA256"]==GT_SHA
        and report["total_cached_training_positions"]==25888 and report["total_Adam_updates"]==1568
        and report["total_actual_decoder_forwards"]==3302 and report["FP32_calls"]==0 and report["no_DEV_scoring"]
        and report["output_gate"]==0 and not report["candidate_accepted"],"both complete actual cached training arms")
    require(report["cache_proof"]["all_positions_top1_equal"] and report["cache_proof"]["real_compressed_decoder_forwards"]==1618,"verified cached native parity")
    require(set(report["saved_research_only"])==set(report["training"])==set(pins["joint_sha256"])==set(ARMS[1:3]),"no partial arms")
    for a in ARMS[1:3]:
        r=report["training"][a];m=report["saved_research_only"][a]
        require(r["Adam_updates"]==784 and r["cached_training_positions"]==12944 and r["trainable_parameters"]==247040
            and r["pending_UTF8_multiplier"]==(4 if a==ARMS[2] else 1),"matched training exposure")
        require(m["joint"]["sha256"]==pins["joint_sha256"][a] and len(pins["joint_sha256"][a])==64 and m["GT12"]["sha256"]==GT_SHA,"explicit physical weight pins and frozen GT12")
if __name__=="__main__":
    print("Z72_PREREG",json.dumps(verify(),ensure_ascii=False,sort_keys=True),flush=True)
