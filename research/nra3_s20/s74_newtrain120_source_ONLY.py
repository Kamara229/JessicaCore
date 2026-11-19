"""NRA2 S74: source-only NEW TRAIN contrastive situation groups, before native CPU.

120 precommitted source-only synthetic Russian examples, grouped as 12
entire lexical/participant families × ten distinct situation contexts.
NO case-specific answer, role, situation label, gold, expectation or
inference route exists in this source file. The manually *constructed*
TRAIN-only situation targets live in the separate S74 gold module and
must never be imported by original Calibri43M tokenizer/H8 capture.

This corpus is deliberately templated and authored by one process; it
is NOT a naturalistic independent evaluation population and its lexical
templates cannot be considered prospective generalization. Only TRAIN.
All original protected S66 HELD32 and EVAL32/128 stay CLOSED.
"""
from __future__ import annotations

import hashlib
import json

# agent nominative/instrumental, recipient dative/nominative,
# pronoun dative, expected participle, verb form and masculine object.
# Lexical groups are kept together in grouped TRAIN splits.
FAMILIES=(
    ("курьер","курьером","Марии","Мария","ей","ждала","акт","передал","передан"),
    ("инженер","инженером","Ивану","Иван","ему","ждал","чертёж","показал","показан"),
    ("архивист","архивистом","Ольге","Ольга","ей","ждала","документ","вручил","вручён"),
    ("секретарь","секретарём","Борису","Борис","ему","ждал","пропуск","отправил","отправлен"),
    ("мастер","мастером","Нине","Нина","ей","ждала","ключ","передал","передан"),
    ("оператор","оператором","Алексею","Алексей","ему","ждал","отчёт","направил","направлен"),
    ("лаборант","лаборантом","Анне","Анна","ей","ждала","пакет","предоставил","предоставлен"),
    ("менеджер","менеджером","Денису","Денис","ему","ждал","договор","вручил","вручён"),
    ("конструктор","конструктором","Елене","Елена","ей","ждала","план","показал","показан"),
    ("техник","техником","Петру","Пётр","ему","ждал","образец","передал","передан"),
    ("водитель","водителем","Татьяне","Татьяна","ей","ждала","кейс","доставил","доставлен"),
    ("контролёр","контролёром","Роману","Роман","ему","ждал","журнал","передал","передан"),
)
CASES_PER_FAMILY=10
SPLIT="NRA2_S74_NEW_SYNTHETIC_TRAIN_ONLY_NOT_EVAL"
GROUP_TRAIN=tuple(f"F{i:02d}" for i in range(1,10))
GROUP_DIAG=tuple(f"F{i:02d}" for i in range(10,13))
assert len(FAMILIES)==12 and len(GROUP_TRAIN)==9 and len(GROUP_DIAG)==3


def require(ok,why):
    if not bool(ok):
        raise ValueError("NRA2 S74 raw-only source: "+why)


def new_source_only_train120():
    """Return ONLY IDs, group and input text. No metadata or target class."""
    result=[]
    for n,(agent,agent_inst,rec_dat,rec_nom,pron,expect,obj,verb,passive) in enumerate(FAMILIES,1):
        a=agent.capitalize()
        # One lexical family supports overlapping miniature causal situations.
        variations=(
            f"{a} {verb} {rec_dat} {obj}.",
            f"В стенограмме сохранилась фраза «{a} {verb} {rec_dat} {obj}», однако подтверждений не было.",
            f"{a} не {verb} {rec_dat} {obj}.",
            f"{rec_dat} был {passive} {obj} {agent_inst}.",
            f"По неподтверждённым сообщениям, {agent} {verb} {rec_dat} {obj}.",
            f"{rec_dat} приснилось, будто {agent} {verb} {pron} {obj}.",
            f"Сначала {agent} не {verb} {rec_dat} {obj}, но затем всё же {verb}.",
            f"{a} {verb} {rec_dat} {obj}, но начальник не отправил бухгалтеру копию.",
            f"{rec_nom} {expect} {obj}, и {agent} наконец {verb} {pron} {obj}.",
            f"{a} лишь процитировал слова «Я {verb} {rec_dat} {obj}», но затем отрицал передачу.",
        )
        require(len(variations)==CASES_PER_FAMILY
                and len(set(variations))==CASES_PER_FAMILY,
                "source minimal-pair variants must be unique")
        for variant,text in enumerate(variations,1):
            require(text==text.strip() and 25<=len(text)<=180
                    and text.count("«")==text.count("»")
                    and "  " not in text, "raw UTF8 input source malformed")
            result.append({
                "row_id":f"T{n:02d}_{variant:02d}",
                "family":f"F{n:02d}",
                "text":text,
            })
    require(len(result)==120 and len({q["text"] for q in result})==120
            and len({q["row_id"] for q in result})==120
            and {r["family"] for r in result}
                ==set(GROUP_TRAIN)|set(GROUP_DIAG),
            "S74 precommitted 120 unique NEW TRAIN text inputs")
    require(all(set(r)=={"row_id","family","text"} for r in result),
            "NEVER add situation/role/gold to H8 source input record")
    return tuple(result)


def source_sha256():
    raw=json.dumps(new_source_only_train120(),
                   sort_keys=True,ensure_ascii=False,separators=(",",":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def grouped_train_source_only():
    rows=new_source_only_train120()
    return (
        tuple(q for q in rows if q["family"] in GROUP_TRAIN),
        tuple(q for q in rows if q["family"] in GROUP_DIAG),
    )
