"""F07: verified own-error mining for subsequent integrated recovery.

Pure curriculum utilities. No model load or training at import. Call only on
authorized TRAIN rows using a frozen physical candidate's real greedy decode.
Do not mine DEV/J7 or score a candidate on its own training mistakes.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import Counter

EOS=2
VOCAB=49152
_INTEGER=re.compile(r"[+-]?[0-9]{1,12}\Z")

def classify(generated, expected, tokenizer):
    """Correct/wrong numeric value vs malformed output vs missing EOS."""
    assert type(expected) is int
    assert isinstance(generated,list) and generated
    assert all(type(x) is int and 0<=x<VOCAB for x in generated)
    assert EOS not in generated[:-1]
    ended=generated[-1]==EOS
    content=generated[:-1] if ended else generated
    text=tokenizer.decode(content,skip_special_tokens=False).strip()
    if not ended:
        return "NO_EOS",text
    if not _INTEGER.fullmatch(text):
        return "MALFORMED",text
    return ("CORRECT" if int(text)==expected else "WRONG_VALUE"),text

def first_divergence(gold,negative):
    """Same causal prefix, valid token IDs, never contradict equal tokens."""
    assert gold and gold[-1]==EOS and EOS not in gold[:-1]
    assert negative and EOS not in negative[:-1]
    assert all(type(x) is int and 0<=x<VOCAB for x in gold+negative)
    for i,(g,b) in enumerate(zip(gold,negative)):
        if g!=b:
            return i
    return None

def generated_curriculum(split,n):
    """TRAIN only here: numeric operations, independently computed gold."""
    if split!="TRAIN":
        raise ValueError("own-error mining forbidden outside TRAIN")
    if not 1<=n<=512:
        raise ValueError("bounded corpus")
    rows=[]
    for i in range(n):
        a=1301+11*i;b=3+i%13
        m=2+i%19
        p=(10,20,25,50)[i%4];v=20*(i+17)
        items=(
            ("add",f"Сложи числа {a} и {b}.",a+b),
            ("sub",f"Найди разность {a} и {b}.",a-b),
            ("mul",f"Чему равно произведение {m} и {b}?",m*b),
            ("percent",f"Вычисли {p}% от {v}.",p*v//100),
            ("equation",f"Найди x: x + {b} = {a}.",a-b),
        )
        for kind,q,y in items:
            rows.append({"split":"TRAIN","id":f"F07_{kind}_{i:04d}",
                         "kind":kind,"prompt":q+" Ответь одним целым числом.",
                         "expected":y})
    assert len({x["prompt"] for x in rows})==len(rows)
    return rows

def mine(rows,tokenizer,generate,encode_gold,checkpoint_sha):
    """Return verified frozen own negatives, positive and correction targets.

    generate(prompt) must be true greedy own-token generation, no gold or
    wrong answer provided to the model. encode_gold(prompt, str(expected))
    returns native completion tokens including EOS for the same prompt.
    A generation with correct value but other formatting is never negative.
    NO_EOS and MALFORMED are diagnosed separately, not trained as wrong math.
    """
    if not re.fullmatch(r"[0-9a-f]{64}",checkpoint_sha):
        raise ValueError("require frozen physical checkpoint SHA")
    if any(r["split"]!="TRAIN" for r in rows):
        raise ValueError("protected/evaluation rows forbidden")
    ids=set();prompts=set();out=[];counts=Counter()
    for row in rows:
        if row["id"] in ids or row["prompt"] in prompts:
            raise ValueError("duplicate TRAIN identity")
        ids.add(row["id"]);prompts.add(row["prompt"])
        own=generate(row["prompt"])
        status,text=classify(own,row["expected"],tokenizer)
        counts[status]+=1
        gold=encode_gold(row["prompt"],str(row["expected"]))
        div=None
        if status=="WRONG_VALUE":
            div=first_divergence(gold,own)
            if div is None:
                raise ValueError("wrong numeric value without divergent token")
        entry={**row,"source_checkpoint_sha256":checkpoint_sha,
               "own_native_ids":own,"gold_native_ids":gold,
               "own_status":status,"own_text":text,
               "negative_first_divergence":div,
               "positive_target":str(row["expected"]),
               "negative_is_math_error":status=="WRONG_VALUE"}
        if status=="WRONG_VALUE":
            # Numeric result is independently checked; do not invent a
            # reasoning trace or assert why the model made this mistake.
            entry["correction_prompt"]=(
                row["prompt"]+" Предложено решение: "+text+
                ". Проверь ответ и исправь его, если он неверен.")
            entry["correction_target"]="Ответ неверен. Правильный результат: "+str(row["expected"])+"."
        out.append(entry)
    raw=json.dumps(out,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    return out,{"rows":len(out),"statuses":dict(counts),
                "source_checkpoint_sha256":checkpoint_sha,
                "train_manifest_sha256":hashlib.sha256(raw).hexdigest(),
                "verified_wrong_value_pairs":counts["WRONG_VALUE"],
                "wrong_answers_used_as_positive_targets":0}

def preference_loss(logits,gold_id,negative_id,*,margin=.5):
    """One paired loss on logits from the identical gold-prefix position."""
    import torch
    from torch.nn import functional as F
    assert logits.shape[-1]==VOCAB and gold_id!=negative_id
    assert 0<=gold_id<VOCAB and 0<=negative_id<VOCAB
    return F.softplus(margin-(logits[...,gold_id]-logits[...,negative_id])).mean()
