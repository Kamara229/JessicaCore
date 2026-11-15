"""Future true 43M Calibri native tokenizer S9 offset ABI, TRAIN-only.

Do not call original model in this module. No original 43M weights, hidden
or any EVAL are read. This is an immutable-source word↔subtoken checker.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord,CausalAlignment,
)
from research.nra3_s9.new_ru_train72 import (
    NewRussianTrain,S9_TRAIN_SPLIT,build_s9_train72,seal_train72,
)


def align_new_train72(tokenizer,row:NewRussianTrain,*,
                      expected_input_ids=None)->CausalAlignment:
    """Use **exact** actual Calibri prefill tokens and fast codepoint offsets.

    No gold syntax or semantic role enters tokenizer or frozen base model.
    A BPE piece spanning two different annotated words fails closed.
    """
    registered={r.sha256():r for r in build_s9_train72()}
    if (not isinstance(row,NewRussianTrain) or
        row.split!=S9_TRAIN_SPLIT or row!=registered.get(row.sha256()) or
        tokenizer is None):
        raise ValueError("NRA3 S9 rejects unknown/EVAL UD source")
    # SHA-check the full annotation manifest, not just text strings.
    seal_train72()
    try:
        full=tokenizer.apply_chat_template(
            [{"role":"user","content":row.text}],
            tokenize=False,add_generation_prompt=True)
    except Exception as exc:
        raise ValueError("NRA3 S9 cannot trust tokenizer chat template") from exc
    if not isinstance(full,str) or full.count(row.text)!=1:
        raise ValueError("NRA3 S9 requires a single exact user prompt")
    pos=full.index(row.text)
    try:
        ids=tuple(int(i) for i in tokenizer.encode(full,add_special_tokens=False))
        details=tokenizer(full,add_special_tokens=False,
                          return_offsets_mapping=True)
        returned=tuple(int(i) for i in details["input_ids"])
        offsets=tuple((int(a),int(b)) for a,b in details["offset_mapping"])
    except Exception as exc:
        raise ValueError("NRA3 S9 requires genuine fast tokenizer offsets") from exc
    if (not ids or ids!=returned or
        (expected_input_ids is not None and ids!=tuple(expected_input_ids))
        or len(ids)!=len(offsets) or
        any(a<0 or b<a or b>len(full) for a,b in offsets)):
        raise ValueError("NRA3 S9 actual CPU prefill/token IDs shifted")
    mapped=[]
    used=set()
    for word in row.words:
        start=pos+word.start;end=pos+word.end
        pieces=tuple(k for k,(a,b) in enumerate(offsets)
                     if a<end and b>start)
        if (not pieces or any(offsets[k][1]==offsets[k][0]
                              for k in pieces) or
            any(k in used for k in pieces) or
            any(not any(offsets[k][0]<=p<offsets[k][1]
                        for k in pieces)
                for p in range(start,end))):
            raise ValueError("NRA3 S9 wordpiece straddles word boundary")
        used.update(pieces)
        mapped.append(AlignedWord(word.word_id,pieces,max(pieces)))
    if (not all(mapped[j].last_token<mapped[j+1].last_token
                for j in range(len(mapped)-1)) or
        mapped[-1].last_token>=len(ids)):
        raise ValueError("NRA3 S9 unordered source UD word alignments")
    return CausalAlignment(row.sha256(),len(ids),ids,tuple(mapped),
                           len(ids)-1,False)
