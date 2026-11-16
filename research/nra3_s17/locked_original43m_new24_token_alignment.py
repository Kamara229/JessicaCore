"""S17b NEW original frozen43M+GT Calibri EVAL24 physical tokenizer offset mapping.

Do not call original model in this module. No model, EVAL scores, or checkpoint are accessed by this module. It
only aligns the immutable 24 EVAL-only source to original tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord,CausalAlignment,
)
from research.nra3_s17.new_ru24_directed_eval_prereg import (
    EVALSentence,SPLIT,build_new_eval24,preregister_new_eval24,
)


def align_directed_new24(tokenizer,row:EVALSentence,*,
                      expected_input_ids=None)->CausalAlignment:
    """Use **exact** actual Calibri prefill tokens and fast codepoint offsets.

    No gold syntax or semantic role enters tokenizer or frozen base model.
    A BPE piece spanning two different annotated words fails closed.
    """
    registered={r.text_sha256():r for r in build_new_eval24()}
    if (not isinstance(row,EVALSentence) or
        row.split!=SPLIT or row!=registered.get(row.text_sha256()) or
        tokenizer is None):
        raise ValueError("NRA3 S17b unknown or unsealed EVAL24 source")
    # SHA-check the full annotation manifest, not just text strings.
    preregister_new_eval24()
    try:
        full=tokenizer.apply_chat_template(
            [{"role":"user","content":row.text}],
            tokenize=False,add_generation_prompt=True)
    except Exception as exc:
        raise ValueError("NRA3 S17b cannot trust tokenizer chat template") from exc
    if not isinstance(full,str) or full.count(row.text)!=1:
        raise ValueError("NRA3 S17b requires a single exact user prompt")
    pos=full.index(row.text)
    try:
        ids=tuple(int(i) for i in tokenizer.encode(full,add_special_tokens=False))
        details=tokenizer(full,add_special_tokens=False,
                          return_offsets_mapping=True)
        returned=tuple(int(i) for i in details["input_ids"])
        offsets=tuple((int(a),int(b)) for a,b in details["offset_mapping"])
    except Exception as exc:
        raise ValueError("NRA3 S17b requires genuine fast tokenizer offsets") from exc
    if (not ids or ids!=returned or
        (expected_input_ids is not None and ids!=tuple(expected_input_ids))
        or len(ids)!=len(offsets) or
        any(a<0 or b<a or b>len(full) for a,b in offsets)):
        raise ValueError("NRA3 S17b actual CPU prefill/token IDs shifted")
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
            raise ValueError("NRA3 S17b wordpiece straddles word boundary")
        used.update(pieces)
        mapped.append(AlignedWord(word.index,pieces,max(pieces)))
    if (not all(mapped[j].last_token<mapped[j+1].last_token
                for j in range(len(mapped)-1)) or
        mapped[-1].last_token>=len(ids)):
        raise ValueError("NRA3 S17b unordered source UD word alignments")
    return CausalAlignment(row.text_sha256(),len(ids),ids,tuple(mapped),
                           len(ids)-1,True)
