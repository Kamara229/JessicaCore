"""One-time NEW32 original Calibri fast tokenizer/wordpiece offsets, raw text only.

No EVAL gold, roles, POS or dependency is passed into tokenization.
All EVAL32 source SHA and label checks are performed before model access
by source-only contract; alignment receives *only* plain Russian text.
No training, evaluation, selection, feedback or new model weights here.
"""
from __future__ import annotations

from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord, CausalAlignment,
)
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar


def align_s18c7_raw_eval_text(tokenizer, raw: str, *, expected_input_ids=None) -> CausalAlignment:
    if tokenizer is None or not isinstance(raw,str) or not raw.endswith("."):
        raise ValueError("S18c7 requires raw Russian text + real Calibri fast tokenizer")
    words=raw_surface_grammar(raw)
    try:
        chat=tokenizer.apply_chat_template(
            [{"role":"user","content":raw}],tokenize=False,add_generation_prompt=True)
    except Exception as exc:
        raise ValueError("S18c7 original user-only chat template missing") from exc
    if not isinstance(chat,str) or chat.count(raw)!=1:
        raise ValueError("S18c7 exactly one raw text in original Calibri prompt")
    begin_text=chat.index(raw)
    try:
        ids=tuple(int(x) for x in tokenizer.encode(chat,add_special_tokens=False))
        enc=tokenizer(chat,add_special_tokens=False,return_offsets_mapping=True)
        returned=tuple(int(x) for x in enc["input_ids"])
        offsets=tuple((int(a),int(b)) for a,b in enc["offset_mapping"])
    except Exception as exc:
        raise ValueError("S18c7 original Calibri tokenizer must expose real offsets") from exc
    if (not ids or ids!=returned or len(ids)!=len(offsets)
            or (expected_input_ids is not None and tuple(expected_input_ids)!=ids)
            or any(a<0 or b<a or b>len(chat) for a,b in offsets)):
        raise ValueError("S18c7 input IDs/real token offsets mismatch")
    mapped=[];used=set()
    for i,word in enumerate(words,1):
        start,end=begin_text+word.start,begin_text+word.end
        indexes=tuple(k for k,(a,b) in enumerate(offsets) if a<end and b>start)
        if (not indexes or any(offsets[k][0]==offsets[k][1] for k in indexes)
                or any(k in used for k in indexes)
                or any(not any(offsets[k][0]<=p<offsets[k][1]
                               for k in indexes) for p in range(start,end))):
            raise ValueError("S18c7 original BPE crosses raw Russian word or lacks bytes")
        used.update(indexes)
        mapped.append(AlignedWord(i,indexes,max(indexes)))
    if (len(mapped)!=len(words) or
            any(mapped[i].last_token>=mapped[i+1].last_token
                for i in range(len(mapped)-1)) or
            mapped[-1].last_token>=len(ids)):
        raise ValueError("S18c7 wrong original BPE ordering")
    import hashlib
    return CausalAlignment(
        hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        len(ids),ids,tuple(mapped),len(ids)-1,False)
