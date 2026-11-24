"""Z76 free generation on existing TRAIN42, NOT generalization.
Separate text quality, exact native prefix, EOS, empty outputs, and loops.
No gold answers provided to decoder during generation.
"""
import torch
from research.calibri_nra3.s119q2i8z66_gpt_paired_eval_prereg import normalize
def check(ok,msg):
    if not ok:raise ValueError(msg)
def common_cap(prefixes):
    cap=min(48,159-max(map(len,prefixes)))
    check(cap>=1,"positive shared context-safe native cap")
    return cap
def greedy(prefix,step,cap,*,vocab=49152,eos=2):
    check(1<=cap<=48 and len(prefix)+cap<160,"bounded native prefix")
    own=[]
    for _ in range(cap):
        x=step(list(prefix)+own)
        check(x.shape==(1,vocab) and not x.requires_grad and bool(torch.isfinite(x).all()),"frozen finite full native logits")
        own.append(int(x.argmax()))
        if own[-1]==eos:break
    return own
def score(ids,gold,decoded,answer,valid,cap):
    eos=bool(ids and ids[-1]==2)
    shared=0
    for a,b in zip(ids,gold):
        if a!=b:break
        shared+=1
    text_match=bool(valid and normalize(decoded)==normalize(answer))
    # A repeated contiguous four-token block (two consecutive copies).
    body=ids[:-1] if eos else ids
    repeat=any(body[i:i+4]==body[i+4:i+8] for i in range(max(0,len(body)-7)))
    return {"valid_UTF8":bool(valid),"nonempty_valid_UTF8":bool(valid and decoded.strip()),
            "empty_text":not bool(decoded.strip()),"EOS":eos,"at_cap_without_EOS":not eos and len(ids)==cap,
            "text_match":text_match,"complete_correct":text_match and eos,
            "native_exact":ids==gold,"gold_prefix_match":shared,"gold_length":len(gold),
            "early_EOS":eos and len(ids)<len(gold),"repeat4_block":repeat,"generated_count":len(ids)}
def summarize(rows):
    check(len(rows)==42 and len({r["id"] for r in rows})==42,"42 distinct training prompts")
    keys=("valid_UTF8","nonempty_valid_UTF8","empty_text","EOS","at_cap_without_EOS","text_match","complete_correct","native_exact","early_EOS","repeat4_block")
    def reduce(rs):
        return {"n":len(rs),**{k:sum(r[k] for r in rs) for k in keys},
                "native_forwards":sum(r["generated_count"] for r in rs),
                "mean_gold_prefix_match":sum(r["gold_prefix_match"] for r in rs)/len(rs),
                "mean_generated_length":sum(r["generated_count"] for r in rs)/len(rs)}
    return {**reduce(rows),"per_kind":{k:reduce([r for r in rows if r["kind"]==k]) for k in sorted({r["kind"] for r in rows})}}
