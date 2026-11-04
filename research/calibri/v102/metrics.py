"""Independent source-only head error; actual vocabulary outputs and CPU decoding."""
from __future__ import annotations
import math
import numpy as np
from v98.metrics import logits_topk
from v101.codec import decode_chunk

def original_head_weight_rmse(handle,teacher_weights):
    exact=np.asarray(teacher_weights.detach().cpu().numpy(),dtype=np.float32)
    m,n=handle.shape
    if exact.shape!=(m,n):raise AssertionError("not original trained head")
    total=0.;error=0.
    for a in range(0,m,256):
        b=min(m,a+256)
        q,_=decode_chunk(handle,a,b)
        w=exact[a:b].astype(np.float64)
        diff=w-q.astype(np.float64)
        total+=float(np.sum(w*w))
        error+=float(np.sum(diff*diff))
    if not total>0:raise AssertionError("missing pretrained source")
    return math.sqrt(error/total)

def compare_vocab_logits(teacher,student,tok):
    ref=logits_topk(teacher,tok)
    quant=logits_topk(student,tok)
    return {
        "reference_top10_overlap":len(set(ref["top10_token_ids"])&set(quant["top10_token_ids"])),
        "reference_top1_matched":ref["top10_token_ids"][0]==quant["top10_token_ids"][0],
        "fp32_first_token_text":ref["top10_text"][0],
        "packed_first_token_text":quant["top10_text"][0],
        "student_next_token_entropy_nats":quant["entropy_nats"],
        "is_not_semantic_quality":True}
