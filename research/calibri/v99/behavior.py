"""Compare exact top-token support and decode visible 12-token continuations.

The decoder deliberately does a full forward per timestep: KV-cache
interventions would require a paired cached teacher and are not validated
here. No full-model throughput figures permitted for this oracle.
"""
from __future__ import annotations
import hashlib
import torch
from v96.grade import string_breakdown
from v98.metrics import logits_topk
from .patch import native_logits,paired_one_token_logits
from .spec import GENERATE_TOKENS

def compare_logits(teacher,native,tok):
    original=logits_topk(teacher,tok)
    student=logits_topk(native,tok)
    return {"FP32":original,"CB59":student,
        "reference_top10_overlap_count":len(set(original["top10_token_ids"])&
                                           set(student["top10_token_ids"])),
        "FP32_top1_matched":original["top10_token_ids"][0]==student["top10_token_ids"][0],
        "not_equivalent_to_semantic_correctness":True}

@torch.inference_mode()
def generate(model,tok,prompt,stage=None,teacher=None):
    if (stage is None)!=(teacher is None):
        raise AssertionError("an original FP32 stage must have original FP32 teacher")
    encoded=tok.encode(prompt,add_special_tokens=False)
    if not (1<len(encoded)<=96):raise AssertionError("unexpected source prompt length")
    emitted=[]
    for _ in range(GENERATE_TOKENS):
        ids=torch.tensor(encoded+emitted,dtype=torch.long)[None,:]
        if stage is None:
            logits=native_logits(model,ids)
        else:
            _,logits=paired_one_token_logits(teacher,model,ids,stage)
        token=int(logits.argmax().item())
        emitted.append(token)
    decoded=tok.decode(emitted,skip_special_tokens=True)
    return {"text":decoded,
        "prompt_SHA256":hashlib.sha256(prompt.encode()).hexdigest(),
        "generated_ids":emitted,
        "lexical":string_breakdown(decoded,emitted),
        "teacher_recomputed_on_every_student_prefix":stage is not None,
        "free_generation_semantic_quality_requires_human_review":True,
        "not_an_eligible_native_token_per_second_benchmark":True}
