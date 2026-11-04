"""Uncached true greedy text for each factor; count full teacher calls honestly."""
from __future__ import annotations
from hashlib import sha256
import torch
from v96.grade import string_breakdown
from v99.patch import native_logits
from .factors import factor_logits
from .spec import GENERATE_TOKENS

@torch.inference_mode()
def greedy(model,tok,prompt,teacher=None,variant=None):
    if (teacher is None)!=(variant is None):
        raise AssertionError("paired oracle must specify teacher and factor")
    tokens=tok.encode(prompt,add_special_tokens=False)
    if not (1<len(tokens)<=96):raise AssertionError("invalid same-token prompt")
    generated=[]
    for _ in range(GENERATE_TOKENS):
        ids=torch.tensor(tokens+generated,dtype=torch.long)[None,:]
        logits=native_logits(model,ids) if variant is None else (
            factor_logits(teacher,model,ids,variant))
        generated.append(int(logits.argmax().item()))
    decoded=tok.decode(generated,skip_special_tokens=True)
    return {"actual_text":decoded,"actual_generated_ids":generated,
        "lexical_observations":string_breakdown(decoded,generated),
        "prompt_sha256":sha256(prompt.encode()).hexdigest(),
        "teacher_model_call_every_token_when_norm_patch":
             variant in ("norm_only","norm_and_head"),
        "dense_FP32_head_used_every_token_when_head_patch":
             variant in ("head_only","norm_and_head"),
        "unpaid_oracle_NEVER_compressed_runtime":variant is not None,
        "semantic_correctness_requires_human_review":True}
