"""Genuine KV-cache deterministic greedy decoding of fresh TRAIN diagnostic prompts."""
from __future__ import annotations
import hashlib
import torch
from .provenance import DIAGNOSTIC_PROMPTS, TOKENS_PER_PROMPT
from .diagnostics import greedy_summary

@torch.inference_mode()
def generate_diagnostic(model, tok):
    output_sequences=[]
    for prompt in DIAGNOSTIC_PROMPTS:
        x=tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        result=model(input_ids=x,use_cache=True)
        cache=result.past_key_values
        token=result.logits[:,-1,:].argmax(dim=-1,keepdim=True)
        emitted=[int(token.item())]
        for _ in range(TOKENS_PER_PROMPT-1):
            result=model(input_ids=token,past_key_values=cache,use_cache=True)
            cache=result.past_key_values
            token=result.logits[:,-1,:].argmax(dim=-1,keepdim=True)
            emitted.append(int(token.item()))
        output_sequences.append(emitted)
    report=greedy_summary(output_sequences)
    for case, prompt in zip(report["prompts"],DIAGNOSTIC_PROMPTS):
        case["diagnostic_prompt_SHA"]=hashlib.sha256(prompt.encode()).hexdigest()
    report["genuine_KV_cache"]=True
    report["source_split"]="TRAIN_diagnostic_prompts_not_official_TEST"
    return report
