"""Architecture-independent byte interface: 10MB or 10GB host LLM.

No shared tokenizer, no model-size-specific dense adapter, no original
layer access required; only UTF-8 prompt and candidate text.
"""
import torch
from .protocol import MAX_TEXT_BYTES

SEP=256
PAD=257

def encode(prompt,answer):
    p=prompt.encode("utf-8",errors="replace")
    a=answer.encode("utf-8",errors="replace")
    # Budget both prompt and candidate, never silently erase entire answer.
    pn=min(len(p),MAX_TEXT_BYTES//2)
    an=MAX_TEXT_BYTES-pn-1
    p=p[-pn:] if pn else b""
    a=a[:an]
    series=list(p)+[SEP]+list(a)
    series.extend([PAD]*(MAX_TEXT_BYTES-len(series)))
    if len(series)!=MAX_TEXT_BYTES:
        raise AssertionError("byte adapter source length drift")
    return series

def pairs(rows):
    return torch.tensor([encode(q,a) for q,a in rows],dtype=torch.long)

def has_candidate(text):
    return bool(text and text.strip())
