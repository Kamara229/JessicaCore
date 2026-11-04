"""Temporary E/B/H factor overrides, allowing untied diagnostic cases.

In normal SmolLM2 the embedding/output head are physically tied. Untying is
ONLY an unpaid FP32/INT6 intervention for locating failure, not a codec.
"""
from contextlib import contextmanager
from v110.protocol import validate_geometry

@contextmanager
def compose(packed, int6, *, embed_int6=False, blocks_int6=(), head_int6=False):
    validate_geometry(len(packed.model.layers))
    ids = tuple(blocks_int6)
    if len(ids) != len(set(ids)) or any(not isinstance(i,int) or i<0 or i>=30 for i in ids):
        raise ValueError("bad layers selection")
    replacements = []
    def replace(obj, name, source):
        old = getattr(obj,name)
        replacements.append((obj,name,old))
        setattr(obj,name,source)
    try:
        if embed_int6:
            replace(packed.model,"embed_tokens",int6.model.embed_tokens)
        if head_int6:
            replace(packed,"lm_head",int6.lm_head)
        for i in ids:
            old=packed.model.layers[i]
            replacements.append((packed.model.layers,i,old))
            packed.model.layers[i]=int6.model.layers[i]
        yield packed
    finally:
        for obj,name,old in reversed(replacements):
            if isinstance(name,int):
                obj[name]=old
            else:
                setattr(obj,name,old)

def arrangements():
    return tuple((e,b,h) for e in (0,1) for b in (0,1) for h in (0,1))
