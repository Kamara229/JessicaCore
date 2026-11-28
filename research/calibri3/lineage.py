"""Pinned model assembly: no new learned tensors or weights."""
from safetensors.torch import load
import torch
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import Residual,PIN,restore,digest,G05
from research.calibri2.fusion_growth.g41_balanced_joint import Balance
from v176.operators import clear
PARENT="2dc970386a5aa761d184d770e83998abe60f7beb288e3986f03bf99549850556"

def load_parent(g05,f08,parent):
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    model,sources,_,full,_,child,_=st
    raw=(parent/"g43_two_letter_joint.safetensors").read_bytes()
    assert digest(raw)==PARENT
    fields=load(raw);l6=Residual();slot=Residual();balance=Balance()
    for prefix,mod in (("l6.",l6),("slot.",slot),("head.",child.language_head)):
        mod.load_state_dict({k[len(prefix):]:v for k,v in fields.items() if k.startswith(prefix)},strict=True)
    with torch.no_grad():
        for k,v in fields.items():
            if k.startswith("core."):sources[k[5:]].centroids.copy_(v.float())
            elif k=="GT.gates":full.gates.copy_(v.float())
            elif k=="balance.logits":balance.logits.copy_(v.float())
    clear(sources)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    size=sum(map(len,blobs))+len(raw)+sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    return restored, l6, slot, balance, tok, raw, 8*size/G05
