"""CPU evaluation of exact reloaded physical weights, no training on DEV."""
import torch
from safetensors.torch import load
from ..weights.architecture import replace_selected
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import install
from research.calibri2.fusion_growth.g41_balanced_joint import balance_scope
from research.calibri2.fusion_growth.g44_512_joint import evaluate
from v176.operators import clear

def load_packet(st,l6,slot,balance,packet):
    model,sources,_,full,_,child,_=st;fields=load(packet)
    with torch.no_grad():
        for prefix,mod in (("head.",child.language_head),("slot.",slot),("l6.",l6)):
            mod.load_state_dict({k[len(prefix):]:v for k,v in fields.items() if k.startswith(prefix)},strict=True)
        for k,v in fields.items():
            if k.startswith("core."):sources[k[5:]].centroids.copy_(v.float())
        full.gates.copy_(fields["GT.gates"].float());balance.logits.copy_(fields["balance.logits"].float())
    clear(sources)

def score(st,l6,slot,balance,tok,sets):
    model,sources=st[:2]
    chosen=[sources[f"model.layers.{i}.{k}.weight"] for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    report={};records={}
    # V001 differentiable CPU reference. Do not mix inference-only C into
    # historical gradient-required generation hooks. Preserve full-head shape
    # for the original CPU forward ABI; V005 last-head is training only.
    with install(st,6,l6),balance_scope(st,l6,balance,"balanced"),replace_selected(chosen):
        for name,rows in sets.items():
            report[name],records[name]=evaluate(st,tok,rows,slot)
            print("V009_SCORE",name,report[name],flush=True)
    return report,records
