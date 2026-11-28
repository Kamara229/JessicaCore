"""V002 training subsystem: establish equivalent gradients before adoption."""
from ..weights.freeze import all_sources
from .probe import probe,compare
from ..weights.architecture import replace_selected

def run_probe(restored,slot,rows):
    st=restored[0];model,sources,_,full,_,child,_=st
    keys=[f"model.layers.{i}.{k}.weight" for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    # Keep V001 compact path for the same six operators in both arms.
    chosen=[sources[k] for k in keys]
    selected={"core."+k:sources[k].centroids for k in keys}
    selected.update({"head."+k:p for k,p in child.language_head.named_parameters()})
    selected.update({"slot."+k:p for k,p in slot.named_parameters()})
    selected["GT.gates"]=full.gates
    for p in selected.values():p.requires_grad_(True)
    allsrc=all_sources(restored)
    with replace_selected(chosen) as counts:
        a,legacy=probe(st,slot,rows,allsrc,selected,False)
        b,frozen=probe(st,slot,rows,allsrc,selected,True)
    c=compare(a,b,selected)
    gate=c["top1_equal"] and max(c["logits"],c["slot"],c["loss"],*c["gradient_by_tensor"].values())<.002
    return {"baseline":legacy,"explicit_freeze":frozen,"parity":c,
            "parity_gate":gate,"speed_ratio":legacy["median_seconds"]/frozen["median_seconds"],
            "compact_calls":counts["compact_calls"]}
