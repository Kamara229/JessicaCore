"""V003 matched benchmark on the validated V002 frozen source path."""
from v176.operators import TrainWeight
from ..weights.cache import frozen_cache
from ..weights.freeze import all_sources
from ..weights.architecture import replace_selected
from .probe import probe,compare

def run_cache_probe(restored,slot,rows):
    st=restored[0];model,sources,_,full,_,child,_=st
    keys=[f"model.layers.{i}.{k}.weight" for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    chosen=[sources[k] for k in keys]
    selected={"core."+k:sources[k].centroids for k in keys}
    selected.update({"head."+k:p for k,p in child.language_head.named_parameters()})
    selected.update({"slot."+k:p for k,p in slot.named_parameters()})
    selected["GT.gates"]=full.gates
    for p in selected.values():p.requires_grad_(True)
    allsrc=all_sources(restored)
    cap=640*1024**2
    with replace_selected(chosen) as counts:
        a,base=probe(st,slot,rows,allsrc,selected,True)
        with frozen_cache(TrainWeight,allsrc,byte_cap=cap) as stats:
            b,cached=probe(st,slot,rows,allsrc,selected,True)
            snapshot=dict(stats)
        # V002's inner legacy profiler observes cached forwards as misses because
        # the persistent cache is external. Do not use those incorrect counts.
        cached.pop("profile",None)
    c=compare(a,b,selected)
    gate=c["top1_equal"] and max(c["logits"],c["slot"],c["loss"],*c["gradient_by_tensor"].values())<.002
    assert snapshot["hits"]>0 and snapshot["bypasses"]>0 and snapshot["peak_bytes"]<=cap
    return {"baseline":base,"cached":cached,"cache_four_rounds_including_warmup":snapshot,
            "byte_cap":cap,"cache_released":stats["resident_bytes"]==0,"parity":c,
            "parity_gate":gate,"speed_ratio":base["median_seconds"]/cached["median_seconds"],
            "compact_calls":counts["compact_calls"]}
