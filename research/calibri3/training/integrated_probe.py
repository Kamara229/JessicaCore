"""V008 matched benchmark and disposable real-update parity; no promoted weights."""
import json
from contextlib import ExitStack
from time import perf_counter
import torch
from ..weights.architecture import error
from ..weights.freeze import all_sources
from .batch_probe import select_rows
from .integrated_step import selected_parameters,optimizer,reset,step,policy

def run_probe(restored,l6,slot,balance,rows):
    st=restored[0];rows=select_rows(rows)
    selected,chosen=selected_parameters(st,l6,slot)
    allsrc=all_sources(restored);initial={k:p.detach().clone() for k,p in selected.items()}
    ref=None;arms={}
    # Warm each arm then alternate order to reduce systematic runner drift.
    timings={"original":[],"integrated":[]};observations={}
    for trial in range(4):
        order=("original","integrated") if trial%2==0 else ("integrated","original")
        for name in order:
            optimized=name=="integrated";reset(selected,initial,allsrc);opt=optimizer(selected)
            with ExitStack() as stack:
                stats=policy(stack,restored,selected,chosen,optimized)
                # Warm cache and execution with one disposable update; reset exactly.
                step(st,l6,slot,balance,rows,selected,opt,4 if optimized else 1,not optimized)
                reset(selected,initial,allsrc);opt=optimizer(selected)
                torch.mps.synchronize();start=perf_counter()
                z,l,g,calls=step(st,l6,slot,balance,rows,selected,opt,4 if optimized else 1,not optimized)
                elapsed=perf_counter()-start
                if trial:timings[name].append(elapsed)
                snapshot=dict(stats) if stats is not None else None
            updates={k:(p.detach()-initial[k]).cpu() for k,p in selected.items()}
            # Preserve actual FP16 serialization effect, not only gradient presence.
            physical={k:int((p.detach().half()!=initial[k].half()).sum()) for k,p in selected.items()}
            observations[name]=(z.cpu(),l.cpu(),[v.cpu() for v in g],updates,physical,calls,snapshot)
            reset(selected,initial,allsrc)
            print("V008_STEP",name,trial,elapsed,flush=True)
    a,b=observations["original"],observations["integrated"]
    def groups(obs):
        out={}
        for k,v in obs[3].items():
            group=k.split(".")[0]
            d=out.setdefault(group,{"delta_l2":0.,"changed_fp16_values":0,"max_abs_delta":0.})
            d["delta_l2"]+=float(v.square().sum());d["changed_fp16_values"]+=obs[4][k]
            d["max_abs_delta"]=max(d["max_abs_delta"],float(v.abs().max()))
        for d in out.values():d["delta_l2"]=d["delta_l2"]**.5
        return out
    for name,obs in observations.items():
        arms[name]={"seconds":timings[name],"median_seconds":sorted(timings[name])[1],"core_calls":obs[5],"groups":groups(obs),"cache":obs[6]}
    parity={"logits":error(b[0],a[0]),"loss":error(b[1],a[1]),"top1_equal":bool(torch.equal(a[0].argmax(-1),b[0].argmax(-1))),
            "gradients":{k:error(y,x) for k,x,y in zip(selected,a[2],b[2])},
            "updates":{k:error(b[3][k],a[3][k]) for k in selected}}
    # Near-zero Adam updates can have poor relative ratios; report absolute errors too.
    gate=parity["top1_equal"] and max(parity["logits"]["relative_l2"],parity["loss"]["relative_l2"],*(v["relative_l2"] for v in parity["gradients"].values()))<.002 and max(v["max_abs"] for v in parity["updates"].values())<.00002
    assert all(torch.equal(p.detach(),initial[k]) for k,p in selected.items())
    return {"arms":arms,"parity":parity,"update_parity_gate":gate,
            "speed_ratio":arms["original"]["median_seconds"]/arms["integrated"]["median_seconds"],
            "rows":[{"q":r["q"],"answer":r["answer"]} for r in rows],"disposable_updates":16,"weights_restored":True}
