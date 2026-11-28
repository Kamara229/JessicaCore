"""V009 real joint update, exact G44 targets, freeze saturated parallel slot."""
import json,time
from contextlib import ExitStack
import torch
from safetensors.torch import load,save
from .curriculum import epoch_batches,microbatches
from .integrated_step import selected_parameters,optimizer,policy
from .batch_native import native_batch
from .batch_probe import objective
from research.calibri2.fusion_growth.g35_full_mps_preflight import native
from research.calibri2.fusion_growth.g41_balanced_joint import balance_scope
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import parameters
from ..weights.freeze import all_sources

def train_epoch(restored,l6,slot,balance,rows,raw,out):
    st=restored[0];model,sources,_,full,_,child,_=st
    selected,chosen=selected_parameters(st,l6,slot)
    # V008 established zero effective slot update on the tested one-letter task.
    # Preserve slot weights; its EOS objective still backpropagates to head/core.
    for k in list(selected):
        if k.startswith("slot."):selected.pop(k).requires_grad_(False)
    ps=tuple(selected.values());ids={id(p) for p in ps}
    frozen=list({id(p):p for p in parameters((restored,l6,slot,balance)) if id(p) not in ids}.values())
    versions={id(p):p._version for p in frozen}
    init={k:p.detach().clone() for k,p in selected.items()}
    old=load(raw)
    opt=optimizer(selected);visited=set();curve=[];calls=0;exposures=0
    def packet():
        fields=dict(old)
        fields.update({k:p.detach().cpu().half().contiguous() for k,p in selected.items()})
        return save(fields)
    start=time.perf_counter()
    with ExitStack() as stack:
        stats=policy(stack,restored,selected,chosen,True)
        for step,(unique,batch) in enumerate(epoch_batches(rows),1):
            visited.update(unique);exposures+=len(batch)
            opt.zero_grad(set_to_none=True);acc=[torch.zeros_like(p) for p in ps]
            losses=[]
            for rr in microbatches(batch):
                for pos in range(2):
                    pp=[r["p"]+r["g"][:pos] for r in rr];ff=[r["features"] for r in rr]
                    if len(rr)==1:
                        with balance_scope(st,l6,balance,"balanced"):
                            z,z2=native(st,pp[0],ff[0],slot)
                    else:z,z2=native_batch(st,pp,ff,slot,l6,balance)
                    l=objective(z,[r["g"][pos] for r in rr])
                    if pos==0:l=l+objective(z2,[r["g"][1] for r in rr])
                    losses.append(l.detach().sum());calls+=1
                    grads=torch.autograd.grad(l.sum()/(2*len(batch)),ps,allow_unused=True)
                    for i,g in enumerate(grads):
                        if g is not None:acc[i]+=g.detach()
            for p,g in zip(ps,acc):p.grad=g
            norm=torch.nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True)
            opt.step();torch.mps.synchronize()
            assert all(p._version==versions[id(p)] and p.grad is None for p in frozen)
            loss=float(torch.stack(losses).sum()/(2*len(batch)))
            record={"step":step,"loss":loss,"grad_norm":float(norm),"seen":len(visited),"exposures":exposures,"core_calls":calls,"elapsed_seconds":time.perf_counter()-start}
            curve.append(record);print("V009_PROGRESS",json.dumps(record),flush=True)
            if step%4==0:
                (out/f"step_{step}.safetensors").write_bytes(packet())
                (out/"CURVE.json").write_text(json.dumps(curve,indent=2))
        snapshot=dict(stats)
    final=packet();fields=load(final)
    assert len(visited)==512 and exposures==576 and len(curve)==32
    assert all(torch.equal(fields[k],v) for k,v in old.items() if k not in selected)
    changes={}
    for k,p in selected.items():
        group=k.split(".")[0];d=changes.setdefault(group,{"delta_l2_squared":0.,"changed_fp16_values":0})
        d["delta_l2_squared"]+=float((p.detach()-init[k]).square().sum())
        d["changed_fp16_values"]+=int((fields[k]!=old[k]).sum())
    for d in changes.values():d["delta_l2"]=d.pop("delta_l2_squared")**.5
    (out/"v009_candidate.safetensors").write_bytes(final)
    report={"curve":curve,"groups":changes,"slot_frozen":True,"unique_train":len(visited),"exposures":exposures,
            "core_calls":calls,"updates":len(curve),"seconds":time.perf_counter()-start,"cache":snapshot}
    (out/"TRAIN_REPORT.json").write_text(json.dumps(report,indent=2));print("V009_TRAIN",json.dumps({k:v for k,v in report.items() if k!="curve"}),flush=True)
    opt.zero_grad(set_to_none=True)
    return final,report
