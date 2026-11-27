"""G35: actual scaled G32 core forward AND backward on MPS, fail closed.
One disposable feasibility update only; G34 remains the training lineage.
"""
import argparse,json,os,time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import Residual,install,PIN,restore,digest,parameters
from research.calibri2.fusion_growth.g34_letter_joint import PARENT_HEAD,encode
from research.calibri2.fusion_growth.g32_word_foundation_mps import PARENTS
from research.calibri2.fusion_growth.f06_integrated_recovery import scoped_reduced_native_head_gain
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import single_pass_early_intervention
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import temporary_learned_native_head
from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear

def native_device_mask(self,original12,*,contextual):
    # Same S119m math; only allocate native validity on the input device.
    assert self.output_gate==0 and not self.candidate_accepted and not self.activation_allowed
    assert original12.ndim==4 and original12.shape[0]==1 and original12.shape[1]==12 and original12.shape[-1]==320 and original12.shape[2]>=2
    valid=torch.ones((1,original12.shape[2]),dtype=torch.bool,device=original12.device)
    graph,rel,plan,planned=self.shadow.unified(original12,valid,contextual=contextual)
    graph_parent=self.shadow.unified.nra3_neural_weights(original12,valid)
    adapted=self.shadow.unified.structured_interface(original12[:,0],graph_parent,valid,contextual=contextual)
    structure_delta=adapted[:,-1]-original12[:,0,-1]
    core_delta=self.shadow.unified.core_growth(original12)[:,-1,-1]-original12[:,-1,-1]
    rel6=F.pad(rel[:,-1],(0,314))
    assert structure_delta.shape==core_delta.shape==rel6.shape==planned.shape==(1,320)
    return torch.stack([core_delta,structure_delta+0.01*rel6,planned],dim=1)

def native(st,p,features,slot):
    model,sources,bridge,full,expansion,child,_=st
    device=next(model.parameters()).device
    ids=torch.tensor([p],dtype=torch.long,device=device)
    cap=[]
    handle=model.lm_head.register_forward_pre_hook(lambda _m,args:cap.append(args[0][:,-1:,:]))
    try:
        with torch.enable_grad(),scoped_reduced_native_head_gain(child.language_head),temporary_learned_native_head(model,child.language_head) as audit,expansion.prefill_only(model),s160.prefill_only_tunnel(model,full),bridge.prompt_scope("S93",features),single_pass_early_intervention(model,child,contextual=True) as witness:
            clear(sources)
            try:
                out=model(input_ids=ids,past_key_values=None,use_cache=True,output_hidden_states=False)
                z=out.logits[:,-1,:].float()
                assert out.past_key_values is not None
            finally:clear(sources)
        assert audit["original_head_calls"]==witness["candidate_forward_calls"]==len(cap)==1
        assert full.last_forward_coverage==tuple(range(12))
        with scoped_reduced_native_head_gain(child.language_head):
            s=slot(cap[0]);z2=(model.lm_head(s).float()+child.language_head(s).float())[:,0,:]
        clear(sources)
        assert z.device==z2.device==device and z.shape==z2.shape==(1,49152)
        return z,z2
    finally:handle.remove()

def run(g05,f08,g29,g32,out):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.manual_seed(20270121);torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    model,sources,bridge,full,expansion,child,_=st
    a=(g29/"adapter.safetensors").read_bytes();h=(g32/"word_head_slot.safetensors").read_bytes()
    assert digest(a)==PARENTS["control"] and digest(h)==PARENT_HEAD
    l6=Residual();l6.load_state_dict(load(a),strict=True)
    raw=load(h);slot=Residual();slot.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("slot.")},strict=True)
    child.language_head.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("head.")},strict=True)
    r=encode(tok,"Напиши букву м.","м")
    with install(st,6,l6):
        cpu=[z.detach().clone() for z in native(st,r["p"],r["features"],slot)]
    from types import MethodType
    mid=child.child.child
    assert hasattr(mid,"delta_components")
    mid.delta_components=MethodType(native_device_mask,mid)
    with install(st,6,l6):
        patched=[z.detach() for z in native(st,r["p"],r["features"],slot)]
    assert all(torch.equal(x,y) for x,y in zip(cpu,patched)),"CPU native validity patch changed logits"
    # Off-model GT/bridge/expansion modules and weak-referenced original donor
    # also need movement; module .to alone would miss some research hooks.
    donor=bridge._existing()
    modules=[model,bridge,full,expansion,child,l6,slot]
    if donor is not None:modules.append(donor)
    for bank in restored[1:]:
        if isinstance(bank,dict):modules.extend(v for v in bank.values() if isinstance(v,torch.nn.Module))
        elif isinstance(bank,torch.nn.Module):modules.append(bank)
    # Legacy donor can be alive only through hook defaults, not bridge weakref.
    # Include the exact bound module objects from installed hooks.
    import inspect
    for owner in list(model.modules()):
        for hook in owner._forward_hooks.values():
            for obj in list(getattr(hook,"__defaults__",None) or ()) + list((getattr(hook,"__kwdefaults__",None) or {}).values()):
                if isinstance(obj,torch.nn.Module):modules.append(obj)
            for cell in getattr(hook,"__closure__",None) or ():
                obj=cell.cell_contents
                if isinstance(obj,torch.nn.Module):modules.append(obj)
            bound=getattr(hook,"__self__",None)
            if isinstance(bound,torch.nn.Module):modules.append(bound)
    modules=list({id(m):m for m in modules}.values())
    for m in modules:m.to("mps")
    (out/"DEVICE_MODULES.json").write_text(json.dumps([type(m).__name__ for m in modules]))
    for s in sources.values():s.clear();s.to("mps")
    assert all(p.device.type=="mps" for m in modules for p in m.parameters())
    assert all(b.device.type=="mps" for m in modules for b in m.buffers())
    keys=[f"model.layers.{i}.{k}.weight" for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    groups={"core":[sources[k].centroids for k in keys],"GT":[full.gates],"L6":list(l6.parameters()),"head":list(child.language_head.parameters()),"slot":list(slot.parameters())}
    ps=[p for g in groups.values() for p in g]
    for p in ps:p.requires_grad_(True)
    original=[p.detach().clone() for p in ps]
    opt=torch.optim.AdamW(ps,lr=.0001,foreach=False,weight_decay=0.)
    with install(st,6,l6):
        start=time.perf_counter()
        z,z2=native(st,r["p"],r["features"],slot)
        err=max(float((x.detach().cpu()-y).abs().max()) for x,y in zip((z,z2),cpu))
        # Numerical parity diagnostic: no certification by loose threshold.
        print("G35_MPS_FORWARD",json.dumps({"logit_max_abs_error":err,"cpu_top1":[int(x.argmax()) for x in cpu],"mps_top1":[int(x.argmax()) for x in (z,z2)]}),flush=True)
        assert torch.isfinite(z).all() and torch.isfinite(z2).all()
        loss=F.cross_entropy(z,torch.tensor([r["g"][0]],device="mps"))+.5*F.cross_entropy(z2,torch.tensor([r["g"][1]],device="mps"))
        grads=torch.autograd.grad(loss,ps,allow_unused=True)
        for p,g in zip(ps,grads):
            if g is not None:assert g.device.type=="mps" and torch.isfinite(g).all()
            p.grad=g
        norms={k:sum(float(p.grad.norm()) for p in v if p.grad is not None) for k,v in groups.items()}
        assert all(v>0 for v in norms.values()),norms
        opt.step();torch.mps.synchronize()
        changes={k:sum(float((p-old).norm()) for p,old in zip(ps,original) if any(p is q for q in v)) for k,v in groups.items()}
        assert all(v>0 for v in changes.values()),changes
        clear(sources);seconds=time.perf_counter()-start
    report={"stage":"G35_FULL_MPS_PREFLIGHT","device":"mps","CPU_fallback_disabled":True,"all12_core_forward":True,"max_cpu_gpu_logit_error":err,"gpu_gradient_norms":norms,"gpu_weight_changes":changes,"seconds":seconds,"loss":float(loss.detach()),"retained_updates":0,"limitations":["One disposable update on G32 parent, not competing with G34 training. No new model checkpoint or quality/retention/speed claim.","Full device pass with explicit MPS ids; original hooks retained. CPU reference uses identical device-aware function, not separate legacy runtime comparison."]}
    (out/"G35_REPORT.json").write_text(json.dumps(report,indent=2));print("G35_REPORT",json.dumps(report),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","g29","g32","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.g29,a.g32,a.out)
