"""G27a: actual native gradient/identity preflight, no retained training.
Compare rank8 residual location 6 vs 12 on pinned G05.
"""
import argparse,json
from pathlib import Path
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g23_original_vs_scaled import PIN,restore,digest,forward
from research.calibri2.fusion_growth.g21_g18_context_attention import ROOT,data,prompt_features
from research.calibri2.fusion_growth.f21_native_cpu_integration import strict_chat_source_ids

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.Linear(320,8,bias=False)
        self.up=nn.Linear(8,320,bias=False)
        nn.init.normal_(self.down.weight,std=.02)
        nn.init.zeros_(self.up.weight)
    def forward(self,h):
        return h+self.up(F.gelu(self.down(F.layer_norm(h.float(),(320,))))).to(h.dtype)

@contextmanager
def install(st,layer,adapter):
    def hook(_m,_a,out):
        return (adapter(out[0]),*out[1:]) if isinstance(out,tuple) else adapter(out)
    handle=st[0].model.layers[layer-1].register_forward_hook(hook)
    try:yield
    finally:handle.remove()

def parameters(obj,seen=None):
    if seen is None:seen=set()
    if id(obj) in seen:return
    seen.add(id(obj))
    if isinstance(obj,nn.Parameter):yield obj
    elif isinstance(obj,nn.Module):
        yield from obj.parameters()
    elif isinstance(obj,dict):
        for v in obj.values():yield from parameters(v,seen)
    elif isinstance(obj,(tuple,list)):
        for v in obj:yield from parameters(v,seen)

def examples(tok):
    rows=[]
    for line in (ROOT/"TRAIN.jsonl").read_text(encoding="utf8").splitlines():
        r=json.loads(line);assert r["split"]=="TRAIN"
        p=strict_chat_source_ids(tok,r["q"])
        chat=tok.apply_chat_template([{"role":"user","content":r["q"]}],tokenize=False,add_generation_prompt=True)
        ids=tok.encode(chat+r["answer"],add_special_tokens=False)
        g=ids[len(p):]+[2]
        if ids[:len(p)]==p and len(g)>1 and 2 not in g[:-1] and len(p)+len(g)<160 and tok.decode(g[:-1],skip_special_tokens=True).strip()==r["answer"]:
            rows.append({**r,"p":p,"g":g,"features":prompt_features(tok,r["q"]),"cohort":"G19"})
            break
    for v in data(tok,"TRAIN").values():
        for r in v:
            if r["split"]=="TRAIN" and len(r["p"])+len(r["g"])<160 and len(r["g"])>1:
                rows.append({**r,"cohort":"G16"});break
        if len(rows)==2:break
    assert len(rows)==2 and {r["cohort"] for r in rows}=={"G19","G16"}
    return rows

def run(g05,f08,out):
    torch.manual_seed(20270113);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    rows=examples(tok);out.mkdir(parents=True,exist_ok=False)
    r=rows[0];ref=forward(st,r["p"],r["features"]).detach().clone()
    frozen=list({id(p):p for p in parameters(restored)}.values())
    assert frozen
    for p in frozen:p.grad=None
    # Existing native path requires differentiable child. Preserve its flags;
    # request gradients only for adapter, never update the existing parameters.
    versions={id(p):p._version for p in frozen}
    report={}
    for layer in (6,12):
        torch.manual_seed(20270113);a=Residual();assert sum(p.numel() for p in a.parameters())==5120
        opt=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=.0001)
        with install(st,layer,a):
            z=forward(st,r["p"],r["features"])
            err=float((z.detach()-ref).abs().max());assert err<1e-4,err
            opt.zero_grad(set_to_none=True)
            a.up.weight.grad, a.down.weight.grad = torch.autograd.grad(F.cross_entropy(z,torch.tensor([r["g"][0]])),(a.up.weight,a.down.weight))
            up=float(a.up.weight.grad.norm());assert up>0 and torch.isfinite(a.up.weight.grad).all()
            opt.step();opt.zero_grad(set_to_none=True)
            s=rows[1];j=min(1,len(s["g"])-1)
            z=forward(st,s["p"]+s["g"][:j],s["features"])
            a.up.weight.grad, a.down.weight.grad = torch.autograd.grad(F.cross_entropy(z,torch.tensor([s["g"][j]])),(a.up.weight,a.down.weight))
            down=float(a.down.weight.grad.norm());assert down>0 and torch.isfinite(a.down.weight.grad).all()
            assert all(p.grad is None and p._version==versions[id(p)] for p in frozen)
        report[str(layer)]={"identity_max_error":err,"up_grad_norm":up,"down_grad_norm_after_ephemeral_step":down,"frozen_parameters":len(frozen),"adapter_parameters":5120}
        print("G27_PREFLIGHT",json.dumps(report[str(layer)]),flush=True)
        (out/"G27_PREFLIGHT.json").write_text(json.dumps({"layers":report,"retained_training_updates":0,"candidate_accepted":False,"limitations":["Two TRAIN examples only, one discarded update per location; no generation quality, J7 or CPU certification."]},indent=2))
    print("G27_PREFLIGHT_PASS",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
