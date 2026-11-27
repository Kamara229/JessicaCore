"""G28: preserve actual G27b L6 weights and compare matched continuation.
UTF8 validity is a TRAIN-only differentiable loss, never an inference mask.
"""
import argparse,codecs,json,random
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import save,load
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from research.calibri2.fusion_growth.g27b_internal_residual_training import (
    Residual,install,parameters,PIN,restore,digest,forward,select,paired_targets,audit,G05)
PARENT_SHA="355db635b2b95bd9f13014e81a67379a40246c2e787ee28768fd41d47b7cf13f"
SEED=20270114
EPOCHS=2
BATCH=8
class ByteValidity:
    def __init__(self,tok):
        assert json.loads(tok.backend_tokenizer.to_str())["decoder"]["type"]=="ByteLevel"
        inv={v:k for k,v in bytes_to_unicode().items()}
        special=set(tok.all_special_ids)
        self.raw=[]
        for i in range(len(tok)):
            s=tok.convert_ids_to_tokens(i)
            self.raw.append(bytes(inv[c] for c in s) if i not in special and s and all(c in inv for c in s) else None)
        self.cache={}
        assert self.raw[155]==bytes([0xd0])
        assert 2 in special
    def advance(self,state,t):
        if t==2:return b"" if not state else None
        raw=self.raw[t]
        if raw is None:return None
        d=codecs.getincrementaldecoder("utf-8")("strict")
        try:
            d.setstate((state,0))
            d.decode(raw,final=False)
            return d.getstate()[0]
        except UnicodeDecodeError:return None
    def legal(self,state):
        if state not in self.cache:
            self.cache[state]=torch.tensor([self.advance(state,i) is not None for i in range(len(self.raw))],dtype=torch.bool)
            assert self.cache[state].any()
        return self.cache[state]
    def loss(self,z,state):
        valid=self.legal(state)
        assert z.shape[-1]==len(valid)
        return (torch.logsumexp(z.float(),dim=-1)-torch.logsumexp(z.float()[...,valid],dim=-1)).mean()
    def states(self,r):
        state=b"";states=[]
        for t in r["g"]:
            states.append(state)
            state=self.advance(state,t)
            assert state is not None,(r["q"],t)
        assert state==b"" and r["g"][-1]==2
        return states

def run(g05,f08,parent,out,arm):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    train,dev,rejected=select(tok)
    assert len(train)==136 and len(dev)==80 and sum(len(r["g"]) for r in train)==2413
    assert all(r["split"]=="TRAIN" for r in train)
    packet=(parent/"adapter.safetensors").read_bytes()
    assert digest(packet)==PARENT_SHA
    a=Residual();a.load_state_dict(load(packet),strict=True)
    out.mkdir(parents=True,exist_ok=False)
    neg=paired_targets(train)
    utf=ByteValidity(tok)
    gold_states=[utf.states(r) for r in train]
    frozen=list({id(p):p for p in parameters(restored)}.values())
    for p in frozen:p.grad=None
    versions={id(p):p._version for p in frozen}
    def unchanged():
        assert all(p.grad is None and p._version==versions[id(p)] for p in frozen)
    params=(a.up.weight,a.down.weight)
    opt=torch.optim.AdamW(params,lr=.0003,weight_decay=.0001)
    baseline=audit(st,tok,dev,install(st,6,a),"G27B_L6_PARENT",out)
    assert baseline["replacement_char"]==76 and baseline["different_outputs"]==15 and baseline["exact_EOS"]==0,baseline
    rng=random.Random(SEED);curve=[];steps=0
    def accumulate(loss):
        assert torch.isfinite(loss)
        for p,g in zip(params,torch.autograd.grad(loss,params)):
            assert torch.isfinite(g).all()
            if p.grad is None:p.grad=g.detach().clone()
            else:p.grad.add_(g.detach())
    with install(st,6,a):
        for epoch in range(1,EPOCHS+1):
            order=list(range(len(train)));rng.shuffle(order)
            ce_sum=0.;utf_sum=0.;bad_rollouts=0;probes=0
            for j in range(0,len(order),BATCH):
                batch=order[j:j+BATCH];opt.zero_grad(set_to_none=True)
                for i in batch:
                    r=train[i]
                    for k,t in enumerate(r["g"]):
                        z=forward(st,r["p"]+r["g"][:k],r["features"])
                        ce=F.cross_entropy(z,torch.tensor([t]))/len(r["g"])
                        if (i,k) in neg:ce=ce+.2*F.softplus(.5-(z[0,t]-z[0,neg[i,k]]))
                        v=utf.loss(z,gold_states[i][k])/len(r["g"])
                        ce_sum+=float(ce.detach());utf_sum+=float(v.detach())
                        accumulate((ce+(.3*v if arm=="utf8" else 0.))/len(batch))
                # Same predetermined TRAIN probe schedule and greedy native path in both arms.
                r=train[batch[0]];ids=[];state=b"";probes+=1
                for _ in range(min(16,159-len(r["p"]))):
                    z=forward(st,r["p"]+ids,r["features"])
                    t=int(z.detach().argmax())
                    ns=utf.advance(state,t)
                    if ns is None:
                        bad_rollouts+=1
                        if arm=="utf8":accumulate(.5*utf.loss(z,state)/len(batch))
                        break
                    if t==2:break
                    ids.append(t);state=ns
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                opt.step();steps+=1;unchanged()
                if steps%8==0:print("G28_PROGRESS",arm,epoch,steps,flush=True)
            item={"epoch":epoch,"objective_ce_contrast":ce_sum/len(train),"utf8_nll":utf_sum/len(train),"illegal_rollouts":bad_rollouts,"train_rollout_probes":probes}
            curve.append(item)
            print("G28_EPOCH",arm,json.dumps(item),flush=True)
            (out/"progress.json").write_text(json.dumps(curve))
            (out/("epoch"+str(epoch)+".safetensors")).write_bytes(save({k:v.detach().cpu().half().contiguous() for k,v in a.state_dict().items()},metadata={"layer":"6","parent":PARENT_SHA,"arm":arm}))
    final=(out/("epoch"+str(EPOCHS)+".safetensors")).read_bytes()
    (out/"adapter.safetensors").write_bytes(final)
    reloaded=Residual();reloaded.load_state_dict(load(final),strict=True);reloaded.eval()
    size=sum(map(len,blobs))+sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())+len(final)
    bpp=8*size/G05;assert bpp<=.7
    score=audit(st,tok,dev,install(st,6,reloaded),"L6_"+arm,out);unchanged()
    report={"arm":arm,"parent_sha256":PARENT_SHA,"adapter_sha256":digest(final),"parent":baseline,"score":score,"curve":curve,"steps":steps,"train_rows":len(train),"known_payload_bpp":bpp,"payload_bytes":size,"frozen_tensors":len(frozen),"candidate_accepted":False,"limitations":["Reused exploratory DEV, not independent holdout; no protected 616 skill or fixed-work CPU certification.","UTF8 validity does not establish semantics. No inference masking, no decoder changes.","Same L6 warm start, fresh matched optimizer; only existing 5120 adapter weights trained."]}
    (out/"G28_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G28_REPORT",json.dumps(report),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","parent","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--arm",choices=("control","utf8"),required=True)
    a=p.parse_args();run(a.g05,a.f08,a.parent,a.out,a.arm)
