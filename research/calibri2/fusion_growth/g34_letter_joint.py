"""G34: strict alphabet-first joint continuation from actual G32 Control.
CPU native forward/backprop + MPS AdamW on compact trainable tensors.
No inference mask, no forced EOS, no new external selector, no word training
until a later independently checked alphabet gate. Matched head-only control.
"""
import argparse,json,os,random
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load,save
from transformers import AutoTokenizer
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import Residual,install,parameters,PIN,restore,digest,forward,G05,block
from research.calibri2.fusion_growth.g28_l6_continuation import ByteValidity
from research.calibri2.fusion_growth.g32_word_foundation_mps import LETTERS,row,PARENTS
from research.calibri2.fusion_growth.f21_native_cpu_integration import prompt_features,strict_chat_source_ids
from v176.operators import clear
PARENT_HEAD="fe27bc3e7b9b6a5fb0c6fb0371d52ea7607833f3a681e96d04d70aa3396e730f"
SEED=20270120
EPOCHS=6
BATCH=6

def encode(tok,q,a):
    p=strict_chat_source_ids(tok,q)
    chat=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    ids=tok.encode(chat+a,add_special_tokens=False);g=ids[len(p):]+[2]
    assert ids[:len(p)]==p and 2 not in g[:-1] and len(p)+len(g)<160 and tok.decode(g[:-1],skip_special_tokens=True).strip()==a
    return {"q":q,"answer":a,"p":p,"g":g,"features":prompt_features(tok,q)}

def suite(tok):
    train=[];dev=[]
    for a in LETTERS:
        for template in ('Повтори букву «{}». Напиши только букву.','Напиши букву {}.'):
            train.append(encode(tok,template.format(a),a))
        for template in ('Запиши без изменений: {}.','Выведи один символ: {}.'):
            dev.append(encode(tok,template.format(a),a))
    assert len(train)==len(dev)==66 and not {r["q"] for r in train}&{r["q"] for r in dev}
    return train,dev

def evaluate(st,tok,rows,slot):
    totals={};records=[]
    for mode in ("sequential","parallel"):
        d={"n":len(rows),"exact_EOS":0,"correct_text":0,"EOS":0,"invalid":0,"empty":0,"first_correct":0,"overrun":0}
        for r in rows:
            ids=[];cap=8
            while len(ids)<cap:
                if mode=="sequential":
                    z=forward(st,r["p"]+ids,r["features"]).detach();ts=[int(z.argmax())]
                else:
                    z,_=block(st,r["p"]+ids,r["features"],slot);ts=z.detach().argmax(-1)[0].tolist()
                for t in ts:
                    if len(ids)>=cap:break
                    ids.append(t)
                    if t==2:break
                if ids[-1]==2:break
            a=tok.decode(ids,skip_special_tokens=True).strip()
            for k,v in {"exact_EOS":a==r["answer"] and ids[-1]==2,"correct_text":a==r["answer"],"EOS":ids[-1]==2,"invalid":"\ufffd" in a,"empty":not a,"first_correct":ids[0]==r["g"][0],"overrun":len(ids)==cap and ids[-1]!=2}.items():d[k]+=int(v)
            records.append({"mode":mode,"q":r["q"],"expected":r["answer"],"answer":a,"ids":ids})
        totals[mode]=d
    return totals,records

def run(g05,f08,g29,g32,out,arm):
    assert torch.backends.mps.is_available() and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0"
    torch.set_num_threads(2);torch.manual_seed(SEED)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    utf=ByteValidity(tok);train,dev=suite(tok)
    for r in train:r["states"]=utf.states(r)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    restored=restore(blobs[:6],blobs[6],blobs[7]);st=restored[0]
    model,sources,_,full,_,child,_=st
    a=(g29/"adapter.safetensors").read_bytes();h=(g32/"word_head_slot.safetensors").read_bytes()
    assert digest(a)==PARENTS["control"] and digest(h)==PARENT_HEAD
    l6=Residual();l6.load_state_dict(load(a),strict=True)
    raw=load(h);slot=Residual();slot.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("slot.")},strict=True)
    child.language_head.load_state_dict({k[5:]:v for k,v in raw.items() if k.startswith("head.")},strict=True)
    # Explicit compact existing codebooks only; no dense trainable copy.
    keys=[f"model.layers.{i}.{k}.weight" for i in (5,6) for k in ("self_attn.v_proj","self_attn.o_proj","mlp.down_proj")]
    assert all(k in sources for k in keys)
    selected={"head."+k:p for k,p in child.language_head.named_parameters()}|{"slot."+k:p for k,p in slot.named_parameters()}
    if arm=="joint":
        selected|={"l6."+k:p for k,p in l6.named_parameters()}
        selected|={"core."+k:sources[k].centroids for k in keys}
        selected["GT.gates"]=full.gates
    for p in selected.values():p.requires_grad_(True)
    frozen=list({id(p):p for p in list(parameters(restored))+list(l6.parameters())+list(slot.parameters()) if id(p) not in {id(v) for v in selected.values()}}.values())
    versions={id(p):p._version for p in frozen}
    for p in frozen:p.grad=None
    # GPU optimizer mirrors only compact selected parameters. CPU autograd is
    # the real native forward/backward; this is NOT full-core GPU execution.
    mirror={k:torch.nn.Parameter(p.detach().to("mps").clone()) for k,p in selected.items()}
    lr=lambda k:.0008 if k.startswith(("head.","slot.")) else .0002
    opt=torch.optim.AdamW([{"params":[p],"lr":lr(k)} for k,p in mirror.items()],foreach=False,weight_decay=0.)
    groups={k:("core" if k.startswith("core.") else k.split(".")[0]) for k in selected}
    out.mkdir(parents=True,exist_ok=False)
    init={k:p.detach().clone() for k,p in selected.items()}
    # Contrast against competing letter tokens at the same causal prefix.
    all_targets={a:encode(tok,"Напиши букву "+a+".",a)["g"] for a in LETTERS}
    rng=random.Random(SEED);curve=[];seen=set();steps=0
    with install(st,6,l6):
        before,items=evaluate(st,tok,dev,slot)
        (out/"PARENT_DEV.json").write_text(json.dumps(items,ensure_ascii=False))
        print("G34_PARENT",arm,json.dumps(before),flush=True)
        for ep in range(1,EPOCHS+1):
            order=list(range(len(train)));rng.shuffle(order);total=0.;tokens=0
            for j in range(0,len(order),BATCH):
                batch=order[j:j+BATCH];opt.zero_grad(set_to_none=True)
                acc={k:torch.zeros_like(p) for k,p in selected.items()}
                for i in batch:
                    r=train[i];g=r["g"]
                    for pos,t in enumerate(g):
                        zz,_=block(st,r["p"]+g[:pos],r["features"],slot)
                        z=zz[:,0];y=torch.tensor([t]);terminal=t==2
                        # Balance EOS and character prediction per letter, rather
                        # than simply forbidding EOS. At EOS teach stop explicitly.
                        ce=F.cross_entropy(z,y)
                        v=utf.loss(z,r["states"][pos])
                        if terminal:
                            maxother=z[:,torch.arange(z.shape[-1])!=2].max()
                            stop=F.softplus(1.+maxother-z[0,2])
                        else:
                            stop=F.softplus(1.+z[0,2]-z[0,t])
                        alternatives=sorted({s[pos] for s in all_targets.values() if len(s)>pos and s[:pos]==g[:pos] and s[pos]!=t and s[pos]!=2})
                        contrast=F.softplus(1.+z[0,alternatives]-z[0,t]).mean() if alternatives and not terminal else z.sum()*0
                        loss=ce+.2*v+.4*stop+.4*contrast
                        if pos+1<len(g):
                            z2=zz[:,1];t2=g[pos+1]
                            loss=loss+.5*F.cross_entropy(z2,torch.tensor([t2]))+.1*utf.loss(z2,r["states"][pos+1])
                        total+=float(loss.detach());tokens+=1
                        grads=torch.autograd.grad(loss/(len(batch)*len(g)),tuple(selected.values()),allow_unused=True)
                        for (k,p),grad in zip(selected.items(),grads):
                            if grad is not None:
                                assert torch.isfinite(grad).all()
                                acc[k]+=grad.detach()
                                if float(grad.abs().sum())>0:seen.add(groups[k])
                for k,p in mirror.items():p.grad=acc[k].to("mps");assert p.grad.device.type=="mps"
                torch.nn.utils.clip_grad_norm_(list(mirror.values()),1.,error_if_nonfinite=True)
                opt.step();torch.mps.synchronize()
                with torch.no_grad():
                    for k,p in selected.items():p.copy_(mirror[k].detach().cpu())
                clear(sources);steps+=1
                assert all(p.grad is None and p._version==versions[id(p)] for p in frozen)
            # Serialize all current selected weights each epoch; same-head slot
            # and L6 are replaced, never charged twice as two active models.
            payload={k:p.detach().half().contiguous() for k,p in selected.items()}
            for prefix,mod in (("head.",child.language_head),("slot.",slot),("l6.",l6)):
                for k,p in mod.state_dict().items():payload[prefix+k]=p.detach().cpu().half().contiguous()
            packet=save(payload);(out/f"epoch_{ep}.safetensors").write_bytes(packet)
            record={"epoch":ep,"loss":total/tokens,"updates":steps,"nonzero_gradient_groups":sorted(seen)}
            curve.append(record);print("G34_EPOCH",arm,json.dumps(record),flush=True)
            if ep in (2,4,6):
                score,items=evaluate(st,tok,dev,slot)
                (out/f"epoch_{ep}_DEV.json").write_text(json.dumps(items,ensure_ascii=False))
                print("G34_DEV",arm,ep,json.dumps(score),flush=True)
        assert {"head","slot"}<=seen and (arm!="joint" or {"core","GT","l6"}<=seen),seen
        final=(out/f"epoch_{EPOCHS}.safetensors").read_bytes();fields=load(final)
        with torch.no_grad():
            for k,p in selected.items():p.copy_(fields[k].float())
        for prefix,mod in (("head.",child.language_head),("slot.",slot),("l6.",l6)):
            mod.load_state_dict({k[len(prefix):]:v for k,v in fields.items() if k.startswith(prefix)},strict=True)
        clear(sources);after,items=evaluate(st,tok,dev,slot)
    (out/"FINAL_DEV.json").write_text(json.dumps(items,ensure_ascii=False))
    (out/"letter_joint.safetensors").write_bytes(final)
    size=sum(map(len,blobs))+sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())+len(final)
    bpp=8*size/G05;assert bpp<=.7
    report={"arm":arm,"parent_head_sha":PARENT_HEAD,"checkpoint_sha":digest(final),"before":before,"after":after,"curve":curve,"train_letters":33,"train_prompts":66,"dev_prompts":66,"known_payload_bpp":bpp,"nonzero_gradient_groups":sorted(seen),"alphabet_gate":all(d["exact_EOS"]==66 for d in after.values()),"candidate_accepted":False,"limitations":["CPU core forward and autograd, MPS compact parameter optimizer only; no full-core GPU claim.","Letter identities shared across train/dev, prompt templates disjoint; no unseen letter possible. Reused exploratory DEV, no independent confirmation.","No word or sentence training in this stage, no historical J7 or fixed-work CPU certification. All previous checkpoints retained."]}
    (out/"G34_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2));print("G34_REPORT",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","g29","g32","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--arm",choices=("joint","head"),required=True)
    a=p.parse_args();run(a.g05,a.f08,a.g29,a.g32,a.out,a.arm)
