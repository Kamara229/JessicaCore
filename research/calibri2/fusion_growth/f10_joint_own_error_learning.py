"""F10: integrated self-error recovery from the actual trained F06 checkpoint.
Use verified own numeric negatives on TRAIN only, learn with paid core and
on-path regions. Compare before and after physical Q8 reload, charge legacy
extras. No protected evaluation or external reference teacher.
"""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f03_rate_train_core import decode,encode,k_for
from research.calibri2.fusion_growth.f06_integrated_recovery import build,forward,evaluate,OLD_PREFIX
from research.calibri2.fusion_growth.f07_own_error_curriculum import generated_curriculum,mine,preference_loss
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,open_bundle,HDR
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import strict_chat_source_ids

PARENT_SHA="7dcc391ce6001f88f6e7c12498c4f371b83431b049e4db2cb65d7f0b89ab722a"
RATE="K16_embed_K8_body"
SEED=20261129
STEPS=120

def native(tok,q,a):
    p=strict_chat_source_ids(tok,q)
    chat=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
    ids=tok.encode(chat+a,add_special_tokens=False)
    assert ids[:len(p)]==p and len(ids)>len(p)
    g=ids[len(p):]+[2]
    assert 2 not in g[:-1] and len(p)+len(g)<160
    return p,g

def own(state,tok,q,cap=8):
    p=strict_chat_source_ids(tok,q);f=prompt_features(tok,q);ids=[]
    for _ in range(cap):
        z=forward(state,p+ids,f).detach()
        ids.append(int(z.argmax()))
        if ids[-1]==2:break
    return ids

def heldout(tok):
    rows=[]
    for i in range(10):
        a=7319+23*i;b=5+i;v=80*(i+7);pct=(10,25,50,20)[i%4]
        for q,y,w in [(f"Чему равна сумма {a} и {b}?",a+b,a-b),
                      (f"Вычисли разность чисел {a} и {b}.",a-b,a+b),
                      (f"Вычисли {21+i} умножить на {b}.",(21+i)*b,21+i+b),
                      (f"Сколько составят {pct} процентов от {v}?",v*pct//100,v-pct),
                      (f"Какое x удовлетворяет x + {b} = {a}?",a-b,a+b)]:
            q+=" Запиши лишь числовой результат."
            p,g=native(tok,q,str(y));_,bad=native(tok,q,str(w))
            div=next(j for j,(x,z) in enumerate(zip(g,bad)) if x!=z)
            rows.append({"q":q,"p":p,"features":prompt_features(tok,q),"g":g,
                         "bad":bad,"div":div,"answer":str(y)})
    assert len(rows)==50
    return rows

def supervised(state,tok,q,answer,negative=None):
    p,g=native(tok,q,answer);f=prompt_features(tok,q)
    total=0.
    div=None;bad=None
    if negative is not None:
        bad=negative
        div=next((j for j,(x,z) in enumerate(zip(g,bad)) if x!=z),None)
    # Every forward sees only the original prompt and past tokens.
    # Give non-EOS content more weight than EOS to avoid empty answers.
    for i,t in enumerate(g):
        z=forward(state,p+g[:i],f)
        w=.25 if t==2 else 1.
        l=w*F.cross_entropy(z,torch.tensor([t]))/max(1,len(g)-1)
        if i==div:
            l=l+.5*preference_loss(z,t,bad[i])
        # Explicit premature-EOS contrast, not a hard-coded generation mask.
        if t!=2:
            l=l+.15*preference_loss(z,t,2,margin=.5)
        assert torch.isfinite(l)
        l.backward();total+=float(l.detach())
    return total

def run(f01,f06,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    raw=(f06/"F06_JOINT.bundle").read_bytes();assert digest(raw)==PARENT_SHA
    archive,side=decode(raw)
    original=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(original[k])==v
    assert all(side[k]==original[k] for k in NAMES[1:3])
    legacy=(f08/"legacy_extras.bundle").read_bytes();assert len(legacy)==137712
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    tr=generated_curriculum("TRAIN",24);dv=heldout(tok)
    assert len(tr)==STEPS and not set(r["prompt"] for r in tr)&set(r["q"] for r in dv)
    out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/NAMES[0]).write_bytes(original[NAMES[0]])
        for k,v in side.items():(root/k).write_bytes(v)
        state=build(root,archive)
        before=evaluate(state,dv,tok)
        mined,audit=mine(tr,tok,lambda q:own(state,tok,q),lambda q,a:native(tok,q,a)[1],PARENT_SHA)
        (out/"F10_TRAIN_OWN_ERRORS.json").write_text(json.dumps(mined,ensure_ascii=False,indent=2))
        print("F10_BEFORE",json.dumps({"before":before,"mine":audit}),flush=True)
        model,sources,bridge,full,expansion,child,region=state
        core=[s.centroids for s in sources.values()]
        opt=torch.optim.Adam([{"params":core,"lr":.00015},{"params":region,"lr":.0003},
                              {"params":[full.gates],"lr":.0001}])
        order=np.random.default_rng(SEED).permutation(STEPS)
        seen=set();regions=set();gates=set();corrections=0;curve=[]
        for i,j in enumerate(order):
            row=mined[int(j)]
            opt.zero_grad(set_to_none=True)
            negative=row["own_native_ids"] if row["negative_is_math_error"] else None
            l=supervised(state,tok,row["prompt"],row["positive_target"],negative)
            if negative is not None:
                l+=supervised(state,tok,row["correction_prompt"],row["correction_target"])
                corrections+=1
            for k,s in sources.items():
                if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(k)
            if full.gates.grad is not None:gates.update(k for k,x in enumerate(full.gates.grad) if float(x.abs())>0)
            for k,p in child.named_parameters():
                if p.grad is not None and float(p.grad.abs().sum())>0:regions.add(k)
            torch.nn.utils.clip_grad_norm_(core+region+[full.gates],1.,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for k,s in sources.items():s.centroids[k_for(RATE,k):].zero_()
            curve.append(l)
            if (i+1)%30==0:print("F10_PROGRESS",i+1,l,len(seen),len(gates),corrections,flush=True)
        assert len(seen)==85 and len(gates)==12 and regions
        new_side=dict(side)
        new_side[NAMES[3]]=save({OLD_PREFIX+k:v.detach().cpu().half().contiguous() for k,v in child.state_dict().items()})
        gate_key=next(iter(load_file(str(root/NAMES[4]))))
        new_side[NAMES[4]]=save({gate_key:full.gates.detach().cpu().half().contiguous()})
        full_packet=encode(archive.blob,sources,RATE,new_side)
        restored,restored_side=decode(full_packet);assert restored_side==new_side
        (out/"F10_FP16.bundle").write_bytes(full_packet)
        # Q8 is the lower-perturbation F09 choice; quantize the newly trained
        # head, not the earlier untrained F09 checkpoint.
        compact,head,_=split_head(new_side,8)
        cp=encode(archive.blob,sources,RATE,compact)
        packed=HDR.pack(b"FG09",len(cp),len(head),len(legacy))+cp+head+legacy
        qcore,qside,qlegacy=open_bundle(packed)
        assert qcore.blob==restored.blob and qlegacy==legacy
        (out/"F10_Q8.bundle").write_bytes(packed)
        del opt,state,model,sources,bridge,full,expansion,child,region,core
        for k,v in restored_side.items():(root/k).write_bytes(v)
        final=build(root,restored);fp16=evaluate(final,dv,tok);del final
        for k,v in qside.items():(root/k).write_bytes(v)
        final=build(root,qcore);q8=evaluate(final,dv,tok);del final
    report={"stage":"F10_INTEGRATED_OWN_ERROR","source_sha256":PARENT_SHA,"depth":12,
            "mining":audit,"before":before,"after_FP16":fp16,"after_Q8":q8,
            "FP16_plus_legacy":metrics(len(full_packet)+len(legacy)),
            "Q8_plus_legacy":metrics(len(packed)),"Q8_sha256":digest(packed),
            "FP16_sha256":digest(full_packet),"steps":STEPS,"correction_updates":corrections,
            "connected_core_codebooks":len(seen),"connected_GT_gates":sorted(gates),
            "connected_region_tensors":sorted(regions),"FP32_teacher_calls":0,
            "candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Single treatment, no matched positive-only control in this round; combined objectives cannot isolate own-error contribution.",
                           "Only verified wrong numeric values used as own negatives; empty/malformed/missing EOS classified, not fabricated mathematical mistakes.",
                           "New 50 internal authored diagnostic prompts; no protected J7 or matched CPU speed.",
                           "Legacy bank included physically but not active; tokenizer and deployment code separate. No functional plateau or specialist branch claim."]}
    (out/"F10_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F10_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("f01","f06","f08","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f06,a.f08,a.out)
