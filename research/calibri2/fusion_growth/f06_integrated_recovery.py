"""F06: attach all existing learned regions to a physically reloaded mixed core.
Train paid core, existing on-path regions and GT gates on verified arithmetic
contrasts, with original same-region baseline. Research only.
"""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,metrics,digest,unpack
from research.calibri2.fusion_growth.f03_rate_train_core import decode,encode,k_for
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import load_actual_original,NativeFullHeadShadow
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import LearnedOnPathInterlayer
from research.calibri_nra3.s119q2f_learned_causal_early_projection import LearnedCausalEarlyProjector
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import JointRuNative,real_joint_native
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import set_gt_route
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import make_joint_trainable,scoped_reduced_native_head_gain
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import strict_chat_source_ids
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import OLD_PREFIX
from direct_codebook_v59 import unpack5
from v176.operators import clear

PARENT_SHA="9e710de6db41d6bb2a99776080f045c537d4af2f6e971529ba6527bf993828d4"
RATE="K16_embed_K8_body"
SEED=20261128
STEPS=64

def curriculum(tok,split):
    n=16 if split=="TRAIN" else 8
    offset=410 if split=="TRAIN" else 810
    data=[]
    for i in range(n):
        a=offset+i*6;b=2+i%7
        for j,(q,y,w) in enumerate([
            (f"Прибавь {b} к {a}.",a+b,a-b),
            (f"Из числа {a} вычти {b}.",a-b,a+b),
            (f"Умножь {b} на {i+11}.",b*(i+11),b+(i+11)),
            (f"Реши уравнение x - {b} = {a}.",a+b,a-b)]):
            if split=="DEV":q="Математическая задача: "+q
            q+=" Напиши только число."
            assert y!=w
            p=strict_chat_source_ids(tok,q)
            text=tok.apply_chat_template([{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)
            g=tok.encode(text+str(y),add_special_tokens=False)
            bad=tok.encode(text+str(w),add_special_tokens=False)
            assert g[:len(p)]==p and bad[:len(p)]==p
            g=g[len(p):]+[2];bad=bad[len(p):]+[2]
            div=next(k for k,(x,z) in enumerate(zip(g,bad)) if x!=z)
            assert len(p)+max(len(g),len(bad))+8<160
            data.append({"q":q,"p":p,"features":prompt_features(tok,q),
                         "g":g,"bad":bad,"div":div,"answer":str(y)})
    return data

def build(root,archive=None,learned_side=None):
    # The original audited loader builds every real hook. Replace only the
    # paid CB59 fields from the compact checkpoint after source verification.
    model,sources,bridge,full,expansion=load_actual_original(*(root/n for n in NAMES[:3]))
    if archive is not None:
        assert set(sources)=={k for k,v in archive.meta["tensor_descriptors"].items() if v["dtype"]=="CB59_K32V8"}
        with torch.no_grad():
            for k,s in sources.items():
                h=archive.handles[k]
                assert np.array_equal(h.row_scales,s.base_scales.numpy())
                s.centroids.copy_(torch.from_numpy(h.centroids.copy()))
                s.codes.copy_(torch.from_numpy(unpack5(h.packed.tobytes(),s.codes.numel()).astype(np.int64)))
        clear(sources)
    child=JointRuNative(LearnedCausalEarlyProjector(LearnedOnPathInterlayer(NativeFullHeadShadow())))
    raw=load_file(str(root/NAMES[3]))
    assert all(k.startswith(OLD_PREFIX) for k in raw) and sum(v.numel() for v in raw.values())==421793
    child.load_state_dict({k[len(OLD_PREFIX):]:v.float() for k,v in raw.items()},strict=True)
    child.eval()
    params=make_joint_trainable(child)
    model.model.add_module("nra3_s119m_onpath",child)
    gates=load_file(str(root/NAMES[4]))
    assert len(gates)==1
    gate=next(iter(gates.values())).float()
    assert gate.shape==(12,)
    set_gt_route(full,"GT_RECURRENT",gate)
    full.gates.requires_grad_(True)
    for s in sources.values():s.centroids.requires_grad_(True)
    return model,sources,bridge,full,expansion,child,params

def forward(state,p,features):
    model,sources,bridge,full,expansion,child,_=state
    with scoped_reduced_native_head_gain(child.language_head):
        z=real_joint_native(model,sources,bridge,full,expansion,child,p,features,contextual=True)
    assert full.last_forward_coverage==tuple(range(12))
    return z

def evaluate(state,rows,tok):
    nll=[];gap=[];exact=0;eos=0;empty=0
    for row in rows:
        vals=[]
        for i,t in enumerate(row["g"]):
            z=forward(state,row["p"]+row["g"][:i],row["features"]).detach()
            vals.append(float(F.cross_entropy(z,torch.tensor([t]))))
            if i==row["div"]:gap.append(float(z[0,t]-z[0,row["bad"][i]]))
        nll.append(sum(vals)/len(vals))
        own=[]
        for _ in range(8):
            z=forward(state,row["p"]+own,row["features"]).detach()
            own.append(int(z.argmax()))
            if own[-1]==2:break
        text=tok.decode(own,skip_special_tokens=True).strip()
        eos+=int(own[-1]==2);empty+=int(not text)
        exact+=int(own[-1]==2 and text==row["answer"])
    return {"n":len(rows),"NLL":sum(nll)/len(nll),"positive_pair_rank":sum(x>0 for x in gap),
            "mean_pair_gap":sum(gap)/len(gap),"greedy_exact_EOS":exact,"EOS":eos,"empty":empty}

def run(f01,f03,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    files=unpack((f01/"F01.xz.bundle").read_bytes())
    packet=(f03/"K16_embed_K8_body.bundle").read_bytes()
    assert digest(packet)==PARENT_SHA
    archive,side=decode(packet)
    assert side=={k:files[k] for k in NAMES[1:]}
    for k,v in PIN.items():assert digest(files[k])==v
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    tr=curriculum(tok,"TRAIN");dv=curriculum(tok,"DEV")
    assert len(tr)==STEPS and len(dv)==32
    out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        for k,v in files.items():(root/k).write_bytes(v)
        original=build(root)
        baseline=evaluate(original,dv,tok);del original
        print("F06_ORIGINAL",json.dumps(baseline),flush=True)
        state=build(root,archive)
        before=evaluate(state,dv,tok)
        print("F06_MIXED_BEFORE",json.dumps(before),flush=True)
        model,sources,bridge,full,expansion,child,region=state
        core=[s.centroids for s in sources.values()]
        opt=torch.optim.Adam([{"params":core,"lr":.00015},
                             {"params":region,"lr":.0003},
                             {"params":[full.gates],"lr":.0001}])
        seen=set();gates=set();regions=set();curve=[]
        order=np.random.default_rng(SEED).permutation(STEPS)
        for i,j in enumerate(order):
            row=tr[int(j)]
            opt.zero_grad(set_to_none=True)
            val=0.
            for pos,t in enumerate(row["g"]):
                z=forward(state,row["p"]+row["g"][:pos],row["features"])
                loss=F.cross_entropy(z,torch.tensor([t]))/len(row["g"])
                if pos==row["div"]:
                    loss=loss+.5*F.softplus(.5-(z[0,t]-z[0,row["bad"][pos]]))
                assert torch.isfinite(loss)
                loss.backward();val+=float(loss.detach())
            for k,s in sources.items():
                if s.centroids.grad is not None and float(s.centroids.grad.abs().sum())>0:seen.add(k)
            if full.gates.grad is not None:
                gates.update(k for k,x in enumerate(full.gates.grad) if float(x.abs())>0)
            for k,p in child.named_parameters():
                if p.grad is not None and float(p.grad.abs().sum())>0:regions.add(k)
            torch.nn.utils.clip_grad_norm_(core+region+[full.gates],1.,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for k,s in sources.items():s.centroids[k_for(RATE,k):].zero_()
            curve.append(val)
            if (i+1)%16==0:print("F06_PROGRESS",i+1,val,len(seen),len(gates),len(regions),flush=True)
        assert len(seen)==85 and gates and regions
        # Persist all regions in the same historical names and shape, with
        # no external sidecar excluded from measured packet bytes.
        new_side=dict(side)
        new_side[NAMES[3]]=save({OLD_PREFIX+k:v.detach().cpu().half().contiguous()
                                for k,v in child.state_dict().items()})
        original_gate_key=next(iter(load_file(str(root/NAMES[4]))))
        new_side[NAMES[4]]=save({original_gate_key:full.gates.detach().cpu().half().contiguous()})
        packed=encode(archive.blob,sources,RATE,new_side)
        restored,rs=decode(packed)
        assert rs==new_side
        (out/"F06_JOINT.bundle").write_bytes(packed)
        del opt,state,model,sources,bridge,full,expansion,child,region,core
        for k in NAMES[1:]: (root/k).write_bytes(rs[k])
        final=build(root,restored)
        after=evaluate(final,dv,tok)
        report={"stage":"F06_INTEGRATED_RECOVERY","original_full":baseline,
                "mixed_full_before":before,"mixed_full_after_reload":after,
                **metrics(len(packed)),"sha256":digest(packed),"depth":12,
                "steps":STEPS,"connected_core_codebooks":len(seen),
                "connected_GT_gates":sorted(gates),"connected_region_parameter_names":sorted(regions),
                "FP32_teacher_calls":0,"candidate_accepted":False,"output_gate":0,"all_in_certified":False,
                "limitations":["Matched original and mixed initialization, single integrated contrastive training arm, not a controlled treatment comparison.",
                               "Existing 421793-region child loaded; 53751 protected old NRA3 parameters and legacy GT donors frozen; not every parameter trainable.",
                               "Internal 32 arithmetic diagnostic, not protected J7 or optimized CPU test; no complete runtime inventory."]}
        (out/"F06_REPORT.json").write_text(json.dumps(report,indent=2))
        print("F06_ASSESS",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("f01","f03","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f03,a.out)
