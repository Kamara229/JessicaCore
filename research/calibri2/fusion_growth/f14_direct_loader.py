"""F14: load the trained mixed-rate core directly, without the old core file.

Research CPU loader equivalence, not a compact inference kernel or J7 pass.
F06 previously constructed original K32 weights and then overwrote codebooks
and IDs. This direct route restores the same tensors from one physical bundle.
"""
import argparse
import json
import tempfile
import time
from pathlib import Path
import torch
from safetensors.torch import load
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,metrics,unpack
from research.calibri2.fusion_growth.f03_rate_train_core import decode,encode,make
from research.calibri2.fusion_growth.f06_integrated_recovery import (
    build,forward,NativeFullHeadShadow,LearnedOnPathInterlayer,
    LearnedCausalEarlyProjector,JointRuNative,OLD_PREFIX,make_joint_trainable,set_gt_route)
from research.calibri2.fusion_growth.f09_head_rate_codec import split_head,open_bundle,HDR
from research.calibri2.fusion_growth.f10_joint_own_error_learning import native
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import prompt_features
from calibri2.s112.prompt_conditioned_first_token_mps import load_cpu_original
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import ENCODED_KEYS,load_compact_q2
from calibri2.s132.progressive_core import ProgressiveRouteCore,IDS_KEY,WEIGHTS_KEY
from v176.operators import clear

PARENT_SHA="7dcc391ce6001f88f6e7c12498c4f371b83431b049e4db2cb65d7f0b89ab722a"
RATE="K16_embed_K8_body"
PROBES=("Назови число 17.","Повтори слово река.","Сколько будет 24 + 12?",
        "Кто пришёл раньше: Олег или Ира, если Ира пришла первой?",
        "У Веры нет мяча. Есть ли у неё мяч?","Слева стоит кот, справа пёс. Кто справа?",
        "Choose the larger number: 19 or 42.","Repeat exactly: blue sky.")

def direct(archive,side):
    """No file paths, original K32 archive, or external model weights accepted."""
    assert set(side)==set(NAMES[1:])
    assert archive.meta["original_unique_parameter_count"]==36380480
    assert digest(side[NAMES[2]])==PIN[NAMES[2]]
    legacy=load(side[NAMES[1]]);inherited=load(side[NAMES[2]])
    assert set(inherited)==set(legacy)|set(ENCODED_KEYS)|{IDS_KEY,WEIGHTS_KEY}
    assert all(torch.equal(inherited[k],v) for k,v in legacy.items())
    assert inherited[WEIGHTS_KEY].shape==(12,320,48)
    cpu=load_cpu_original(archive.blob,legacy)
    model,sources,*_=cpu
    bridge=PromptConditionedDualTunnel(cpu[6]).eval().install(model)
    bridge.coefficients.requires_grad_(False)
    full=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
    load_compact_q2(full,{k:inherited[k] for k in ENCODED_KEYS});full.install(model)
    expansion=ProgressiveRouteCore(rank=896).eval().bind_original_route_bridge(bridge).install(model)
    expansion.load_active_fp16({IDS_KEY:inherited[IDS_KEY],WEIGHTS_KEY:inherited[WEIGHTS_KEY]})
    model.eval().requires_grad_(False)
    assert len(model.model.layers)==12 and len(sources)==85
    child=JointRuNative(LearnedCausalEarlyProjector(LearnedOnPathInterlayer(NativeFullHeadShadow())))
    raw=load(side[NAMES[3]])
    assert all(k.startswith(OLD_PREFIX) for k in raw) and sum(v.numel() for v in raw.values())==421793
    child.load_state_dict({k[len(OLD_PREFIX):]:v.float() for k,v in raw.items()},strict=True)
    child.eval();params=make_joint_trainable(child)
    model.model.add_module("nra3_s119m_onpath",child)
    gates=load(side[NAMES[4]]);assert len(gates)==1
    gate=next(iter(gates.values())).float();assert gate.shape==(12,)
    set_gt_route(full,"GT_RECURRENT",gate);full.gates.requires_grad_(True)
    for s in sources.values():s.centroids.requires_grad_(True)
    clear(sources)
    # Historical research hook insists on differentiable logits even for
    # diagnostic inference. This is deliberately NOT a production speed path.
    return model,sources,bridge,full,expansion,child,params

def compare_state(a,b):
    for x,y in zip((a[0],a[2],a[3],a[4],a[5]),(b[0],b[2],b[3],b[4],b[5])):
        p,q=x.state_dict(),y.state_dict()
        assert set(p)==set(q)
        assert all(torch.equal(p[k],q[k]) for k in p)
    assert set(a[1])==set(b[1])
    for k,s in a[1].items():
        t=b[1][k]
        assert torch.equal(s.codes,t.codes) and torch.equal(s.centroids,t.centroids)
        assert torch.equal(s.base_scales,t.base_scales)

def generate(state,p,f,cap=8):
    ids=[];logits=[]
    for _ in range(cap):
        z=forward(state,p+ids,f).detach().cpu()
        logits.append(z.clone())
        ids.append(int(z.argmax()))
        if ids[-1]==2:break
    return ids,logits

def run(f01,f06,f08,out):
    torch.set_num_threads(2);torch.manual_seed(20261202)
    source=(f06/"F06_JOINT.bundle").read_bytes();assert digest(source)==PARENT_SHA
    archive,side=decode(source)
    legacy=(f08/"legacy_extras.bundle").read_bytes();assert len(legacy)==137712
    previous=json.loads((f08/"F08_REPORT.json").read_text())
    from transformers import AutoTokenizer
    folder=f08/"tokenizer"
    for item in previous["tokenizer_files"]:
        assert digest((folder/item["name"]).read_bytes())==item["sha256"]
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    # New Q8 packet is derived from trained F06, not old F09 weights.
    model,sources=make(archive)
    compact,head,_=split_head(side,8)
    cp=encode(archive.blob,sources,RATE,compact)
    packed=HDR.pack(b"FG09",len(cp),len(head),len(legacy))+cp+head+legacy
    qcore,qside,qlegacy=open_bundle(packed)
    assert qcore.blob==archive.blob and qlegacy==legacy
    del model,sources
    out.mkdir(parents=True,exist_ok=False);(out/"F14_F06_Q8.bundle").write_bytes(packed)
    original=unpack((f01/"F01.xz.bundle").read_bytes())
    for k,v in PIN.items():assert digest(original[k])==v
    results={}
    for mode,a,s in (("FP16",archive,side),("Q8",qcore,qside)):
        # The only use of old core bytes is the reference implementation.
        # The direct loader accepts exclusively decoded physical candidate data.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/NAMES[0]).write_bytes(original[NAMES[0]])
            for k,v in s.items():(root/k).write_bytes(v)
            ref=build(root,a)
        t=time.perf_counter();new=direct(a,s);load_s=time.perf_counter()-t
        compare_state(ref,new)
        maximum=0.;n=0;tokens=0;exact=0;records=[]
        for q in PROBES:
            from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import strict_chat_source_ids
            p=strict_chat_source_ids(tok,q);f=prompt_features(tok,q)
            x,lx=generate(ref,p,f);y,ly=generate(new,p,f)
            exact+=int(x==y);assert x==y and len(lx)==len(ly)
            for u,v in zip(lx,ly):
                maximum=max(maximum,float((u-v).abs().max()));n+=1
                assert torch.equal(u,v),"direct loader logit drift"
            tokens+=len(x);records.append({"prompt_sha256":digest(q.encode()),"ids":y})
        results[mode]={"same_model_state":True,"exact_generation":exact,"prompts":len(PROBES),
                       "exact_full_vocab_logit_steps":n,"generated_tokens":tokens,
                       "max_logit_difference":maximum,"direct_load_seconds":load_s,
                       "records":records}
        del ref,new
        print("F14_ARM",mode,json.dumps(results[mode],sort_keys=True),flush=True)
    report={"stage":"F14_DIRECT_PHYSICAL_RESTORE","parent_sha256":PARENT_SHA,
            "depth":12,"Q8_sha256":digest(packed),"Q8_plus_legacy":metrics(len(packed)),
            "FP16_plus_legacy":metrics(len(source)+len(legacy)),
            "tokenizer_bytes_separate":previous["tokenizer_bytes_separate"],
            "arms":results,"candidate_accepted":False,"output_gate":0,"all_in_certified":False,
            "limitations":["Exact parity only against F06 research loader for the same physical weights, not S194/J7.",
                           "Q8 weight quantization compared only to same Q8 reference, not evidence of quality preservation.",
                           "No original K32 file is needed by direct loader; reference side of audit uses it.",
                           "Legacy extras still not active. Tokenizer local, runtime source not charged. CPU compact kernels and protected retention pending.",
                           "Differentiable diagnostic path; load duration not native decode benchmark. No training."]}
    (out/"F14_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F14_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f06","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f06,a.f08,a.out)
