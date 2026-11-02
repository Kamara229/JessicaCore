#!/usr/bin/env python3
"""CALIBRI v45: offline PRE quantization + POST teacher-residual sidecar bridge.

Runs ONLY on genuine pretrained Pythia-14m weights in isolated CI research.
A (WikiText TRAIN [0:2048]) calibrates frozen CT33 quantization. Disjoint
B ([2048:4096]) fits low-rank and/or output-affine teacher corrections.
Post-correction remains a small physically persisted FP16 sidecar, never
pretends that a lossy compressed archive alone could recover lost info.
Evaluates measured full-model PPL with separate PyTorch hooks, PLUS
functional row-stream/mmap bridge on real activation samples.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib
import numpy as np
import torch

from all_mlp_agnews_v36 import package,unpack_bundle,module_identity
from direct_tiles_v33 import encode as encode_ct33,decode as decode_ct33
from calibrated_rows_v35 import calibrated_encode
from pareto_fixed_highbits_v43 import trained_fixedbit
from fixedbit_v15 import decode as decode_cf15
from quality import MODEL,DATASET,load_token_ids,module_selection,calibrate,ppl_eval
from mmap_rowstream_worker_v44 import load_mmap_index,map_forward

RANK=2
TRAIN_A=2048
TRAIN_B=2048
CONTEXT=128
BASE_BYTES=343840
INT4_BYTES=401224
PARAMS=786432
MAGIC=b"C45C"
HEAD=struct.Struct("<4sBBBBI")
ITEM=struct.Struct("<BBHH")
MODES={"rank2":1,"affine":2,"hybrid":3}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def fp16_bytes(a):
    a=np.asarray(a,dtype=np.float32)
    if not np.isfinite(a).all() or np.max(np.abs(a))>=65504:
        raise ValueError("sidecar has nonfinite or FP16-out-of-range parameters")
    return a.astype("<f2").tobytes()


def fit_sidecar(w,q,x):
    """Fit exactly rank 2 from B's input-weighted teacher residual.

    Weight scaling uses B activations; teacher original W is available ONLY
    at offline calibration. No labels/test tokens, no online teacher calls.
    Affine correction is fitted on B after rank correction; affine-only is
    independently fitted against the plain quantized matrix.
    """
    x=np.asarray(x,np.float32)
    w=np.asarray(w,np.float32)
    q=np.asarray(q,np.float32)
    m,n=w.shape
    if x.shape!=(TRAIN_B,n) or q.shape!=w.shape:
        raise AssertionError("teacher source geometry or B split invalid")
    std=np.sqrt(np.mean(np.square(x.astype(np.float64)),axis=0)+1e-9)
    residual=(w.astype(np.float64)-q.astype(np.float64))*std[None,:]
    u,s,vt=np.linalg.svd(residual,full_matrices=False)
    # Balanced factors reduce FP16 roundoff, deterministic canonical signs.
    a=u[:,:RANK]*np.sqrt(s[:RANK])[None,:]
    b=(np.sqrt(s[:RANK])[:,None]*vt[:RANK,:])/std[None,:]
    for k in range(RANK):
        pivot=int(np.argmax(np.abs(a[:,k])))
        if a[pivot,k]<0:
            a[:,k]*=-1
            b[k,:]*=-1
    a=np.frombuffer(fp16_bytes(a),dtype="<f2").reshape(m,RANK).astype(np.float32)
    b=np.frombuffer(fp16_bytes(b),dtype="<f2").reshape(RANK,n).astype(np.float32)
    teacher=x@w.T
    raw=x@q.T
    lowrank=raw+(x@b.T)@a.T

    def affine_fit(pred):
        # Ridge-to-identity 5% of each channel's B-output variance.
        # One fixed fit, never selected by external validation.
        p=pred.astype(np.float64)
        t=teacher.astype(np.float64)
        pm=p.mean(axis=0)
        tm=t.mean(axis=0)
        pp=p-pm[None,:]
        tt=t-tm[None,:]
        var=np.sum(pp*pp,axis=0)
        cross=np.sum(pp*tt,axis=0)
        ridge=0.05*var+1e-8
        gain=(cross+ridge)/(var+ridge)
        bias=tm-gain*pm
        return (np.frombuffer(fp16_bytes(gain),dtype="<f2").astype(np.float32),
                np.frombuffer(fp16_bytes(bias),dtype="<f2").astype(np.float32))
    plain_gain,plain_bias=affine_fit(raw)
    hybrid_gain,hybrid_bias=affine_fit(lowrank)
    return {"rank2":{"a":a,"b":b},
            "affine":{"gain":plain_gain,"bias":plain_bias},
            "hybrid":{"a":a,"b":b,"gain":hybrid_gain,"bias":hybrid_bias}}


def make_sidecar(items,mode):
    """Real fixed-format physical file with per-layer shapes, CRC and SHA."""
    if mode not in MODES or len(items)!=12:
        raise AssertionError("invalid sidecar mode/12-module count")
    body=bytearray()
    for name,p in sorted(items.items(),key=lambda z:module_identity(z[0])):
        layer,kind=module_identity(name)
        m,n=((512,128) if kind==0 else (128,512))
        body.extend(ITEM.pack(layer,kind,m,n))
        if mode in ("rank2","hybrid"):
            if p["a"].shape!=(m,RANK) or p["b"].shape!=(RANK,n):
                raise AssertionError("rank sidecar dimensions incorrect")
            body.extend(fp16_bytes(p["a"]))
            body.extend(fp16_bytes(p["b"]))
        if mode in ("affine","hybrid"):
            if p["gain"].shape!=(m,) or p["bias"].shape!=(m,):
                raise AssertionError("affine sidecar dimensions incorrect")
            body.extend(fp16_bytes(p["gain"]))
            body.extend(fp16_bytes(p["bias"]))
    pre=HEAD.pack(MAGIC,1,MODES[mode],RANK,len(items),len(body))+body
    pre+=struct.pack("<I",zlib.crc32(pre))
    return bytes(pre)+hashlib.sha256(pre).digest()


def open_sidecar(raw,mode):
    """Checksum and geometry validated BEFORE loading ANY learned correction."""
    if len(raw)<HEAD.size+4+32:
        raise ValueError("short correction sidecar")
    magic,version,encoded,rank,count,size=HEAD.unpack_from(raw)
    if (magic!=MAGIC or version!=1 or encoded!=MODES[mode] or
        rank!=RANK or count!=12 or len(raw)!=HEAD.size+size+4+32):
        raise ValueError("bad correction sidecar header")
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError("invalid correction SHA256")
    if zlib.crc32(raw[:-36])!=struct.unpack_from("<I",raw,len(raw)-36)[0]:
        raise ValueError("invalid correction CRC")
    off=HEAD.size
    items={}
    for _ in range(12):
        layer,kind,m,n=ITEM.unpack_from(raw,off)
        off+=ITEM.size
        if not (0<=layer<6 and kind in (0,1)
                and (m,n)==((512,128) if kind==0 else (128,512))):
            raise ValueError("invalid correction geometry")
        name=f"gpt_neox.layers.{layer}.mlp."+(
            "dense_h_to_4h" if kind==0 else "dense_4h_to_h")
        if name in items:
            raise ValueError("duplicate correction module")
        p={}
        def read(shape):
            nonlocal off
            count=int(np.prod(shape))
            v=np.frombuffer(raw,dtype="<f2",count=count,offset=off).astype(np.float32).reshape(shape).copy()
            off+=2*count
            if not np.isfinite(v).all():
                raise ValueError("nonfinite sidecar floats")
            return v
        if mode in ("rank2","hybrid"):
            p["a"]=read((m,RANK))
            p["b"]=read((RANK,n))
        if mode in ("affine","hybrid"):
            p["gain"]=read((m,))
            p["bias"]=read((m,))
        items[name]=p
    if off!=len(raw)-36 or len(items)!=12:
        raise ValueError("sidecar size mismatch")
    return items


def apply_numpy(y,x,p):
    if "a" in p:
        y=y+(x@p["b"].T)@p["a"].T
    if "gain" in p:
        y=y*p["gain"][None,:]+p["bias"][None,:]
    return y


def streamed_bridge_check(file,sidecars,examples,decoded):
    """Use v44's real file mmap rowstream then sidecar, never full W on path."""
    fp,mm,index,codec=load_mmap_index(str(file))
    try:
        x={f"{k}_{t}":np.ascontiguousarray(examples[name][:8].astype(np.float32))
           for name in decoded for k,t in [module_identity(name)]}
        raw,scratch=map_forward(mm,index,x)
    finally:
        mm.close()
        fp.close()
    if scratch>32768:
        raise AssertionError("runtime bridge exceeds frozen 16-row decoding scratch")
    differences={}
    for mode,items in sidecars.items():
        diff=0.
        for name,q in decoded.items():
            k,t=module_identity(name)
            key=f"{k}_{t}"
            result=apply_numpy(raw[key],x[key],items[name])
            dense=apply_numpy(x[key]@q.T,x[key],items[name])
            diff=max(diff,float(np.max(np.abs(result-dense))))
        differences[mode]=diff
        if diff>2e-4:
            raise AssertionError("real physical mmap + learned compensation fidelity failed")
    return {"mmap_path_never_materializes_full_W":True,
            "max_decoded_W_scratch_bytes":scratch,
            "maximum_abs_delta_mmap_vs_dense_corrected":differences}


def new_text(tokenizer,revision,which):
    """Two fixed disjoint previous-dataset slices, not reused text."""
    from datasets import load_dataset
    if which=="ARC_Challenge_new":
        ds=load_dataset("allenai/ai2_arc","ARC-Challenge",
                        split="validation[:128]",revision=revision)
        parts=[]
        for row in ds:
            choices=row["choices"]
            parts.append("Question: "+str(row["question"]).strip()+"\nOptions: "+
                "; ".join(str(a)+") "+str(b) for a,b in zip(
                    choices.get("label",[]),choices.get("text",[]))))
    elif which=="Yelp_disjoint_64_128":
        ds=load_dataset("Yelp/yelp_review_full",
                        split="test[64:128]",revision=revision)
        parts=[str(r["text"]).strip() for r in ds]
    else:
        raise ValueError("unregistered evaluation corpus")
    ids=tokenizer("\n\n".join(parts),add_special_tokens=False,
                  return_tensors="pt").input_ids[0,:2048]
    if len(ids)!=2048:
        raise AssertionError("external predeclared slice lacks 2048 tokens")
    return ids


def measure_ppl(model,originals,decoded,sidecars,corpora):
    output={}
    variants=("teacher","base_CT33","pre_post_rank2","pre_post_affine",
              "pre_post_hybrid","calibrated_INT4")
    try:
        for kind in variants:
            hooks=[]
            with torch.no_grad():
                for name,w in originals.items():
                    model.get_submodule(name).weight.copy_(
                        w if kind=="teacher" else
                        torch.from_numpy(decoded["int4"][name] if kind=="calibrated_INT4"
                                         else decoded["ct33"][name]))
            if kind.startswith("pre_post"):
                mode=kind.removeprefix("pre_post_")
                for name,p in sidecars[mode].items():
                    params={k:torch.from_numpy(v.copy()) for k,v in p.items()}
                    def hook(mod,inp,out,params=params):
                        x=inp[0]
                        y=out
                        if "a" in params:
                            y=y+(x@params["b"].T)@params["a"].T
                        if "gain" in params:
                            y=y*params["gain"]+params["bias"]
                        return y
                    hooks.append(model.get_submodule(name).register_forward_hook(hook))
            try:
                output[kind]={}
                with torch.inference_mode():
                    for domain,tokens in corpora.items():
                        metric=ppl_eval(model,tokens,CONTEXT)
                        output[kind][domain]={"ppl":float(metric["ppl"]),
                                             "nll":float(metric["nll"])}
                        print(json.dumps({"variant":kind,"domain":domain,
                            "PPL":metric["ppl"]}),flush=True)
            finally:
                for h in hooks:
                    h.remove()
    finally:
        with torch.no_grad():
            for name,w in originals.items():
                model.get_submodule(name).weight.copy_(w)
    return output


def run(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2"
            and os.getenv("DNNL_MAX_CPU_ISA")=="AVX2"
            and os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2"
            and str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("must pin real selected AVX2 CPU capability")
    torch.manual_seed(41)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    api=HfApi()
    model_rev=api.model_info(MODEL).sha
    wiki_rev=api.dataset_info(DATASET).sha
    external={"ARC_Challenge_new":api.dataset_info("allenai/ai2_arc").sha,
              "Yelp_disjoint_64_128":api.dataset_info("Yelp/yelp_review_full").sha}
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=model_rev,
                                               torch_dtype=torch.float32)
    model.eval()
    modules=module_selection(model)
    names=sorted(modules,key=module_identity)
    originals={n:modules[n].weight.detach().clone() for n in names}
    w={n:originals[n].numpy().astype(np.float32).copy() for n in names}
    if len(names)!=12 or sum(a.size for a in w.values())!=PARAMS:
        raise AssertionError("pretrained source Pythia 12-MLP shape changed")
    train=load_token_ids(tokenizer,"train",wiki_rev,TRAIN_A+TRAIN_B)
    a=train[:TRAIN_A]
    b=train[TRAIN_A:]
    if len(a)!=TRAIN_A or len(b)!=TRAIN_B or sha(a.numpy().tobytes())==sha(b.numpy().tobytes()):
        raise AssertionError("pre/post calibration train partitions invalid")
    with torch.inference_mode():
        act_a=calibrate(model,modules,a,CONTEXT)
        act_b=calibrate(model,modules,b,CONTEXT)
    pre={}
    high={}
    side_params={mode:{} for mode in MODES}
    for n in names:
        xa=act_a[n].numpy().astype(np.float32)
        xb=act_b[n].numpy().astype(np.float32)
        source=w[n]
        moments=np.mean(xa*xa,axis=0,dtype=np.float32)
        raw,_=encode_ct33(source,moments,int(source.size*3.5/8),
                          br=1,bc=source.shape[1])
        pre[n],m=calibrated_encode(source,xa,raw)
        if len(raw)!=len(pre[n]) or not m["bits_are_identical_per_row"]:
            raise AssertionError("pre quantization changed the frozen v44 physical rate")
        q=decode_ct33(pre[n])
        _,high[n],_=trained_fixedbit(source,moments,4)
        fitted=fit_sidecar(source,q,xb)
        for mode in MODES:
            side_params[mode][n]=fitted[mode]

    root=Path(args.output).parent
    root.mkdir(parents=True,exist_ok=True)
    files={}
    decoded={}
    for tag,components,codec,size,decoder in (
        ("ct33",pre,1,BASE_BYTES,decode_ct33),
        ("int4",high,2,INT4_BYTES,decode_cf15)):
        binary=package(components,codec)
        file=root/("CALIBRI_v45_"+tag+".bin")
        file.write_bytes(binary)
        again=file.read_bytes()
        if again!=binary or len(again)!=size:
            raise AssertionError("physical pretrained weight source size mismatch")
        unpacked=unpack_bundle(again,codec)
        decoded[tag]={n:decoder(unpacked[n]) for n in names}
        files[tag]={"path":str(file),"real_bytes":size,"SHA256":sha(again)}
    sidecars={}
    for mode in MODES:
        binary=make_sidecar(side_params[mode],mode)
        file=root/("CALIBRI_v45_"+mode+".c45c")
        file.write_bytes(binary)
        again=file.read_bytes()
        sidecars[mode]=open_sidecar(again,mode)
        # Corrupt test MUST fail integrity checks before any correction.
        damaged=bytearray(again)
        damaged[-2]^=1
        try:
            open_sidecar(bytes(damaged),mode)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt correction data accepted")
        files[mode]={"path":str(file),"real_bytes":len(again),"SHA256":sha(again),
                     "all_in_storage_bytes":BASE_BYTES+len(again),
                     "physical_bpw":8*(BASE_BYTES+len(again))/PARAMS}
    for mode in MODES:
        if files[mode]["all_in_storage_bytes"]>=INT4_BYTES:
            raise AssertionError("v45 sidecar storage crossed predeclared physical INT4 rate")
    bridge=streamed_bridge_check(Path(files["ct33"]["path"]),sidecars,
                                 {n:act_b[n].numpy().astype(np.float32) for n in names},
                                 decoded["ct33"])
    # Never read evaluation TEXT during pre/post quantizer fitting or writing.
    corpus={"WikiText2_DEV_known":load_token_ids(tokenizer,"validation",wiki_rev,1024)}
    for name in external:
        corpus[name]=new_text(tokenizer,external[name],name)
    out=measure_ppl(model,originals,decoded,sidecars,corpus)
    data={
        "experiment":"CALIBRI v45 offline pre-quant and post-quant teacher-compensation bridge",
        "source_model":MODEL,"model_revision":model_rev,"wiki_revision":wiki_rev,
        "source_MLP_SHA256":{n:sha(v.tobytes()) for n,v in w.items()},
        "train_A_tokens_SHA256":sha(a.numpy().tobytes()),
        "train_B_disjoint_tokens_SHA256":sha(b.numpy().tobytes()),
        "eval_SHA256":{name:sha(v.numpy().tobytes()) for name,v in corpus.items()},
        "eval_dataset_revisions":external,
        "source_coefficients":PARAMS,
        "pre_fit":"frozen v44 TRAIN-A rowwise INT3/4 plus FP16 scale/codes",
        "post_fit":"TRAIN-B teacher residual input-weighted rank2 SVD and/or ridge-to-identity affine",
        "post_rank":RANK,"physical_files":files,"streamed_runtime_fidelity":bridge,
        "full_model_PPL_after_dense_weight_restore_and_torch_hooks":out,
        "compare_quality_against_plain_prequant":{
            k:{d:out[k][d]["ppl"]-out["base_CT33"][d]["ppl"] for d in corpus}
            for k in ("pre_post_rank2","pre_post_affine","pre_post_hybrid")},
        "limitations":[
            "Sidecar MUST be stored: lost information cannot be recovered from a lossy file alone.",
            "Teacher W and TRAIN-B activations used OFFLINE only; runtime requires neither original W nor teacher.",
            "Native mmap bridge functional test uses real inputs, but PPL hooks use dense-restored compressed Q for correctness; NOT production packed end-to-end runtime or speed.",
            "Combined rates count all CT33 bits/CRC/scales plus actual correction file CRC/SHA/FP16, NOT full checkpoint.",
            "No evaluation-tuned rank, regularization, data splits, bit budget or sample count. External slices: ARC-Challenge validation[:128] and Yelp test[64:128].",
            "These are 2048-token exploratory PPL excerpts, not task accuracy or broad statistical significance.",
            "No integration with Jessica/Render/Supabase/APK, no QMoE 0.8bpw replication or native speed claims."
        ]}
    Path(args.output).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"final":"actual_measured_v45","files":{
        k:{"bytes":v["real_bytes"],"total":v.get("all_in_storage_bytes")}
        for k,v in files.items()},
        "mmap_bridge":bridge,
        "PPL":{k:{d:round(v["ppl"],5) for d,v in x.items()} for k,x in out.items()}
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/calibri_v45.json")
    run(parser.parse_args())
