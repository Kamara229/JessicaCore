#!/usr/bin/env python3
"""CALIBRI v76: physically paid low-rank output-weight residual on real CB59 LLM.

A single CB76 file contains the original complete 135M CB59 source AND a
rank-4 (49152x576) output-weight residual U @ V.T plus a 49152-element
output bias, all fp16. Physical file <=0.700 bits / ORIGINAL unique param.
The base embedding+linear projection remains direct-native packed 5-bit C.
Residual is a small FP32-on-CPU factor operation decoded ONLY from sidecar.

Teacher-trained full-model KL on new WikiText + AG-News TRAIN FIT, choose
shrinkage/iteration only on independent TRAIN VALID, then open frozen
disjoint genuine TEST. Does not involve live Jessica, main, Render, APK.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from accelerate import init_empty_weights
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from direct_codebook_v59 import CB59
from direct_runtime_v59 import NativeKernel, bind_into_llama, compile_kernel
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from quality import DATASET, load_token_ids

SOURCE_BYTES=11229303
RANK=4
DIM=576
VOCAB=49152
TAIL=struct.Struct("<8sIII")
MAGIC=b"CLBR76\0\0"
DIGEST_BYTES=36
DATA_BYTES=2*(VOCAB*RANK+DIM*RANK+VOCAB)
TRAIN_WINDOWS=4
TEST_WINDOWS=8
WINDOW=64
FIT_CORPORA=("wiki_fit","news_fit")
VALID_CORPORA=("wiki_valid","news_valid")
TEST_CORPORA=("wiki_test","news_test")
ALL_CORPORA=FIT_CORPORA+VALID_CORPORA+TEST_CORPORA
LEARNING_STEPS=(20,50,90)
REG_B=0.01
REG_UV=0.001
MIN_GAIN_EACH_VALID_CORPUS=0.10
AG_DATASET="fancyzhx/ag_news"
REQUIRED_PHYSICAL_BYTES=SOURCE_BYTES+TAIL.size+DATA_BYTES+DIGEST_BYTES
MAX_PHYSICAL_BYTES=math.floor(.700*EXPECTED_UNIQUE/8)
assert RANK==4 and REQUIRED_PHYSICAL_BYTES<=MAX_PHYSICAL_BYTES

def get_tokens(corpus):
    if corpus not in ALL_CORPORA:raise ValueError(corpus)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    count=(TEST_WINDOWS if corpus in TEST_CORPORA else TRAIN_WINDOWS)*WINDOW
    if corpus.startswith("wiki_"):
        split="test" if corpus=="wiki_test" else "train"
        start={"wiki_fit":12288,"wiki_valid":14336,"wiki_test":70000}[corpus]
        revision=HfApi().dataset_info(DATASET).sha
        ids=load_token_ids(tokenizer,split,revision,start+count)[start:start+count]
        origin={"dataset":DATASET,"split":split,"revision":revision,
                "token_span":[start,start+count]}
    else:
        start={"news_fit":8000,"news_valid":8500,"news_test":3000}[corpus]
        split="test" if corpus=="news_test" else "train"
        revision=HfApi().dataset_info(AG_DATASET).sha
        records=load_dataset(AG_DATASET,split=f"{split}[{start}:{start+100}]",revision=revision)
        raw="\n\n".join(str(record["text"]) for record in records)
        ids=tokenizer(raw,add_special_tokens=False,return_tensors="pt").input_ids[0,:count]
        origin={"dataset":AG_DATASET,"split":split,"revision":revision,
                "record_range":[start,start+100],
                "token_span":[0,count],"text_only_no_labels":True}
    if len(ids)!=count:raise AssertionError("not enough original real text")
    origin["token_sha256"]=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    return ids,origin

def zero_factors():
    return (np.zeros((VOCAB,RANK),dtype="<f2"),
            np.zeros((DIM,RANK),dtype="<f2"),
            np.zeros(VOCAB,dtype="<f2"))

def pack(base,U,V,b):
    if len(base)!=SOURCE_BYTES:raise AssertionError("source complete CB59 byte length")
    cb=CB59(base)
    if cb.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE:
        raise AssertionError("not original fully trained source")
    for arr,shape in ((U,(VOCAB,RANK)),(V,(DIM,RANK)),(b,(VOCAB,))):
        if np.asarray(arr).shape!=shape or not np.isfinite(arr).all():
            raise AssertionError("low rank sidecar wrong shape or non-finite")
    header=TAIL.pack(MAGIC,RANK,DIM,VOCAB)
    payload=b"".join(np.asarray(array,dtype="<f2").tobytes(order="C")
                     for array in (U,V,b))
    if len(payload)!=DATA_BYTES:raise AssertionError("sidecar physical byte cost incorrect")
    prefix=base+header+payload
    crc=struct.pack("<I",zlib.crc32(prefix))
    with_crc=prefix+crc
    blob=with_crc+hashlib.sha256(with_crc).digest()
    if len(blob)!=REQUIRED_PHYSICAL_BYTES or len(blob)>MAX_PHYSICAL_BYTES:
        raise AssertionError("complete original weights + corrections exceed paid 0.7bpw")
    return blob

def unpack(blob):
    if len(blob)!=REQUIRED_PHYSICAL_BYTES or len(blob)>MAX_PHYSICAL_BYTES:
        raise ValueError("CB76 physically incomplete or above source rate cap")
    base=blob[:SOURCE_BYTES]
    cb=CB59(base)
    if cb.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE:
        raise ValueError("base original parameter count mismatch")
    magic,r,d,v=TAIL.unpack_from(blob,SOURCE_BYTES)
    if (magic,r,d,v)!=(MAGIC,RANK,DIM,VOCAB):
        raise ValueError("CB76 unsupported physical sidecar geometry")
    if struct.unpack("<I",blob[-36:-32])[0]!=zlib.crc32(blob[:-36]):
        raise ValueError("CB76 full archive CRC failed")
    if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
        raise ValueError("CB76 entire source+correction SHA failed")
    payload=memoryview(blob)[SOURCE_BYTES+TAIL.size:-36]
    if len(payload)!=DATA_BYTES:raise ValueError("missing physically paid source payload")
    n1=VOCAB*RANK*2;n2=DIM*RANK*2
    U=np.frombuffer(payload[:n1],dtype="<f2").copy().reshape(VOCAB,RANK)
    V=np.frombuffer(payload[n1:n1+n2],dtype="<f2").copy().reshape(DIM,RANK)
    bias=np.frombuffer(payload[n1+n2:],dtype="<f2").copy()
    if not all(np.isfinite(a).all() for a in (U,V,bias)):
        raise ValueError("non-finite physical low-rank factors")
    return cb,U,V,bias

class PaidResidualHead(nn.Module):
    def __init__(self,base,U,V,bias):
        super().__init__()
        self.base=base
        self.register_buffer("U",torch.from_numpy(U.astype(np.float32)))
        self.register_buffer("V",torch.from_numpy(V.astype(np.float32)))
        self.register_buffer("bias",torch.from_numpy(bias.astype(np.float32)))
    def forward(self,x):
        # Full 2D trained source stays packed. The paid correction does not
        # materialize original dense source nor form its own dense U@V.T.
        normalized=x/torch.sqrt(torch.mean(x.square(),dim=-1,keepdim=True)+1e-6)
        return self.base(x)+((normalized@self.V)@self.U.T)+self.bias

def make_model(kind,args):
    if kind=="teacher":
        m=AutoModelForCausalLM.from_pretrained(
            MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
            attn_implementation="eager").eval()
        return m,{"pretrained_teacher_loaded":True},None
    if kind=="baseline":
        source=args.original.read_bytes()
        archive=CB59(source)
        if len(source)!=SOURCE_BYTES:raise AssertionError("wrong CB59 original model")
        corrections=None
        physical=len(source)
        sha=hashlib.sha256(source).hexdigest()
    elif kind=="corrected":
        full=args.folder/"CALIBRI_v76_paid_full.cb76"
        source=full.read_bytes()
        archive,*factors=unpack(source)
        corrections=factors
        physical=len(source)
        sha=hashlib.sha256(source).hexdigest()
        if len(factors)!=3:raise AssertionError("sidecar absent")
    else:raise ValueError(kind)
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    # The archive supplies every trained parameter. Keep RoPE's computed
    # buffers on CPU, but never allocate a full random dense checkpoint.
    with init_empty_weights(include_buffers=False):
        m=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    native=NativeKernel(compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"),
                      args.folder/("libv76_"+kind+".so")))
    evidence=bind_into_llama(m,archive,native)
    if evidence["runtime_dense_multi_dim_source_parameters"]!=0:
        raise AssertionError("dense trained 2D source leaked into packed runtime")
    if corrections is not None:
        m.lm_head=PaidResidualHead(m.lm_head,*corrections)
    return m,{"pretrained_teacher_loaded":False,
              "direct_native_C_baseline_GEMV":True,
              "paid_lowrank_factor_operation_CPU_torch":corrections is not None,
              "zero_dense_pretrained_2D_hot_weights":True,
              "original_source_blob_length":SOURCE_BYTES,
              "physical_complete_model_bytes":physical,
              "all_in_bits_per_original_parameter":8*physical/EXPECTED_UNIQUE,
              "complete_physical_SHA256":sha,
              "tied_input_output_trained_source_handles_verified_before_correction":
                    evidence["tied_embed_and_output_codebook_SHARED_IN_RAM"]},m.get_output_embeddings()

def collect(args):
    if args.corpus not in ALL_CORPORA or args.kind not in ("teacher","baseline","corrected"):
        raise ValueError("not a frozen corpus or known full model")
    torch.set_num_threads(2)
    ids,origin=get_tokens(args.corpus)
    model,info,head=make_model(args.kind,args)
    output_dir=args.folder/"arrays";output_dir.mkdir(parents=True,exist_ok=True)
    prefix=output_dir/(args.corpus+"_"+args.kind)
    scores=np.lib.format.open_memmap(str(prefix)+".npy",mode="w+",dtype="<f2",
                     shape=(len(ids)//WINDOW,WINDOW,VOCAB))
    hiddens=None
    captures=[]
    handle=None
    if args.kind=="baseline" and args.corpus not in TEST_CORPORA:
        hiddens=np.lib.format.open_memmap(str(prefix)+"_hidden.npy",mode="w+",
                      dtype="<f2",shape=(len(ids)//WINDOW,WINDOW,DIM))
        def capture(m,inputs):
            captures.append(inputs[0].detach().cpu().numpy()[0].astype("<f2"))
        handle=model.lm_head.register_forward_pre_hook(capture)
    times=[]
    try:
        with torch.inference_mode():
            model(input_ids=ids[:4][None],use_cache=False)
            # Ignore warmup capture, real windows only below.
            captures.clear()
            for wi in range(len(ids)//WINDOW):
                x=ids[wi*WINDOW:(wi+1)*WINDOW][None]
                t0=time.perf_counter()
                a=model(input_ids=x,use_cache=False).logits[0].detach().cpu().numpy()
                times.append(time.perf_counter()-t0)
                if a.shape!=(WINDOW,VOCAB) or not np.isfinite(a).all():
                    raise AssertionError("broken actual full pretrained model logits")
                scores[wi]=a.astype("<f2")
                if hiddens is not None:
                    if len(captures)!=1 or captures[0].shape!=(WINDOW,DIM):
                        raise AssertionError("native student hidden source hook missing")
                    hiddens[wi]=captures.pop()
    finally:
        if handle is not None:handle.remove()
    scores.flush()
    if hiddens is not None:hiddens.flush()
    meta={"kind":args.kind,"corpus":args.corpus,"real_source":origin,
          "whole_model":info,"context_windows":len(ids)//WINDOW,
          "context_size":WINDOW,"native_model_prefill_seconds":sum(times),
          "per_window_prefill_seconds":times,
          "arrays_are_fp16_storage_not_original_F32_or_KL_exact":True,
          "student_hidden_capture_only_on_TRAIN":hiddens is not None}
    (Path(str(prefix)+".json")).write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({"phase":"collect","kind":args.kind,"corpus":args.corpus,
         "source_bytes":info.get("physical_complete_model_bytes"),
         "cpu_prefill_sec":round(sum(times),3)}),flush=True)

def tensors(folder,corpus):
    root=folder/"arrays"
    raw={}
    for name in ("teacher","baseline"):
        base=root/(corpus+"_"+name)
        raw[name]=np.load(str(base)+".npy",mmap_mode="r").astype(np.float32)[:,:-1,:]
        meta=json.loads(Path(str(base)+".json").read_text())
        raw[name+"_meta"]=meta
    if raw["teacher_meta"]["real_source"]["token_sha256"]!=raw["baseline_meta"]["real_source"]["token_sha256"]:
        raise AssertionError("teacher and baseline saw different original actual TRAIN tokens")
    hidden=np.load(str(root/(corpus+"_baseline_hidden.npy")),mmap_mode="r").astype(np.float32)[:,:-1,:]
    return (torch.from_numpy(raw["teacher"].reshape(-1,VOCAB).copy()),
            torch.from_numpy(raw["baseline"].reshape(-1,VOCAB).copy()),
            torch.from_numpy(hidden.reshape(-1,DIM).copy()))

def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(76);np.random.seed(76)
    fit=[tensors(args.folder,k) for k in FIT_CORPORA]
    valid=[tensors(args.folder,k) for k in VALID_CORPORA]
    combined=[torch.cat([tr[ix] for tr in fit],dim=0) for ix in range(3)]
    teacher,baseline,hidden=combined
    # Dataset-mean teacher distribution is the *target* for full vocabulary KL.
    target=torch.softmax(teacher,dim=-1).detach()
    h=hidden/torch.sqrt(torch.mean(hidden.square(),dim=-1,keepdim=True)+1e-6)
    U=torch.nn.Parameter(torch.zeros(VOCAB,RANK,dtype=torch.float32))
    V=torch.nn.Parameter(torch.randn(DIM,RANK)*.007)
    bias=torch.nn.Parameter(torch.zeros(VOCAB,dtype=torch.float32))
    opt=torch.optim.Adam([U,V,bias],lr=0.035)
    # Snap to true physical FP16 before validation, not unrealistically FP32.
    def snap(u,v,b):
        return (np.asarray(u.detach().numpy(),dtype="<f2").copy(),
                np.asarray(v.detach().numpy(),dtype="<f2").copy(),
                np.asarray(b.detach().numpy(),dtype="<f2").copy())
    def kl_on_corpus(data,weights,mode):
        t,z,feat=data
        q=torch.log_softmax(t,dim=-1)
        p=q.exp()
        u,v,b=(torch.from_numpy(arr.astype(np.float32)) for arr in weights)
        x=feat/torch.sqrt(torch.mean(feat.square(),dim=-1,keepdim=True)+1e-6)
        change=(x@v)@u.T if mode=="rank4" else 0.0
        logits=z+change+b
        lp=torch.log_softmax(logits,dim=-1)
        return float(torch.sum(p*(q-lp),dim=-1).mean())
    candidates=[{"step":0,"mode":"baseline","factor":zero_factors()}]
    # At 20/50/90 fit iterations evaluate both paid bias-only and paid
    # rank4+bias sidecars on TWO untouched, separately sourced TRAIN VALID.
    for step in range(1,max(LEARNING_STEPS)+1):
        opt.zero_grad(set_to_none=True)
        z=baseline+((h@V)@U.T)+bias
        logp=torch.log_softmax(z,dim=-1)
        cross_entropy=-(target*logp).sum(-1).mean()
        regularizer=REG_B*bias.square().mean()+REG_UV*(U.square().mean()+V.square().mean())
        loss=cross_entropy+regularizer
        loss.backward()
        opt.step()
        if step in LEARNING_STEPS:
            with torch.no_grad():
                s=snap(U,V,bias)
                if not np.isfinite(s[0]).all() or not np.isfinite(s[1]).all() or not np.isfinite(s[2]).all():
                    raise AssertionError("non-finite actual packed correction")
                candidates.append({"step":step,"mode":"rank4","factor":s})
                candidates.append({"step":step,"mode":"bias_only",
                    "factor":(np.zeros_like(s[0]),np.zeros_like(s[1]),s[2])})
            print(json.dumps({"train_step":step,"fit_full_KL_cross_entropy":round(float(cross_entropy),5),
                              "paid_factor_U_rms":round(float(U.detach().square().mean().sqrt()),6)}),flush=True)
    with torch.no_grad():
        bases={c:kl_on_corpus(data,zero_factors(),"rank4") for c,data in zip(VALID_CORPORA,valid)}
        choices=[]
        for item in candidates:
            if item["mode"]=="baseline":
                score=bases.copy()
            else:
                score={c:kl_on_corpus(data,item["factor"],"rank4")
                       for c,data in zip(VALID_CORPORA,valid)}
            improvements={c:bases[c]-score[c] for c in VALID_CORPORA}
            choices.append({"step":item["step"],"mode":item["mode"],
                            "VALID_KL":score,"KL_gain_vs_baseline":improvements,
                            "worst_VALID_gain":min(improvements.values())})
    best=max(range(len(choices)),key=lambda i:(choices[i]["worst_VALID_gain"],-i))
    selected_idx=best if choices[best]["worst_VALID_gain"]>=MIN_GAIN_EACH_VALID_CORPUS else 0
    selected=candidates[selected_idx]
    final=pack(args.original.read_bytes(),*selected["factor"])
    actual=args.folder/"CALIBRI_v76_paid_full.cb76"
    actual.write_bytes(final)
    cb,u,v,b=unpack(actual.read_bytes())
    assert cb.meta["original_unique_parameter_count"]==EXPECTED_UNIQUE
    if len(final)!=REQUIRED_PHYSICAL_BYTES:raise AssertionError("unpaid physical corrections")
    meta={"experiment":"CALIBRI v76 paid rank4 trained output-weight+logit bias correction",
          "trained_original_source":MODEL,"pinned_original_model_revision":MODEL_SHA,
          "real_original_unique_parameter_count":EXPECTED_UNIQUE,
          "original_CB59_source_bytes":SOURCE_BYTES,
          "additional_physical_fp16_lowrank_rank4_and_bias_bytes":DATA_BYTES,
          "CB76_header_bytes":TAIL.size,"CB76_CRC_SHA_bytes":DIGEST_BYTES,
          "actual_complete_CB76_source_bytes":len(final),
          "maximum_allowed_bytes_at_0_700bpw":MAX_PHYSICAL_BYTES,
          "actual_all_in_original_parameter_bpw":8*len(final)/EXPECTED_UNIQUE,
          "original_base_source_sha256":hashlib.sha256(args.original.read_bytes()).hexdigest(),
          "full_paid_CB76_sha256":hashlib.sha256(final).hexdigest(),
          "TRAIN_FIT_corpora":list(FIT_CORPORA),
          "TRAIN_VALID_corpora":list(VALID_CORPORA),
          "source_revision_hashes":{k:json.loads(Path(str(args.folder/"arrays"/(k+"_teacher"))+".json").read_text())["real_source"]
                                    for k in FIT_CORPORA+VALID_CORPORA},
          "checkpoint_steps":list(LEARNING_STEPS),
          "fixed_training_regularizer_B":REG_B,
          "fixed_training_regularizer_rank_factor":REG_UV,
          "fixed_minimum_VALID_gain_each_domain":MIN_GAIN_EACH_VALID_CORPUS,
          "candidate_two_corpus_VALID_heldout_KL":choices,
          "two_corpus_VALID_baseline_KL":bases,
          "maximin_raw_candidate_index":best,
          "chosen_checkpoint_index":selected_idx,
          "chosen_source_mode":selected["mode"],
          "chosen_source_step":selected["step"],
          "correction_tensor_U_shape":[VOCAB,RANK],
          "correction_tensor_V_shape":[DIM,RANK],
          "correction_tensor_bias_shape":[VOCAB],
          "correction_U_nonzero":int(np.count_nonzero(u)),
          "correction_V_nonzero":int(np.count_nonzero(v)),
          "correction_bias_nonzero":int(np.count_nonzero(b)),
          "TEST_was_NOT_read_for_fit_or_validation":True,
          "new_training_sidecar_in_one_reproducible_entire_source_file":True,
          "packed_original_native_C_plus_paid_rank4_CPU_factor_operator":True,
          "source_2D_dense_original_never_reconstructed_in_student":True}
    (args.folder/"v76_selection.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({"stage":"v76_TRAIN_FIT_VALID_global_KL_selected",
          "chosen":selected["mode"],"step":selected["step"],
          "best_unfiltered_gain":round(choices[best]["worst_VALID_gain"],5),
          "chosen_worst_gain":round(choices[selected_idx]["worst_VALID_gain"],5),
          "source_bytes":len(final),
          "all_in_bpw":round(meta["actual_all_in_original_parameter_bpw"],9)}),flush=True)

def summarize(args):
    freeze=json.loads((args.folder/"v76_selection.json").read_text())
    original=args.original.read_bytes()
    corrected=(args.folder/"CALIBRI_v76_paid_full.cb76").read_bytes()
    cb,U,V,b=unpack(corrected)
    if original!=corrected[:SOURCE_BYTES]:
        raise AssertionError("paid lowrank source mutates original CB59 baseline")
    if hashlib.sha256(corrected).hexdigest()!=freeze["full_paid_CB76_sha256"]:
        raise AssertionError("physically stored full source SHA differs from frozen TRAIN")
    out={"experiment":"CALIBRI_v76_REAL_frozen_rank4_full_output_weight_residual",
         "source_model":MODEL,"rank4_sidecar_true_full_file":True,
         "unique_trained_original_parameters":EXPECTED_UNIQUE,
         "original_CB59_bytes":SOURCE_BYTES,
         "corrected_CB76_entire_source_bytes":len(corrected),
         "actual_all_in_bpw":8*len(corrected)/EXPECTED_UNIQUE,
         "TRAIN_only_selection":freeze["chosen_source_mode"],
         "independent_TEST_never_used_in_selection":True,
         "results":{}}
    for domain in TEST_CORPORA:
        source={}
        ids,provenance=get_tokens(domain)
        for name in ("teacher","baseline","corrected"):
            path=args.folder/"arrays"/(domain+"_"+name)
            v=np.load(str(path)+".npy",mmap_mode="r").astype(np.float32)
            meta=json.loads(Path(str(path)+".json").read_text())
            if meta["real_source"]["token_sha256"]!=provenance["token_sha256"]:
                raise AssertionError("unmatched genuine TEST input")
            source[name]=(v,meta)
        if source["corrected"][1]["whole_model"]["physical_complete_model_bytes"]!=len(corrected):
            raise AssertionError("rejected hidden model source bytes")
        t=torch.from_numpy(source["teacher"][0][:,:-1,:].astype(np.float64))
        log_t=t.log_softmax(-1)
        p=log_t.exp()
        scores={}
        for name in ("baseline","corrected"):
            model_logits=torch.from_numpy(source[name][0][:,:-1,:].astype(np.float64))
            log_q=model_logits.log_softmax(-1)
            kl=(p*(log_t-log_q)).sum(-1)
            true_next=ids.reshape(-1,WINDOW)[:,1:]
            nll=-torch.gather(log_q,2,true_next[:,:,None]).squeeze(-1)
            agree=(t.argmax(-1)==model_logits.argmax(-1)).double()
            scores[name]={"KL":kl.mean(-1).numpy(),"NLL":nll.mean(-1).numpy(),
                          "agreement":agree.mean(-1).numpy()}
        domain_results={}
        rng=np.random.default_rng(7601)
        for key in ("KL","NLL","agreement"):
            ba=scores["baseline"][key];ga=scores["corrected"][key]
            delta=ga-ba
            resampled=rng.integers(0,len(delta),(3000,len(delta)))
            q=np.quantile(delta[resampled].mean(-1),[.025,.975])
            domain_results[key]={"original_CB59":float(ba.mean()),
                                 "paid_CB76":float(ga.mean()),
                                 "paid_minus_baseline":float(delta.mean()),
                                 "paired_window_bootstrap_95":[float(q[0]),float(q[1])]}
        domain_results["source"]=provenance
        domain_results["scored_positions"]=len(ids)-len(ids)//WINDOW
        domain_results["cpu_prefill_seconds"]={
            k:source[k][1]["native_model_prefill_seconds"]
            for k in ("teacher","baseline","corrected")}
        out["results"][domain]=domain_results
    out["both_TEST_domains_KL_NLL_improve"]=all(
        out["results"][c]["KL"]["paid_minus_baseline"]<0 and
        out["results"][c]["NLL"]["paid_minus_baseline"]<0 for c in TEST_CORPORA)
    out["no_claim_of_usable_QA_or_real_generation_speed"]=True
    out["source_tooling_note"]="CB59 base original C LUT GEMV; rank4 correction FP32 CPU torch only"
    (args.folder/"v76_independent_TEST.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"stage":"v76_REAL_native_paid_correction_heldout_TEST",
       "correction_mode":freeze["chosen_source_mode"],
       "actual_source_bpw":round(out["actual_all_in_bpw"],9),
       "both_domains_kl_nll_improve":out["both_TEST_domains_KL_NLL_improve"],
       "domains":{c:{"KL_delta":round(out["results"][c]["KL"]["paid_minus_baseline"],5),
                     "NLL_delta":round(out["results"][c]["NLL"]["paid_minus_baseline"],5)}
                  for c in TEST_CORPORA}}),flush=True)

def toy():
    from direct_codebook_v59 import encode
    # The real package proof (CRC/SHA) is independently checked even
    # without trained source in a constrained local toy.
    u,v,b=zero_factors()
    assert u.shape==(VOCAB,RANK) and v.shape==(DIM,RANK) and b.shape==(VOCAB,)
    assert DATA_BYTES==2*(VOCAB*RANK+DIM*RANK+VOCAB)
    assert REQUIRED_PHYSICAL_BYTES<MAX_PHYSICAL_BYTES
    print("CALIBRI v76 paid full-file budget, factor shapes and trailer geometry smoke PASS")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",required=True,choices=("smoke","collect","train","summarize"))
    parser.add_argument("--corpus",choices=ALL_CORPORA,default="wiki_fit")
    parser.add_argument("--kind",choices=("teacher","baseline","corrected"),default="teacher")
    parser.add_argument("--original",type=Path)
    parser.add_argument("--folder",type=Path,default=Path("results"))
    a=parser.parse_args()
    a.folder.mkdir(parents=True,exist_ok=True)
    if a.phase=="smoke":toy()
    elif a.phase=="collect":
        if a.kind=="corrected" and a.corpus not in TEST_CORPORA:
            raise AssertionError("candidate corrected must be frozen on TRAIN before TEST")
        if a.kind=="corrected" and not (a.folder/"v76_selection.json").is_file():
            raise AssertionError("TEST model source not yet frozen on TRAIN")
        if a.kind=="baseline" and a.original is None:
            parser.error("baseline requires --original exact CB59 archive")
        collect(a)
    elif a.phase=="train":
        if a.original is None:parser.error("--original physical source required")
        train(a)
    elif a.phase=="summarize":
        if a.original is None:parser.error("--original physical source required")
        summarize(a)
if __name__=="__main__":main()
