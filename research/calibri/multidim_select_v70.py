#!/usr/bin/env python3
"""CALIBRI v70: preregistered TRAIN-only selection of GQA-safe 2D gauge sources.

All candidates: full pinned real SmolLM2-135M original CB59 archive, EXACT
same source bytes (11,229,303; 0.6678394bpw), native C direct 5-bit execution.
No real pretrained dense 2D tensor is loaded into packed inference processes.
No TEST data participates in fitting/selecting gauges. Research ONLY.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi
from direct_codebook_v59 import CB59, compress_matrix
from direct_runtime_v59 import compile_kernel, NativeKernel, bind_into_llama
from gqa_gauge_v68 import scales_for_pair, gauge, patch_pair
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from quality import DATASET, load_token_ids

H_Q,H_K,HD=9,3,64
LAYERS=(0,15)
# This candidate set is frozen before either TRAIN or TEST inference.
VARIANTS=("baseline","l0_diag","l15_diag","l0_hadamard","l15_hadamard",
          "l0_diag_hadamard","l15_diag_hadamard","both_diag")
TRAIN_START=2048
TEST_START=16384
WINDOW=32
WINDOWS=8
TOKENS=WINDOW*WINDOWS
SOURCE_BYTES=11229303

def make_hadamard(v,o):
    """Orthogonal 2x2 block mixing within EACH KV head.

    GQA repeats each KV value vector for H_Q/H_K distinct Q head outputs.
    For each original V 2-channel pair [a,b], replace by [(a+b),(a-b)]/sqrt2;
    each corresponding repeated O input-column pair gets the SAME transform.
    T.T=T and T*T=I, so original attention function remains invariant.
    """
    repeat=H_Q//H_K
    m,n=v.shape
    assert m==H_K*HD and o.shape[1]==H_Q*HD
    factor=np.float32(1/math.sqrt(2))
    a=v.reshape(H_K,HD,n).copy()
    a0=a[:,0::2,:].copy();a1=a[:,1::2,:].copy()
    a[:,0::2,:]=(a0+a1)*factor
    a[:,1::2,:]=(a0-a1)*factor
    b=o.reshape(o.shape[0],H_K,repeat,HD).copy()
    b0=b[...,0::2].copy();b1=b[...,1::2].copy()
    b[...,0::2]=(b0+b1)*factor
    b[...,1::2]=(b0-b1)*factor
    return a.reshape(v.shape),b.reshape(o.shape)

def expanded_v(v):
    return np.repeat(v.reshape(H_K,HD,v.shape[1]),H_Q//H_K,axis=0).reshape(H_Q*HD,v.shape[1])

def transformed(v,o,kind):
    if kind in ("diag","diag_hadamard"):
        d=scales_for_pair(v,o,H_Q,H_K,HD)
        v,o=gauge(v,o,d,H_Q,H_K,HD)
    if kind in ("hadamard","diag_hadamard"):
        v,o=make_hadamard(v,o)
    return v,o

def variant_layers(name):
    if name=="baseline":return ()
    if name=="both_diag":return ((0,"diag"),(15,"diag"))
    which,kind=name.split("_",1)
    if which not in ("l0","l15") or kind not in ("diag","hadamard","diag_hadamard"):
        raise AssertionError("unknown frozen gauge variant")
    return ((0 if which=="l0" else 15,kind),)

def build(args):
    original=args.source.read_bytes()
    base=CB59(original)
    assert base.meta["original_unique_parameter_count"]==EXPECTED_UNIQUE
    assert len(original)==SOURCE_BYTES
    trained=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).eval()
    cfg=trained.config
    assert (int(cfg.num_attention_heads),int(cfg.num_key_value_heads),
            int(getattr(cfg,"head_dim",cfg.hidden_size//H_Q)))==(H_Q,H_K,HD)
    prepared={}
    for layer in LAYERS:
        pre=f"model.layers.{layer}.self_attn"
        vname=pre+".v_proj.weight";oname=pre+".o_proj.weight"
        v=trained.model.layers[layer].self_attn.v_proj.weight.detach().cpu().numpy().astype(np.float32)
        o=trained.model.layers[layer].self_attn.o_proj.weight.detach().cpu().numpy().astype(np.float32)
        assert v.shape==(H_K*HD,576) and o.shape==(576,H_Q*HD)
        assert tuple(base.handles[vname].shape)==v.shape
        assert tuple(base.handles[oname].shape)==o.shape
        # Re-encode NULL control from exact source through the unchanged codec.
        for name,w in ((vname,v),(oname,o)):
            ref=base.handles[name]
            trial=compress_matrix(w)
            if not (trial["packed"]==ref.packed.tobytes() and
                    np.array_equal(np.frombuffer(trial["centroids"],dtype="<f2"),
                                   ref.centroids.astype("<f2").reshape(-1)) and
                    np.array_equal(np.frombuffer(trial["row_scales"],dtype="<f2"),
                                   ref.row_scales.astype("<f2"))):
                raise AssertionError("NULL CB59 source re-encode differs: "+name)
        for kind in ("diag","hadamard","diag_hadamard"):
            vv,oo=transformed(v,o,kind)
            expected=(o.astype(np.float64)@expanded_v(v).astype(np.float64))
            actual=(oo.astype(np.float64)@expanded_v(vv).astype(np.float64))
            rel=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30))
            if rel>5e-6:raise AssertionError(f"real trained exact GQA operator gauge invalid: {layer}/{kind}/{rel}")
            enc_v=compress_matrix(vv)
            enc_o=compress_matrix(oo)
            prepared[layer,kind]={
                vname:enc_v,oname:enc_o,"_verification":{
                    "layer":layer,"kind":kind,"unquantized_operator_relative_difference":rel}}
    dest=args.folder/"sources";dest.mkdir(parents=True,exist_ok=True)
    audit={"version":70,"model":MODEL,"model_revision":MODEL_SHA,
           "unique_source_parameter_count":EXPECTED_UNIQUE,
           "original_sha256":hashlib.sha256(original).hexdigest(),
           "original_full_physical_bytes":len(original),
           "candidate_quantizer":"unaltered CB59 K32V8 5bits/8 coefficients",
           "candidate_gauge_hyperparameters_source_only":True,
           "gqa_heads":[H_Q,H_K,HD],"variants":{}}
    for choice in VARIANTS:
        changes={}
        checks=[]
        for layer,kind in variant_layers(choice):
            record=prepared[layer,kind]
            checks.append(record["_verification"])
            changes.update({k:v for k,v in record.items() if not k.startswith("_")})
        if changes:
            candidate,_=patch_pair(original,base,changes,EXPECTED_UNIQUE)
        else:candidate=original
        assert len(candidate)==len(original)==SOURCE_BYTES
        file=dest/(choice+".cb59");file.write_bytes(candidate)
        checked=CB59(file.read_bytes())
        assert checked.meta["original_unique_parameter_count"]==EXPECTED_UNIQUE
        audit["variants"][choice]={
            "source_sha256":hashlib.sha256(candidate).hexdigest(),
            "actual_bytes":len(candidate),
            "full_source_bpw":8*len(candidate)/EXPECTED_UNIQUE,
            "transforms":checks,
            "identical_full_CB59_runtime_abi":True}
        print(json.dumps({"candidate":choice,"bytes":len(candidate),
                          "invariance":max([z["unquantized_operator_relative_difference"]
                                           for z in checks],default=0)}),flush=True)
    args.folder.mkdir(parents=True,exist_ok=True)
    (args.folder/"v70_build.json").write_text(json.dumps(audit,indent=2)+"\n")

def get_ids(split):
    start=TRAIN_START if split=="train" else TEST_START
    end=start+TOKENS
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    ds_rev=HfApi().dataset_info(DATASET).sha
    ids=load_token_ids(tokenizer,split,ds_rev,end)[start:end]
    assert len(ids)==TOKENS
    return ids,ds_rev,start,end

def run(args):
    ids,ds_rev,start,end=get_ids(args.split)
    name=args.variant
    if args.phase=="teacher":
        assert name=="teacher"
        model=AutoModelForCausalLM.from_pretrained(
            MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
            attn_implementation="eager").eval()
        physical=None
        archive_sha=None
        evidence=None
    else:
        source=args.folder/"sources"/(name+".cb59")
        archive=CB59(source.read_bytes())
        assert archive.meta["original_unique_parameter_count"]==EXPECTED_UNIQUE
        assert archive.physical_bytes()==SOURCE_BYTES
        cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
        cfg._attn_implementation="eager"
        model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
        kernel=NativeKernel(compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"),
                args.folder/("libcalibri_v70_"+args.split+"_"+name+".so")))
        evidence=bind_into_llama(model,archive,kernel)
        assert evidence["runtime_dense_multi_dim_source_parameters"]==0
        physical=archive.physical_bytes()
        archive_sha=archive.hash()
    with torch.inference_mode():
        windows=[];timings=[]
        for ix in range(0,len(ids),WINDOW):
            x=ids[ix:ix+WINDOW][None]
            tic=time.perf_counter()
            logits=model(input_ids=x,use_cache=False).logits.detach().cpu().numpy()[0].astype(np.float32)
            timings.append(time.perf_counter()-tic)
            assert logits.shape[0]==WINDOW and np.isfinite(logits).all()
            windows.append(logits)
    array=np.stack(windows)
    folder=args.folder/"logits";folder.mkdir(parents=True,exist_ok=True)
    path=folder/(args.split+"_"+name)
    np.save(str(path)+".npy",array)
    meta={"split":args.split,"name":name,"source_model":MODEL,
          "model_revision":MODEL_SHA,"data_revision":ds_rev,
          "token_span":[start,end],"token_sha256":hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
          "windows":WINDOWS,"context":WINDOW,
          "physical_entire_pretrained_source_bytes":physical,
          "physical_sha256":archive_sha,
          "hot_original_trained_2D_parameters":0 if evidence is not None else None,
          "native_packed_C_5bit":evidence is not None,
          "total_cpu_prefill_seconds":sum(timings),"timings_seconds":timings}
    (Path(str(path)+".json")).write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({"phase":args.phase,"split":args.split,"variant":name,
                      "bytes":physical,"timing":round(sum(timings),3)}),flush=True)

def score_candidate(t,p,ids):
    teacher=torch.from_numpy(t[:,:-1,:].astype(np.float64))
    student=torch.from_numpy(p[:,:-1,:].astype(np.float64))
    lpt=teacher.log_softmax(dim=-1)
    lps=student.log_softmax(dim=-1)
    probs=lpt.exp()
    kl=(probs*(lpt-lps)).sum(-1)
    truth=ids.reshape(WINDOWS,WINDOW)[:,1:]
    nll=-torch.gather(lps,2,truth[:,:,None]).squeeze(-1)
    agree=(teacher.argmax(-1)==student.argmax(-1)).double()
    return {"KL":float(kl.mean()),"NLL":float(nll.mean()),
            "top1_agreement":float(agree.mean()),
            "KL_per_window":[float(z) for z in kl.mean(-1)],
            "NLL_per_window":[float(z) for z in nll.mean(-1)]}

def load_logits(folder,split,name):
    root=folder/"logits"/(split+"_"+name)
    m=json.loads(Path(str(root)+".json").read_text())
    data=np.load(str(root)+".npy")
    assert tuple(data.shape[:2])==(WINDOWS,WINDOW)
    return data,m

def select(args):
    test=args.split=="test"
    teacher,tmeta=load_logits(args.folder,args.split,"teacher")
    ids,rev,start,end=get_ids(args.split)
    assert rev==tmeta["data_revision"]
    hashes=[]
    results={}
    times={}
    source=json.loads((args.folder/"v70_build.json").read_text())
    # Only candidate list from preregistration, independent of any TEST measurement.
    names=VARIANTS if not test else ("baseline",
            json.loads((args.folder/"v70_train_selection.json").read_text())["chosen"])
    names=tuple(dict.fromkeys(names))
    for name in names:
        logits,meta=load_logits(args.folder,args.split,name)
        assert meta["token_sha256"]==tmeta["token_sha256"]
        assert meta["native_packed_C_5bit"] is True and meta["hot_original_trained_2D_parameters"]==0
        assert meta["physical_entire_pretrained_source_bytes"]==SOURCE_BYTES
        assert meta["physical_sha256"]==source["variants"][name]["source_sha256"]
        hashes.append(meta["physical_sha256"])
        results[name]=score_candidate(teacher,logits,ids)
        times[name]=meta["total_cpu_prefill_seconds"]
    if not test:
        # Frozen TRAIN-only decision: >0.10 nats improvement over baseline.
        # The exact threshold and candidates were fixed before TEST access.
        baseline=results["baseline"]["KL"]
        winner=min(VARIANTS,key=lambda k:(results[k]["KL"],VARIANTS.index(k)))
        chosen=winner if results[winner]["KL"]<baseline-0.10 else "baseline"
        report={"protocol":"v70 frozen TRAIN-only 8 variants, primary teacher KL",
                "split":"train","TRAIN_token_span":[start,end],
                "source_sha256_count":len(set(hashes)),
                "TEST_was_NOT_read_for_selection":True,"selection_min_KL_delta":0.10,
                "chosen":chosen,"best_raw_train_candidate":winner,
                "TRAIN_teacher_KL_by_variant":{k:v["KL"] for k,v in results.items()},
                "TRAIN_next_token_NLL_by_variant":{k:v["NLL"] for k,v in results.items()}}
        (args.folder/"v70_train_selection.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"stage":"v70_train_selection","winner":chosen,
                          "baseline_train_KL":round(baseline,6),
                          "raw_best_train_KL":round(results[winner]["KL"],6)}),flush=True)
    else:
        selection=json.loads((args.folder/"v70_train_selection.json").read_text())
        winner=selection["chosen"]
        if winner not in results:raise AssertionError("chosen candidate not evaluated on TEST")
        base=results["baseline"]
        current=results[winner]
        delta=np.array(current["KL_per_window"])-np.array(base["KL_per_window"])
        rg=np.random.default_rng(70)
        choices=rg.integers(0,WINDOWS,size=(1000,WINDOWS))
        boot=delta[choices].mean(axis=1)
        lo,hi=[float(v) for v in np.percentile(boot,[2.5,97.5])]
        report={"experiment":"CALIBRI_v70_REAL_135M_2D_GQA_gauge_TRAIN_selection_heldout_TEST",
                "model":MODEL,"unique_trained_params":EXPECTED_UNIQUE,
                "all_candidate_full_physical_bytes":SOURCE_BYTES,
                "all_candidate_full_bpw":8*SOURCE_BYTES/EXPECTED_UNIQUE,
                "TRAIN_choice_fixed_before_TEST":True,"train_selection":selection,
                "heldout_split":"test","TEST_token_span":[start,end],
                "next_token_positions":WINDOWS*(WINDOW-1),
                "native_CPU_no_trained_dense_2D":True,
                "full_model_results":results,"native_CPU_prefill_seconds":times,
                "chosen_minus_baseline_TEST_KL":current["KL"]-base["KL"],
                "chosen_minus_baseline_TEST_NLL":current["NLL"]-base["NLL"],
                "paired_8_window_bootstrap_KL_delta_95_interval":[lo,hi],
                "confidence_note":"Only 8 windows; bootstrap exploratory, not task accuracy proof.",
                "no_robust_quality_or_speed_claim":True,
                "task_QA_not_measured":True}
        (args.folder/"v70_independent_TEST.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"stage":"v70_independent_full_model_TEST","chosen":winner,
                          "baseline_KL":round(base["KL"],5),"chosen_KL":round(current["KL"],5),
                          "delta_KL":round(report["chosen_minus_baseline_TEST_KL"],5),
                          "baseline_NLL":round(base["NLL"],5),"chosen_NLL":round(current["NLL"],5),
                          "delta_NLL":round(report["chosen_minus_baseline_TEST_NLL"],5),
                          "positions":WINDOWS*(WINDOW-1)}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",required=True,choices=("build","teacher","packed","select"))
    p.add_argument("--split",choices=("train","test"),default="train")
    p.add_argument("--variant",default="teacher")
    p.add_argument("--source",type=Path)
    p.add_argument("--folder",type=Path,default=Path("results"))
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    if a.phase=="build":
        if a.source is None:raise ValueError("--source required")
        build(a)
    elif a.phase in ("teacher","packed"):
        if a.phase=="packed" and a.variant not in VARIANTS:
            raise ValueError("unknown variant")
        run(a)
    else:select(a)

if __name__=="__main__":main()
