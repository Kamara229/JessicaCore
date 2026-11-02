#!/usr/bin/env python3
"""CALIBRI v46: practical all-model INT6 storage with measured warm inference SLA.

Speed-first product tier: complete original-trained Pythia-14M parameters
physically stored rowwise INT6 + small exact raw vectors, then decode ONCE at
cold startup into a hot FP32 runtime cache. This is INT6 *disk/checkpoint*
compression, NOT INT6 working-RAM compression or packed native compute.

PRE: frozen TRAIN-A [0:2048] calibrates 12 sensitive MLP matrices.
POST: disjoint TRAIN-B [2048:4096] fits output-affine teacher correction.
Offline original weights do not exist in the fresh from_config runtime.
Measures REAL full-model 128-token prefill + 32 KV-cache decode steps,
same teacher-forced input IDs, same CPU, same shapes, warmed 2+7 runs.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time,statistics
from pathlib import Path
import numpy as np
import torch

from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from huggingface_hub import HfApi

from full_checkpoint_v46 import pack_state,unpack_state,read_runtime,smoke as checkpoint_smoke
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode
from pareto_fixed_highbits_v43 import trained_fixedbit
from all_mlp_agnews_v36 import module_identity
from compensation_bridge_v45 import make_sidecar,open_sidecar
from quality import MODEL,DATASET,load_token_ids,module_selection,calibrate,ppl_eval

TRAIN_A=2048
TRAIN_B=2048
CONTEXT=128
DECODE_STEPS=32
WARMUPS=2
REPEATS=7
TOTAL_MAX_BPW=6.5
MAX_QUALITY_REGRESSION=1.05
MAX_LATENCY_RATIO=1.20
CORPORA_NEW=(("ARC_Challenge_test_disjoint","allenai/ai2_arc","ARC-Challenge","test[:128]"),
             ("Yelp_test_192_256_disjoint","Yelp/yelp_review_full",None,"test[192:256]"))


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def affine_fit_on_B(source,q,x):
    """Fixed no-validation affine per output channel on DISJOINT B activations."""
    if source.shape!=q.shape or x.ndim!=2 or x.shape!=(TRAIN_B,source.shape[1]):
        raise AssertionError("post calibration must use source-trained real B inputs")
    pred=(x@q.T).astype(np.float64)
    gold=(x@source.T).astype(np.float64)
    pm=pred.mean(axis=0)
    tm=gold.mean(axis=0)
    xp=pred-pm
    yt=gold-tm
    var=np.sum(xp*xp,axis=0)
    ridge=0.05*var+1e-8
    gain=(np.sum(xp*yt,axis=0)+ridge)/(var+ridge)
    bias=tm-gain*pm
    if not np.isfinite(gain).all() or not np.isfinite(bias).all():
        raise AssertionError("nonfinite POST correction parameters")
    return {"gain":gain.astype(np.float16).astype(np.float32),
            "bias":bias.astype(np.float16).astype(np.float32)}


def add_affine_hooks(model,sidecar):
    hooks=[]
    for name,p in sidecar.items():
        mod=model.get_submodule(name)
        gg=torch.from_numpy(p["gain"]).to(device=mod.weight.device,dtype=mod.weight.dtype)
        bb=torch.from_numpy(p["bias"]).to(device=mod.weight.device,dtype=mod.weight.dtype)
        def hook(module,inputs,output,g=gg,b=bb):
            return output*g+b
        hooks.append(mod.register_forward_hook(hook))
    return hooks


def load_new_corpus(tokenizer,name,repository,config,split,rev):
    from datasets import load_dataset
    ds=load_dataset(repository,config,split=split,revision=rev) if config else load_dataset(
        repository,split=split,revision=rev)
    rows=[]
    for row in ds:
        if "ARC_Challenge" in name:
            ch=row["choices"]
            s=("Question: "+str(row["question"])+"\nOptions: "+
               "; ".join(str(a)+") "+str(b) for a,b in zip(
                   ch.get("label",[]),ch.get("text",[]))))
        else:
            s=str(row["text"])
        if s.strip():
            rows.append(s)
    if not rows:
        raise AssertionError("first-use quality text empty")
    tokens=tokenizer("\n\n".join(rows),add_special_tokens=False,
                     return_tensors="pt").input_ids[0,:2048]
    if len(tokens)!=2048:
        raise AssertionError("new first-use corpus shorter than 2048 tokens")
    return tokens


def batch_kernel(model,prompt,follow):
    """Exactly the same real HuggingFace KV-cache autoregressive workload."""
    with torch.inference_mode():
        start=time.perf_counter_ns()
        out=model(input_ids=prompt,use_cache=True)
        midpoint=time.perf_counter_ns()
        past=out.past_key_values
        for i in range(len(follow)):
            out=model(input_ids=follow[i:i+1],past_key_values=past,use_cache=True)
            past=out.past_key_values
        end=time.perf_counter_ns()
        # Force one tiny actual read, preventing accidental unused output.
        if not torch.isfinite(out.logits).all():
            raise AssertionError("nonfinite warm forward")
    return ((midpoint-start)/1e6,(end-midpoint)/1e6,(end-start)/1e6)


def timing(model,prompt,follow):
    if len(prompt)!=CONTEXT or len(follow)!=DECODE_STEPS:
        raise AssertionError("timing token geometry must be fixed")
    model.eval()
    prompt=prompt.reshape(1,CONTEXT)
    follow=follow.reshape(-1,1)
    for _ in range(WARMUPS):
        batch_kernel(model,prompt,follow)
    samples=[batch_kernel(model,prompt,follow) for _ in range(REPEATS)]
    keys=("prefill_ms","decode32_ms","total_prefill_and_decode32_ms")
    result={}
    for k,ind in zip(keys,range(3)):
        values=[round(sample[ind],6) for sample in samples]
        result[k]={"raw":values,
                   "median":statistics.median(values),
                   "min":min(values),"max":max(values)}
    return result


def ppl(model,corpora):
    model.eval()
    results={}
    for name,ids in corpora.items():
        with torch.inference_mode():
            metrics=ppl_eval(model,ids,CONTEXT)
        results[name]={"ppl":float(metrics["ppl"]),
                       "nll":float(metrics["nll"])}
        print(json.dumps({"model":getattr(model,"_calibri_variant","unknown"),
                          "corpus":name,"PPL":results[name]["ppl"]}),flush=True)
    return results


def run(args):
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2"
            and os.getenv("DNNL_MAX_CPU_ISA")=="AVX2"
            and os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2"
            and str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("CPU baseline cannot be compared unless AVX2 pinned")
    torch.manual_seed(46)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checkpoint_smoke()

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    api=HfApi()
    model_rev=api.model_info(MODEL).sha
    data_rev=api.dataset_info(DATASET).sha
    new_revs={name:api.dataset_info(repo).sha for name,repo,_,_ in CORPORA_NEW}
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
    config=AutoConfig.from_pretrained(MODEL,revision=model_rev)
    teacher=AutoModelForCausalLM.from_pretrained(MODEL,revision=model_rev,
                                                torch_dtype=torch.float32)
    teacher.eval()
    teacher._calibri_variant="original_teacher"
    frozen={k:v.detach().cpu().contiguous().clone() for k,v in teacher.state_dict().items()}
    numel=sum(x.numel() for x in frozen.values())
    source_bytes=sum(x.numel()*x.element_size() for x in frozen.values())
    model_state_sha=sha(b"".join(k.encode()+v.numpy().tobytes() for k,v in sorted(frozen.items())))
    original_mlp=module_selection(teacher)
    modules=sorted(original_mlp,key=module_identity)
    if len(modules)!=12 or sum(frozen[n+".weight"].numel() for n in modules)!=786432:
        raise AssertionError("source trained Pythia original 12 MLP modules changed")

    training=load_token_ids(tokenizer,"train",data_rev,TRAIN_A+TRAIN_B)
    a,b=training[:TRAIN_A],training[TRAIN_A:]
    if len(a)!=2048 or len(b)!=2048 or sha(a.numpy().tobytes())==sha(b.numpy().tobytes()):
        raise AssertionError("pre and post calibration TRAIN windows overlap or missing")
    with torch.inference_mode():
        pre_x=calibrate(teacher,original_mlp,a,CONTEXT)
        post_x=calibrate(teacher,original_mlp,b,CONTEXT)
    encoded_pre={}
    affine={}
    for name in modules:
        w=frozen[name+".weight"].numpy().astype(np.float32)
        xa=pre_x[name].numpy().astype(np.float32)
        xb=post_x[name].numpy().astype(np.float32)
        moments=np.mean(xa*xa,axis=0,dtype=np.float32)
        before,after,info=trained_fixedbit(w,moments,6)
        if len(before)!=len(after) or info["TRAIN_diagonal_weighted_MSE_after"]>info["TRAIN_diagonal_weighted_MSE_before"]*1.000001:
            raise AssertionError("PRE trained row calibration changed physical bytes or worsened its proxy")
        encoded_pre[name+".weight"]=after
        affine[name]=affine_fit_on_B(w,fixed_decode(after),xb)

    raw,raw_manifest=pack_state(frozen)
    pre,pre_manifest=pack_state(frozen,preencoded=encoded_pre)
    original_2d=sum(v.numel() for v in frozen.values() if v.ndim==2 and v.is_floating_point())
    if original_2d<0.90*numel:
        raise AssertionError("unexpected source: under 90% matrix coefficients compressed")
    disk_files={}
    for tag,data in (("raw",raw),("pre",pre)):
        file=out.parent/("CALIBRI_v46_all_weights_"+tag+".cb46")
        file.write_bytes(data)
        actual=file.read_bytes()
        if actual!=data or len(actual)<=0:
            raise AssertionError("failed physical checkpoint file reread")
        decoded=unpack_state(actual,expected_keys=frozen)
        if any(decoded[k].shape!=v.shape for k,v in frozen.items()):
            raise AssertionError("physical file is missing an original weight shape")
        digest=sha(actual)
        disk_files[tag]={"file":str(file),"physical_bytes":len(actual),
                         "SHA256":digest,"model_state_param_count":numel,
                         "effective_bpw":8*len(actual)/numel}
        del decoded
    if disk_files["pre"]["physical_bytes"]!=disk_files["raw"]["physical_bytes"]:
        raise AssertionError("PRE calibration must be byte-neutral at full model rate")
    if disk_files["pre"]["effective_bpw"]>TOTAL_MAX_BPW:
        raise AssertionError("INT6-level full model file exceeds predeclared 6.5 physical bpw")

    correction=make_sidecar(affine,"affine")
    sidepath=out.parent/"CALIBRI_v46_int6_post_affine.c45c"
    sidepath.write_bytes(correction)
    correction=sidepath.read_bytes()
    parsed=open_sidecar(correction,"affine")
    disk_files["post"]={"file":str(sidepath),"physical_bytes":len(correction),
                        "SHA256":sha(correction),
                        "total_checkpoint_plus_sidecar_bytes":len(pre)+len(correction),
                        "all_parameter_effective_bpw":8*(len(pre)+len(correction))/numel}
    if disk_files["post"]["all_parameter_effective_bpw"]>TOTAL_MAX_BPW:
        raise AssertionError("pre+post model storage crossed 6.5bpw upper gate")
    damaged=bytearray(correction)
    damaged[-2]^=1
    try:open_sidecar(bytes(damaged),"affine")
    except ValueError:pass
    else:raise AssertionError("corrupted POST correction accepted")

    # Quality texts first accessed AFTER all PRE and POST training finished
    # and full physical checkpoints were finalized and reread/verified.
    corpora={"WikiText2_DEV_known":load_token_ids(tokenizer,"validation",data_rev,1024)}
    for name,repo,conf,split in CORPORA_NEW:
        corpora[name]=load_new_corpus(tokenizer,name,repo,conf,split,new_revs[name])
    tokenhash={k:sha(v.numpy().tobytes()) for k,v in corpora.items()}
    timed=a[:CONTEXT+DECODE_STEPS].clone()
    if len(timed)!=CONTEXT+DECODE_STEPS:
        raise AssertionError("REAL input timing requires 160 train tokens")
    prompt,follow=timed[:CONTEXT],timed[CONTEXT:]
    experiments={}
    experiments["original_teacher"]={"ppl":ppl(teacher,corpora),
                                     "warm_timing":timing(teacher,prompt,follow),
                                     "original_checkpoint_bpw_F32_reference":8*source_bytes/numel}

    def fresh_instance(filename):
        begin=time.perf_counter()
        m=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32)
        res=read_runtime(m,filename)
        elapsed=time.perf_counter()-begin
        m.eval()
        # No HF pretrained checkpoint is loaded here. The from_config
        # parameters are overwritten only from the physically stored CB46.
        return m,{**res,"cold_random_init_plus_CB46_decoding_seconds":elapsed}

    for tag,archive,post in (("full_INT6_raw","raw",False),
                              ("full_INT6_PRE","pre",False),
                              ("full_INT6_PRE_POST","pre",True)):
        runtime,loadinfo=fresh_instance(disk_files[archive]["file"])
        runtime._calibri_variant=tag
        hooks=add_affine_hooks(runtime,parsed) if post else []
        try:
            score=ppl(runtime,corpora)
            speed=timing(runtime,prompt,follow)
            experiments[tag]={"cold_loader":loadinfo,"ppl":score,"warm_timing":speed}
        finally:
            for hook in hooks:hook.remove()
            del runtime

    teacher_latency=experiments["original_teacher"]["warm_timing"]
    external_names=[row[0] for row in CORPORA_NEW]
    for tag,record in experiments.items():
        if tag=="original_teacher":
            continue
        record["SLA_quality_5pct_each_new_corpus"]={
            name:{"PPL_ratio_vs_original_teacher":record["ppl"][name]["ppl"]/
                                                     experiments["original_teacher"]["ppl"][name]["ppl"],
                  "pass":record["ppl"][name]["ppl"]<=MAX_QUALITY_REGRESSION*
                          experiments["original_teacher"]["ppl"][name]["ppl"]}
            for name in external_names}
        record["SLA_warm_latency_20pct"]={
            key:{"ratio_vs_uncompressed_teacher":
                 record["warm_timing"][key]["median"]/teacher_latency[key]["median"],
                 "pass":record["warm_timing"][key]["median"]<=
                        MAX_LATENCY_RATIO*teacher_latency[key]["median"]}
            for key in ("prefill_ms","decode32_ms","total_prefill_and_decode32_ms")}
        record["overall_practical_stage_gate"]=(
            all(v["pass"] for v in record["SLA_quality_5pct_each_new_corpus"].values()) and
            record["SLA_warm_latency_20pct"]["total_prefill_and_decode32_ms"]["pass"] and
            disk_files["post" if tag.endswith("POST") else ("raw" if tag.endswith("raw") else "pre")][
                "all_parameter_effective_bpw" if tag.endswith("POST") else "effective_bpw"] <=TOTAL_MAX_BPW)
    result={
        "experiment":"CALIBRI_v46_all_model_true_INT6_physical_checkpoint_speed_first_hot_cache",
        "source_model":MODEL,"model_revision":model_rev,
        "wiki_revision":data_rev,"external_revision":new_revs,
        "original_source_state_SHA256":model_state_sha,
        "all_parameter_count":numel,"two_dimensional_matrix_coefficients":original_2d,
        "full_source_FP32_state_theoretical_bytes":source_bytes,
        "train_A_first2048_SHA256":sha(a.numpy().tobytes()),
        "train_B_disjoint2048_SHA256":sha(b.numpy().tobytes()),
        "eval_token_SHA256":tokenhash,
        "pinned_timing": {"CPU_ATen":str(torch._C._get_cpu_capability()),
            "training_CPU_threads":torch.get_num_threads(),"context_tokens":CONTEXT,
            "autoregressive_KV_cache_decode_steps":DECODE_STEPS,
            "warmups":WARMUPS,"repetitions":REPEATS,
            "same_original_TRAIN_A_token_input_trace_for_all_variants":True},
        "physical_checkpoint_files":disk_files,
        "variant_results":experiments,
        "SLA":{"max_effective_disk_bpw_incl_real_metadata":TOTAL_MAX_BPW,
            "max_PPL_ratio_each_new_corpus":MAX_QUALITY_REGRESSION,
            "max_warm_total_runtime_ratio":MAX_LATENCY_RATIO,
            "original_teacher_comparison_is_fp32":True},
        "limitations":[
            "This is ACTUAL full-model physical compressed coefficient storage for this Pythia model, not just 12 MLP matrices. Config/tokenizer must accompany checkpoint.",
            "Speed-first hot-cache tier fully expands compressed INT6 matrices to FP32 in RAM at model load. No compressed-RAM or native packed-GEMM gain claimed.",
            "Measured warm throughput excludes one-time file decode; cold_random_init_plus_decode time is recorded separately, teacher cold load from hub is not a fair cold baseline.",
            "A CPU-only batch-1 128-context prefill plus 32 teacher-forced autoregressive KV-cache steps, no end-user Jessica chat/API latency.",
            "Only 12 MLP matrices have PRE trained calibration and POST affine correction; attention and huge embeddings are INT6 raw rowwise.",
            "Fixed 2048-token external excerpts are exploratory; quality limit 5% PPL is a proposed project-stage engineering gate, not a user-specified quality tolerance.",
            "No SwitchTransformer 0.8bpw, native packed inference, true production Jessica integration or changed deployed Jessica files."
        ]
    }
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"RESULT":"CALIBRI_v46_REAL",
                      "size":{k:{"bytes":v["physical_bytes"],"bpw":v.get("effective_bpw",
                         v.get("all_parameter_effective_bpw"))} for k,v in disk_files.items()},
                      "SLA":{k:{"practical":v.get("overall_practical_stage_gate"),
                                "PPL":{name:round(score["ppl"],4)
                                       for name,score in v["ppl"].items()},
                                "warm_total_ms":v["warm_timing"]["total_prefill_and_decode32_ms"]["median"],
                                "ratio":v.get("SLA_warm_latency_20pct",{}).get(
                                     "total_prefill_and_decode32_ms",{}).get("ratio_vs_uncompressed_teacher")}
                             for k,v in experiments.items()}},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/calibri_v46.json")
    run(p.parse_args())
