#!/usr/bin/env python3
"""CALIBRI v74: TRAIN-only *joint FULL packed-model* KL variant selection.

Frozen candidate pool combines independent real pretrained v70 paired 2D
GQA-Hadamard V/O representation and the four v73 locally 8D-activation
covariance-shaped 5bit symbol streams, in exact complete native CB59 source.
Candidates never acquire side metadata, dense source, or a new CPU kernel.
12 candidate models are evaluated using true native-packed whole-model
outputs on unseen TRAIN windows; choose by full vocabulary teacher KL, then
open two separate frozen TEST corpora. Negative outcomes remain valid.

Research only. Does not modify production Jessica, Render, Supabase, APK.
"""
from __future__ import annotations
import argparse, hashlib, json, math, struct, time, zlib
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from direct_codebook_v59 import CB59, HEADER
from direct_runtime_v59 import compile_kernel,NativeKernel,bind_into_llama
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from quality import DATASET,load_token_ids

ORIGINAL_BYTES=11229303
TRAIN_START=4096
WIKI_TEST_START=50000
WINDOW=64
TRAIN_WINDOWS=4
TEST_WINDOWS=8
AG_DATASET="fancyzhx/ag_news"
COV_NAMES=(
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.15.mlp.gate_proj.weight",
    "model.layers.29.mlp.up_proj.weight",
)
GAUGE_NAMES=(
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
)
COV_TAGS=("q","down","gate","up")
VARIANTS=("baseline","hadamard","cov_all","hadamard_cov_all",
          "cov_q","cov_down","cov_gate","cov_up",
          "hadamard_cov_q","hadamard_cov_down","hadamard_cov_gate","hadamard_cov_up")
# The candidate list and minimum TRAIN improvement are frozen before any
# official TEST input is loaded; the experiment does NOT reoptimize on TEST.
MIN_TRAIN_GAIN=0.05

def get_conditions(name):
    if name not in VARIANTS: raise ValueError("unregistered v74 representation")
    use_h=(name=="hadamard" or name.startswith("hadamard_"))
    cov=()
    if name.endswith("cov_all"):cov=COV_NAMES
    else:
        for tag,key in zip(COV_TAGS,COV_NAMES):
            if name.endswith("cov_"+tag):cov=(key,)
    return use_h,cov

def rebuild(base_bytes,gauge_bytes,cov_bytes,use_h,cov_keys):
    """Only replace previously physically encoded CB59 tensor payloads."""
    original=CB59(base_bytes)
    gauge=CB59(gauge_bytes)
    cov=CB59(cov_bytes)
    if len({len(base_bytes),len(gauge_bytes),len(cov_bytes)})!=1:
        raise AssertionError("source physical mismatch")
    if not (original.meta==gauge.meta==cov.meta):
        raise AssertionError("one source changed trained model manifest")
    _,_,mlen,_=HEADER.unpack_from(base_bytes)
    offset=HEADER.size+mlen
    result=bytearray(base_bytes)
    names=[]
    for key in GAUGE_NAMES if use_h else ():
        names.append((key,("packed","row_scales","centroids"),gauge_bytes))
    for key in cov_keys:
        names.append((key,("packed",),cov_bytes))
    for key,parts,donor in names:
        for part in parts:
            off,n=original.meta["tensor_descriptors"][key]["parts"][part]
            result[offset+off:offset+off+n]=donor[offset+off:offset+off+n]
    prefix=bytes(result[:-36])
    protected=prefix+struct.pack("<I",zlib.crc32(prefix))
    candidate=protected+hashlib.sha256(protected).digest()
    if len(candidate)!=ORIGINAL_BYTES:raise AssertionError("new complete file byte count")
    parsed=CB59(candidate)
    if set(parsed.handles)!=set(original.handles) or set(parsed.raw)!=set(original.raw):
        raise AssertionError("trained source missing after graft")
    if parsed.meta!=original.meta:raise AssertionError("model source metadata changed")
    return candidate

def build(args):
    base=args.original.read_bytes()
    h=args.gauge.read_bytes()
    cov=args.covariance.read_bytes()
    o=CB59(base)
    assert o.meta["original_unique_parameter_count"]==EXPECTED_UNIQUE
    if len(base)!=ORIGINAL_BYTES:raise AssertionError("not a physically complete original pretrained source")
    out=args.folder/"sources";out.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name in VARIANTS:
        use_h,keys=get_conditions(name)
        candidate=rebuild(base,h,cov,use_h,keys)
        f=out/(name+".cb59")
        f.write_bytes(candidate)
        if hashlib.sha256(f.read_bytes()).hexdigest()!=hashlib.sha256(candidate).hexdigest():
            raise AssertionError("persisted source hash differs")
        manifest[name]={"source_SHA256":hashlib.sha256(candidate).hexdigest(),
                        "physical_complete_bytes":len(candidate),
                        "complete_bpw":8*len(candidate)/EXPECTED_UNIQUE,
                        "uses_Hadamard":use_h,"uses_covariance_streams":list(keys),
                        "same_original_native_K32V8_5bit_ABI":True}
        if name=="baseline" and candidate!=base:raise AssertionError("baseline not original")
        if name=="hadamard" and candidate!=h:raise AssertionError("gauge source not exact v70")
        if name=="cov_all" and candidate!=cov:raise AssertionError("cov source not exact v73")
    (args.folder/"v74_source_manifest.json").write_text(
        json.dumps({"experiment":"v74 fixed full physical 135M source candidates",
                    "source_model":MODEL,"model_revision":MODEL_SHA,
                    "unique_pretrained_parameters":EXPECTED_UNIQUE,
                    "training_only_selection":True,
                    "all_sources_same_bytes":ORIGINAL_BYTES,
                    "candidates":manifest},indent=2)+"\n")
    print(json.dumps({"stage":"v74_real_sources","candidate_count":len(VARIANTS),
                      "source_bytes_each":ORIGINAL_BYTES,
                      "bpw":8*ORIGINAL_BYTES/EXPECTED_UNIQUE}),flush=True)

def tokens(corpus):
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    if corpus=="train":
        ds_rev=HfApi().dataset_info(DATASET).sha
        n=TRAIN_WINDOWS*WINDOW
        result=load_token_ids(tok,"train",ds_rev,TRAIN_START+n)[TRAIN_START:TRAIN_START+n]
        origin={"dataset":DATASET,"split":"train","revision":ds_rev,
                "token_span":[TRAIN_START,TRAIN_START+n]}
    elif corpus=="wiki_test":
        ds_rev=HfApi().dataset_info(DATASET).sha
        n=TEST_WINDOWS*WINDOW
        result=load_token_ids(tok,"test",ds_rev,WIKI_TEST_START+n)[WIKI_TEST_START:WIKI_TEST_START+n]
        origin={"dataset":DATASET,"split":"test","revision":ds_rev,
                "token_span":[WIKI_TEST_START,WIKI_TEST_START+n]}
    elif corpus=="news_test":
        ds_rev=HfApi().dataset_info(AG_DATASET).sha
        n=TEST_WINDOWS*WINDOW
        ds=load_dataset(AG_DATASET,split="test[1000:1100]",revision=ds_rev)
        text="\n\n".join(str(row["text"]) for row in ds)
        result=tok(text,return_tensors="pt",add_special_tokens=False).input_ids[0,:n]
        origin={"dataset":AG_DATASET,"split":"test[1000:1100]","revision":ds_rev,
                "token_span":[0,n],"only_record_text_no_label":True}
    else:raise ValueError(corpus)
    if len(result)!=n:raise AssertionError("insufficient real independent corpus tokens")
    origin["token_SHA256"]=hashlib.sha256(result.numpy().tobytes()).hexdigest()
    return result,origin

def create(kind,args):
    if kind=="teacher":
        model=AutoModelForCausalLM.from_pretrained(
            MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
            attn_implementation="eager").eval()
        return model,{"teacher_original_full_source_loaded":True}
    source=args.folder/"sources"/(kind+".cb59")
    blob=source.read_bytes()
    cb=CB59(blob)
    if cb.physical_bytes()!=ORIGINAL_BYTES or cb.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE:
        raise AssertionError("wrong full actual trained source")
    manifest=json.loads((args.folder/"v74_source_manifest.json").read_text())
    if cb.hash()!=manifest["candidates"][kind]["source_SHA256"]:
        raise AssertionError("source hash drift")
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32).eval()
    kernel=NativeKernel(compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"),
                         args.folder/("lib_v74_"+kind+".so")))
    audit=bind_into_llama(model,cb,kernel)
    if audit["runtime_dense_multi_dim_source_parameters"]!=0:
        raise AssertionError("packed inference has trained dense source")
    return model,{"pretrained_teacher_loaded":False,
                  "source_sha256":cb.hash(),"full_source_bytes":cb.physical_bytes(),
                  "source_bpw":8*cb.physical_bytes()/EXPECTED_UNIQUE,
                  "native_C_direct_5bit":True,"trained_dense_2D_hot_parameters":0,
                  "tied_alias":audit["tied_embed_and_output_codebook_SHARED_IN_RAM"]}

def run(args):
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    ids,origin=tokens(args.corpus)
    model,audit=create(args.kind,args)
    nwin=TRAIN_WINDOWS if args.corpus=="train" else TEST_WINDOWS
    if len(ids)!=nwin*WINDOW:raise AssertionError("wrong context token count")
    folder=args.folder/"scores";folder.mkdir(parents=True,exist_ok=True)
    reference=folder/("teacher_"+args.corpus+".f16")
    if args.kind=="teacher":
        logits_store=np.memmap(reference,mode="w+",dtype="<f2",shape=(nwin,WINDOW,49152))
    else:
        logits_store=np.memmap(reference,mode="r",dtype="<f2",shape=(nwin,WINDOW,49152))
    reports=[];timings=[]
    with torch.inference_mode():
        model(input_ids=ids[:4][None],use_cache=False)
        for wi in range(nwin):
            current=ids[wi*WINDOW:(wi+1)*WINDOW][None]
            started=time.perf_counter()
            output=model(input_ids=current,use_cache=False).logits[0].detach().cpu().numpy().astype(np.float32)
            timings.append(time.perf_counter()-started)
            if output.shape!=(WINDOW,49152) or not np.isfinite(output).all():
                raise AssertionError("nonfinite real full-model outputs")
            if args.kind=="teacher":
                logits_store[wi]=output.astype("<f2")
                continue
            reference_f64=torch.from_numpy(
                np.array(logits_store[wi,:-1],dtype=np.float64))
            student_f64=torch.from_numpy(output[:-1].astype(np.float64))
            teacher_logprobs=reference_f64.log_softmax(-1)
            student_logprobs=student_f64.log_softmax(-1)
            teacher_p=teacher_logprobs.exp()
            kl=float((teacher_p*(teacher_logprobs-student_logprobs)).sum(-1).mean())
            target=current[0,1:]
            nll=float(-student_logprobs[torch.arange(WINDOW-1),target].mean())
            agreement=float((reference_f64.argmax(-1)==student_f64.argmax(-1)).double().mean())
            if not all(math.isfinite(z) for z in (kl,nll,agreement)):
                raise AssertionError("broken model quality metric")
            reports.append({"window":wi,"teacher_to_candidate_KL":kl,
                            "heldout_next_token_NLL":nll,
                            "teacher_next_token_top1_agreement":agreement})
    if args.kind=="teacher":logits_store.flush()
    result={"source_kind":args.kind,"corpus":args.corpus,"origin":origin,
            "evaluation_windows":nwin,"tokens_per_window":WINDOW,
            "evaluated_next_token_count":nwin*(WINDOW-1),
            "source_integrity":audit,"teacher_logits_stored_fp16":True,
            "results_per_window":reports,
            "total_native_packed_context_prefill_seconds":sum(timings),
            "source_warmup_excluded":True}
    (folder/(args.corpus+"_"+args.kind+".json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"corpus":args.corpus,"kind":args.kind,
                      "KL":round(float(np.mean([r["teacher_to_candidate_KL"] for r in reports])),5)
                            if reports else None,
                      "NLL":round(float(np.mean([r["heldout_next_token_NLL"] for r in reports])),5)
                            if reports else None,
                      "physical_source_bytes":audit.get("full_source_bytes")}),flush=True)

def collect(args,corpus,variant):
    file=args.folder/"scores"/(corpus+"_"+variant+".json")
    data=json.loads(file.read_text())
    if data["corpus"]!=corpus or data["source_kind"]!=variant:raise AssertionError("wrong score file")
    return data

def select(args):
    teacher=collect(args,"train","teacher")
    data={name:collect(args,"train",name) for name in VARIANTS}
    sha=teacher["origin"]["token_SHA256"]
    if any(z["origin"]["token_SHA256"]!=sha for z in data.values()):
        raise AssertionError("TRAIN source differed between candidate whole models")
    measure={k:float(np.mean([r["teacher_to_candidate_KL"] for r in z["results_per_window"]]))
             for k,z in data.items()}
    nll={k:float(np.mean([r["heldout_next_token_NLL"] for r in z["results_per_window"]]))
         for k,z in data.items()}
    name=min(VARIANTS,key=lambda k:(measure[k],VARIANTS.index(k)))
    selected=name if measure[name]<=measure["baseline"]-MIN_TRAIN_GAIN else "baseline"
    frozen={"experiment":"CALIBRI_v74_frozen_TRAIN_joint_whole_model_KL_selection",
            "model":MODEL,"train_contexts":TRAIN_WINDOWS,"context":WINDOW,
            "TRAIN_positions":TRAIN_WINDOWS*(WINDOW-1),
            "TRAIN_source":teacher["origin"],"candidate_pool":list(VARIANTS),
            "candidate_mean_full_model_teacher_KL":measure,
            "candidate_mean_full_model_next_token_NLL":nll,
            "selection_threshold_nats":MIN_TRAIN_GAIN,
            "best_raw_joint_KL_choice":name,"chosen":selected,
            "TEST_never_accessed_for_candidate_selection":True}
    (args.folder/"v74_train_choice.json").write_text(json.dumps(frozen,indent=2)+"\n")
    print(json.dumps({"stage":"v74_TRUSTED_TRAIN_selection","chosen":selected,
                      "baseline_train_KL":round(measure["baseline"],5),
                      "best_train_KL":round(measure[name],5),
                      "best_unfiltered_variant":name}),flush=True)

def audit_test(args):
    frozen=json.loads((args.folder/"v74_train_choice.json").read_text())
    selected=frozen["chosen"]
    source=json.loads((args.folder/"v74_source_manifest.json").read_text())
    result={"experiment":"CALIBRI_v74_full_native_multimatrix_joint_KL_heldout_TEST",
            "model":MODEL,"unique_source_parameters":EXPECTED_UNIQUE,
            "physical_original_and_candidate_source_bytes":ORIGINAL_BYTES,
            "actual_all_in_source_bpw":8*ORIGINAL_BYTES/EXPECTED_UNIQUE,
            "frozen_TRAIN_only_winner":selected,"prior_TRAIN_choice":frozen,
            "native_5bit_whole_model_teacher_free":True,
            "no_TEST_selection_or_training":True,"domain_results":{}}
    rng=np.random.default_rng(7400)
    for corpus in ("wiki_test","news_test"):
        records={k:collect(args,corpus,k) for k in set(("teacher","baseline",selected))}
        if len({z["origin"]["token_SHA256"] for z in records.values()})!=1:
            raise AssertionError("unmatched independent actual original corpus tokens")
        for k in ("baseline",selected):
            audit=records[k]["source_integrity"]
            if audit["source_sha256"]!=source["candidates"][k]["source_SHA256"]:
                raise AssertionError("source checksum changed from frozen encoding")
            if not audit["native_C_direct_5bit"] or audit["trained_dense_2D_hot_parameters"]!=0:
                raise AssertionError("not native compressed inference")
        metrics={}
        if selected=="baseline":records["baseline"]=collect(args,corpus,"baseline")
        for key in ("teacher_to_candidate_KL","heldout_next_token_NLL",
                    "teacher_next_token_top1_agreement"):
            a=np.array([z[key] for z in records["baseline"]["results_per_window"]])
            b=np.array([z[key] for z in records[selected]["results_per_window"]])
            if len(a)!=TEST_WINDOWS or len(b)!=TEST_WINDOWS:raise AssertionError("too few TEST windows")
            delta=b-a
            ix=rng.integers(0,TEST_WINDOWS,size=(2000,TEST_WINDOWS))
            interval=np.quantile(delta[ix].mean(1),[.025,.975])
            metrics[key]={"original_baseline":float(a.mean()),
                          "TRAIN_winner":float(b.mean()),
                          "candidate_minus_baseline":float(delta.mean()),
                          "paired_8_window_bootstrap95":[float(interval[0]),float(interval[1])]}
        result["domain_results"][corpus]={"official_source":records["teacher"]["origin"],
                 "scored_next_token_positions":TEST_WINDOWS*(WINDOW-1),
                 "metrics":metrics,
                 "native_packed_forward_seconds":{
                     k:records[k]["total_native_packed_context_prefill_seconds"]
                     for k in ("baseline",selected)}}
    result["both_domains_KL_NLL_gain"]=all(
       result["domain_results"][name]["metrics"]["teacher_to_candidate_KL"]["candidate_minus_baseline"]<0 and
       result["domain_results"][name]["metrics"]["heldout_next_token_NLL"]["candidate_minus_baseline"]<0
       for name in ("wiki_test","news_test"))
    result["absolute_model_quality_still_NOT_proven"]=True
    result["sample_uncertainty_warning"]="Only 8 nonindependent text windows per domain; bootstrap descriptive."
    (args.folder/"v74_independent_test.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"stage":"v74_real_full_packed_independent_TEST",
                      "chosen":selected,"strict_cross_domain_gain":result["both_domains_KL_NLL_gain"],
                      "domains":{k:{
                          "KL_delta":round(v["metrics"]["teacher_to_candidate_KL"]["candidate_minus_baseline"],5),
                          "NLL_delta":round(v["metrics"]["heldout_next_token_NLL"]["candidate_minus_baseline"],5)}
                          for k,v in result["domain_results"].items()}}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",required=True,choices=("build","run","select","audit"))
    parser.add_argument("--original",type=Path)
    parser.add_argument("--gauge",type=Path)
    parser.add_argument("--covariance",type=Path)
    parser.add_argument("--kind",default="teacher")
    parser.add_argument("--corpus",choices=("train","wiki_test","news_test"),default="train")
    parser.add_argument("--folder",type=Path,default=Path("results"))
    args=parser.parse_args()
    args.folder.mkdir(parents=True,exist_ok=True)
    if args.phase=="build":
        if any(z is None for z in (args.original,args.gauge,args.covariance)):
            parser.error("build requires original, gauge and covariance real physical sources")
        build(args)
    elif args.phase=="run":
        if args.kind not in ("teacher",*VARIANTS):parser.error("invalid frozen source kind")
        run(args)
    elif args.phase=="select":select(args)
    else:audit_test(args)
if __name__=="__main__":main()
