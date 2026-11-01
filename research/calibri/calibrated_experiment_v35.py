#!/usr/bin/env python3
"""Real Pythia-14m: matched-size 3.5bpw training-only output calibration.

True WikiText2 and TinyStories validation PPL measured with one physically
decoded 128x512 MLP matrix changed; both original and calibrated copies use
identical per-row INT3/INT4 bitwidths and identical actual serialized bytes.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch

from calibrated_rows_v35 import calibrated_encode,smoke
from direct_tiles_v33 import encode,decode
from fixedbit_v15 import encode as fixed_encode,decode as fixed_decode
from geometric_v18 import load_new_corpus
from quality import MODEL,DATASET,load_token_ids,ppl_eval,module_selection,calibrate


def run(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import HfApi
    if (os.getenv("ATEN_CPU_CAPABILITY")!="avx2"
        or os.getenv("DNNL_MAX_CPU_ISA")!="AVX2"
        or os.getenv("ONEDNN_MAX_CPU_ISA")!="AVX2"
        or str(torch._C._get_cpu_capability()).upper()!="AVX2"):
        raise AssertionError("Pinned ATen selected AVX2 / oneDNN requested AVX2 not satisfied")
    torch.set_num_threads(2)
    torch.manual_seed(34)  # Exactly v34 baseline random stream
    torch.use_deterministic_algorithms(True)
    smoke()
    api=HfApi()
    model_rev=api.model_info(MODEL).sha
    data_rev=api.dataset_info(DATASET).sha
    tiny_rev=api.dataset_info("roneneldan/TinyStories").sha
    tok=AutoTokenizer.from_pretrained(MODEL,revision=model_rev)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=model_rev,torch_dtype=torch.float32)
    model.eval()
    mods=module_selection(model)
    if len(mods)!=12:
        raise AssertionError("Pythia pretrained MLP count changed")
    name=sorted(mods)[0]
    mod=mods[name]
    w=mod.weight.detach().cpu().numpy().astype(np.float32).copy()
    if w.shape!=(128,512):
        raise AssertionError("source matrix shape changed")
    budget=int(w.size*3.5/8)
    if budget!=28672:raise AssertionError("physical budget mismatch")
    train=load_token_ids(tok,"train",data_rev,args.train_tokens)
    wiki=load_token_ids(tok,"validation",data_rev,args.valid_tokens)
    tiny,n_examples=load_new_corpus(tok,args.valid_tokens,tiny_rev)
    if len(wiki)!=len(tiny) or len(tiny)%args.context:
        raise AssertionError("corpora token budget differs")
    with torch.inference_mode():
        x=calibrate(model,{name:mod},train,args.context)[name].cpu().numpy().astype(np.float32)
    moment=(x*x).mean(axis=0,dtype=np.float32)
    baseline,meta=encode(w,moment,budget,br=1,bc=512)
    modified,calmeta=calibrated_encode(w,x,baseline)
    if len(baseline)!=len(modified) or len(modified)>budget:
        raise AssertionError("additional bytes from calibration")
    original_int4=fixed_encode(w,4)
    physical={
        "train_only_v34_rowwise_mixed3_4_control":baseline,
        "train_output_SSE_calibrated_same_row_bits":modified,
    }
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    matrices={}
    physical_stats={}
    for key,b in physical.items():
        path=dest.parent/(key+".bin")
        path.write_bytes(b)
        read=path.read_bytes()
        if len(read)!=len(b) or read!=b:
            raise AssertionError("physical binary roundtrip failed")
        matrices[key]=decode(read)
        if matrices[key].shape!=w.shape or not np.isfinite(matrices[key]).all():
            raise AssertionError("CRC decoder invalid")
        physical_stats[key]={
            "actual_bytes":path.stat().st_size,
            "actual_bits_per_weight_including_metadata":8*len(b)/w.size,
            "physical_SHA256":hashlib.sha256(read).hexdigest(),
        }
    if matrices["train_only_v34_rowwise_mixed3_4_control"].shape!=matrices["train_output_SSE_calibrated_same_row_bits"].shape:
        raise AssertionError("calibrated matrix geometry not identical")
    input_corps={
        "WikiText2_validation":wiki,
        "TinyStories_validation":tiny,
    }
    source=mod.weight.detach().clone()
    ppl={}
    with torch.inference_mode():
        try:
            for key,restored in (
                ("source_fp32_original",None),
                ("v34_unmodified_3p5bpw_rowwise",matrices["train_only_v34_rowwise_mixed3_4_control"]),
                ("v35_train_calibrated_3p5bpw_rowwise",matrices["train_output_SSE_calibrated_same_row_bits"]),
                ("INT4_OVER_BUDGET_high_rate_anchor",fixed_decode(original_int4)),
            ):
                mod.weight.copy_(source if restored is None else torch.from_numpy(restored))
                ppl[key]={corpus:ppl_eval(model,ids,args.context)
                          for corpus,ids in input_corps.items()}
        finally:
            mod.weight.copy_(source)
    if not all(np.isfinite(v["ppl"]) and v["ppl"]>0
               for per in ppl.values() for v in per.values()):
        raise AssertionError("bad PPL from actual pretrained model")
    report={
        "experiment":"CALIBRI_v35_same_bytes_train_output_SSE_calibrated_Pythia14m",
        "model":MODEL,"model_revision":model_rev,"matrix_name":name,
        "matrix_source_weight_sha256":hashlib.sha256(w.tobytes()).hexdigest(),
        "source_shape":list(w.shape),
        "datasets":{
            "WikiText2":{"name":DATASET,"revision":data_rev,"split":"validation"},
            "TinyStories":{"name":"roneneldan/TinyStories","revision":tiny_rev,"split":"validation"},
        },
        "token_stream_sha256":{
            "training_WikiText2":hashlib.sha256(train.numpy().tobytes()).hexdigest(),
            "validation_WikiText2":hashlib.sha256(wiki.numpy().tobytes()).hexdigest(),
            "validation_TinyStories":hashlib.sha256(tiny.numpy().tobytes()).hexdigest(),
        },
        "tokens":{
            "train_WikiText2":len(train),"validation_WikiText2":len(wiki),
            "validation_TinyStories":len(tiny),"TinyStories_examples":n_examples,
            "context":args.context,
        },
        "physical_budget_bytes":budget,
        "v34_baseline_bits_per_row":meta,
        "calibration_train_only":calmeta,
        "physical_binary":physical_stats,
        "higher_rate_INT4_bytes":len(original_int4),
        "higher_rate_INT4_bpw":8*len(original_int4)/w.size,
        "model_PPL_validation":ppl,
        "quality_comparison":{
            corpus:{
                "calibrated_minus_v34_baseline_PPL":
                    ppl["v35_train_calibrated_3p5bpw_rowwise"][corpus]["ppl"]-
                    ppl["v34_unmodified_3p5bpw_rowwise"][corpus]["ppl"],
                "calibrated_divided_by_v34_PPL":
                    ppl["v35_train_calibrated_3p5bpw_rowwise"][corpus]["ppl"]/
                    ppl["v34_unmodified_3p5bpw_rowwise"][corpus]["ppl"],
            } for corpus in input_corps
        },
        "cpu_control":{
            "selected_ATen_dispatch":str(torch._C._get_cpu_capability()),
            "requested_oneDNN_max_CPU_ISA":"AVX2",
            "torch_version":str(torch.__version__),
            "torch_threads":torch.get_num_threads(),
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        },
        "limits":[
            "Only ONE actual pretrained Pythia14m MLP projection is compressed; not whole model.",
            "Calibration uses WikiText TRAIN source activations, not PPL feedback.",
            "Both WikiText and TinyStories splits have appeared in CALIBRI exploratory research; neither is an untouched prospective blind benchmark.",
            "The full-model PPL uses fully decoded dense matrix from actual physical file, NOT a native packed compressed inference kernel.",
            "No QMoE 0.8bpw Switch result, no mobile/ARM speed, no newly invented neural parameter architecture.",
            "Train-output SSE is local single-layer approximation, and may not predict task PPL.",
            "Same physical size and rowwise bit allocation: comparison isolates scale/rounding calibration.",
        ],
    }
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "experiment":report["experiment"],
        "physical_bytes":{k:v["actual_bytes"] for k,v in physical_stats.items()},
        "row_bits_unmodified":meta["tiles_4bit"],
        "calibration_rows_changed":calmeta["rows_changed"],
        "train_output_SSE_ratio":calmeta["train_output_SSE_ratio"],
        "higher_rate_INT4_bytes":len(original_int4),
        "quality_PPL":{k:{c:round(v["ppl"],6) for c,v in d.items()} for k,d in ppl.items()},
        "PPL_calibrated_minus_original_3p5bpw":{
            k:round(v["calibrated_minus_v34_baseline_PPL"],6)
            for k,v in report["quality_comparison"].items()},
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/calibri_v35_calibration.json")
    p.add_argument("--train-tokens",type=int,default=256)
    p.add_argument("--valid-tokens",type=int,default=1024)
    p.add_argument("--context",type=int,default=128)
    args=p.parse_args()
    if args.train_tokens%args.context or args.valid_tokens%args.context:
        raise SystemExit("token limits must be whole contexts")
    run(args)
