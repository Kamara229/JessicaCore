"""Compare original packed CB59, paid v141 two GEMVs, paid v143 fused GEMV.

All arms evaluate original CB140 physical artifact, same 16 locked
diagnostic prompts, exact byte equality. Native v143 fuses base and residual
calculation and changes NO model parameters, codebook or residual weights.
This experiment diagnoses speed and numerical equivalence, NOT LM quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
import numpy as np
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v132.protocol import all_examples
from v140.codec import unpack,report
from v141.kernel import install as install_v141
from v141.run import ORIGINAL_V140_SHA
from v131.probe import first_logits
from v108.generation import build_chat
from v94.generation import generate_one
from .runtime import install as install_v143

ARMS=("original_CB59","paid_v141_two_pass","paid_v143_fused")


@torch.inference_mode()
def infer(model,tok,example):
    pair,side,question,opposite,gold,foil=example
    logit,_,_=first_logits(model,tok,question)
    sample=generate_one(model,tok,build_chat(tok,question),stop_at_eos=True)
    return {
        "pair":pair,"side":side,"question":question,
        "logit_snapshot":np.asarray(logit,dtype=np.float32).copy(),
        "first_argmax":int(logit.argmax()),
        "full_generated_tokens":sample["ids"],
        "full_response":sample["text"],
        "seconds":sample["elapsed_seconds"],
        "tokens":len(sample["ids"]),
        "EOS":bool(sample["eos_generated"]),
    }


def run(packed_path,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    raw=packed_path.read_bytes()
    phys=unpack(raw)
    if phys.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("not the exact same previously frozen CB140 bytes")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    corpus=tuple(all_examples())
    if len(corpus)!=16:raise ValueError("corpus changed")
    with TemporaryDirectory(dir=out) as temp:
        root=Path(temp)
        base=root/"native.cb59"
        base.write_bytes(phys.original)
        model,meta,_=make_model("baseline",SimpleNamespace(original=base,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("unpacked source model accidentally used")
        base_emb,base_head=model.model.embed_tokens,model.lm_head
        install_v141(model,phys,root)
        v141_emb,v141_head=model.model.embed_tokens,model.lm_head
        model.model.embed_tokens=base_emb
        model.lm_head=base_head
        install_v143(model,phys,root)
        v143_emb,v143_head=model.model.embed_tokens,model.lm_head
        ops={
            "original_CB59":(base_emb,base_head),
            "paid_v141_two_pass":(v141_emb,v141_head),
            "paid_v143_fused":(v143_emb,v143_head),
        }
        records={k:[] for k in ARMS}
        for i,item in enumerate(corpus):
            order=ARMS[i%3:]+ARMS[:i%3]
            for arm in order:
                emb,head=ops[arm]
                model.model.embed_tokens=emb
                model.lm_head=head
                records[arm].append(infer(model,tokenizer,item))
        model.model.embed_tokens=base_emb
        model.lm_head=base_head

    timings={}
    for arm,rows in records.items():
        total=sum(r["seconds"] for r in rows)
        output_count=sum(r["tokens"] for r in rows)
        timings[arm]={
            "complete_generation_elapsed_seconds":total,
            "complete_generated_tokens":output_count,
            "full_model_tokens_per_second":output_count/max(total,1e-9),
            "end_of_turn_count":sum(r["EOS"] for r in rows),
        }
    numeric=[]
    for a,b in zip(records["paid_v141_two_pass"],records["paid_v143_fused"]):
        if (a["pair"],a["side"],a["question"])!=(b["pair"],b["side"],b["question"]):
            raise ValueError("loss of paired prompt matching")
        numeric.append({
            "pair":a["pair"],"side":a["side"],
            "max_abs_first_logit_diff":float(np.max(np.abs(
                a["logit_snapshot"]-b["logit_snapshot"]))),
            "same_first_argmax":a["first_argmax"]==b["first_argmax"],
            "identical_full_token_stream":a["full_generated_tokens"]==b["full_generated_tokens"],
        })
    out_data={
        "stage":"v143_FUSED_PACKED_CB140_CPU_RUNTIME",
        "physical_CB140_SHA256":phys.sha256,
        "physical_bpp":report(raw)["full_source_bpp"],
        "no_recompression_or_weight_fitting":True,
        "no_additional_physical_parameters":True,
        "no_independent_v89_TEST":True,
        "no_Green_or_Jessica_LIVE_changes":True,
        "all_arms_same_pinned_16_prompts":True,
        "timings":timings,
        "relative_speed_v143_over_CB59":
            timings["paid_v143_fused"]["full_model_tokens_per_second"]/
            timings["original_CB59"]["full_model_tokens_per_second"],
        "relative_speed_v143_over_v141":
            timings["paid_v143_fused"]["full_model_tokens_per_second"]/
            timings["paid_v141_two_pass"]["full_model_tokens_per_second"],
        "first_argmax_all16_match":all(r["same_first_argmax"] for r in numeric),
        "all16_generated_token_sequences_identical":all(r["identical_full_token_stream"] for r in numeric),
        "max_abs_logit_difference":max(r["max_abs_first_logit_diff"] for r in numeric),
        "pairwise_check":numeric,
        "generated_text_sample_v143":[
            {"pair":r["pair"],"side":r["side"],"text":r["full_response"]}
            for r in records["paid_v143_fused"][:4]],
        "no_semantic_improvement_claim":True,
    }
    path=out/"v143_fused_native_runtime.json"
    path.write_text(json.dumps(out_data,ensure_ascii=False,indent=2)+"\n")
    compact={k:v for k,v in out_data.items() if k not in ("pairwise_check","generated_text_sample_v143")}
    print(json.dumps(compact,ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("paid_cb140",type=Path)
    p.add_argument("out",type=Path)
    opts=p.parse_args()
    run(opts.paid_cb140,opts.out)
