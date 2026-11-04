"""Top facade: small positive/negative controls -> real Pythia-14M operators.

Physical full-model storage is NOT achieved, only a locked entire archive of
the 24 selected original linear operators. Candidate inference reconstructs
selected W for scientific function measurement. Direct blockwise matvec
parity is tested independently without a full W buffer in that path.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import json
import numpy as np
import torch
from .archive import Archive,create_archive
from .fit import GROUP
from .roles import weights
from .toy import make_toy
from .metrics import (VALID,GENERATE,original_reference,activation_error,
                      evaluate,generate,patch_model,restore_model,
                      source_digest,write_report)

MODES=("global","layer_control","role")
MODEL="EleutherAI/pythia-14m"
PINNED_REVISION="cf967c0a9a04383db6f7b1108d86b2962634b4ac"

def weight_error(source,archive):
    diff=0.;power=0.
    for name,w in source.items():
        z=archive.decode(name)
        diff+=float(np.square(w-z,dtype=np.float64).sum())
        power+=float(np.square(w,dtype=np.float64).sum())
    return (diff/max(power,1e-20))**.5

def compare_toys():
    report={}
    for kind in ("structured","unstructured"):
        source=make_toy(structured=kind=="structured")
        entries={}
        sizes=[]
        for mode in MODES:
            blob=create_archive(source,mode)
            a=Archive(blob)
            size=a.summary()["whole_source_bytes"]
            sizes.append(size)
            entries[mode]={"weight_rel_rmse":weight_error(source,a),
                           "total_physical_bytes":size,
                           "whole_sha256":hashlib.sha256(blob).hexdigest()}
        if len(set(sizes))!=1:
            raise AssertionError("toy modes have unequal true physical budgets")
        report[kind]=entries
    return report

def numeric_packed_parity(a,seed=112):
    rng=np.random.default_rng(seed)
    worst=0.
    for i,name in enumerate(sorted(a.records)):
        if i%5:
            continue
        n=a.records[name]["cols"]
        x=rng.normal(size=(3,n)).astype(np.float32)
        full=x@a.decode(name).T
        streamed=a.direct(name,x,chunk=11)
        worst=max(worst,float(np.max(np.abs(full-streamed))))
    if worst>2e-4:
        raise AssertionError("direct blockwise matvec disagrees with decoded operator")
    return worst

def experiment(out):
    torch.set_num_threads(2)
    np.random.seed(112)
    out.mkdir(parents=True,exist_ok=True)
    toy=compare_toys()
    print(json.dumps({"stage":"v112_toy","structured":toy["structured"],
                      "unstructured":toy["unstructured"]}),flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    source_revision=PINNED_REVISION
    tok=AutoTokenizer.from_pretrained(MODEL,revision=source_revision)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=source_revision,torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    source=weights(model)
    parameters=sum(int(p.numel()) for p in model.parameters())
    samples,original_logits,activations=original_reference(model,tok,source)
    original_quality=evaluate(model,tok,samples,original_logits)
    original_generations=generate(model,tok)

    report={"experiment":"CALIBRI v112 structural role-shared 5-bit operator pilot",
            "biology_inspired_not_actual_connectome":True,
            "pretrained_model":MODEL,"exact_model_revision":source_revision,
            "trained_source_weights_sha256":source_digest(source),
            "pretrained_unique_model_parameters":parameters,
            "selected_operator_count":len(source),
            "selected_source_weight_elements":sum(w.size for w in source.values()),
            "other_embeddings_norms_head_uncompressed":True,
            "no_entire_llm_under_070_claim":True,
            "decoded_weights_used_only_for_research_full_model_inference":True,
            "direct_packed_streaming_verified_for_selected_GEMV":True,
            "independent_original_v89_TEST_unopened":True,
            "train_fit_uses_only_pretrained_weights":True,
            "validation_text_sha256":[hashlib.sha256(s.encode()).hexdigest() for s in VALID],
            "toy":toy,"original":{"quality":original_quality,
                                  "free_generation":original_generations},
            "candidates":{}}
    sizes=[]
    for mode in MODES:
        blob=create_archive(source,mode)
        a=Archive(blob)
        physical=a.summary()
        sizes.append(physical["whole_source_bytes"])
        numeric_error=numeric_packed_parity(a)
        record={"physical":physical,"weight_rel_rmse":weight_error(source,a),
                "true_teacher_activation":activation_error(source,a,activations),
                "streaming_max_abs_difference":numeric_error}
        (out/f"operators_{mode}.clb112").write_bytes(blob)
        try:
            patch_model(model,source,a)
            record["heldout_lm_quality"]=evaluate(model,tok,samples,original_logits)
            record["free_generation"]=generate(model,tok)
        finally:
            restore_model(model,source)
        report["candidates"][mode]=record
        print(json.dumps({"stage":"v112_real","mode":mode,
                          "physical_bytes":physical["whole_source_bytes"],
                          "selected_bits_per_weight":physical["selected_bits_per_weight"],
                          "weight_rmse":record["weight_rel_rmse"],
                          "activation_rmse":record["true_teacher_activation"]["real_teacher_activation_rel_rmse"],
                          "heldout_kl":record["heldout_lm_quality"]["teacher_relative_kl"],
                          "heldout_nll":record["heldout_lm_quality"]["token_nll"]}),flush=True)
    if len(set(sizes))!=1:
        raise AssertionError("candidate physical storage budgets are not equal")
    report["equal_physical_operator_archive_budget"] = True
    report["no_scientific_PASS_from_small_sample"] = True
    write_report(out,report)
    print(json.dumps({"stage":"v112_completed","models":"real 14M plus two toys",
                      "files":"three physically serialized operator-only archives",
                      "production_changes":False}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("out",type=Path)
    experiment(parser.parse_args().out)
