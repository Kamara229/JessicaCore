#!/usr/bin/env python3
"""S5 unpaid diagnostic FP32 layer sensitivity: do NOT ship hybrid as model.

Causal interventions reveal where the trained 135M original suffers largest
source→CB59 damage, before designing another physical mixed-precision codec.
Only already-inspected TRAIN-debug prompts. Never EVAL/QA.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from direct_codebook_v59 import CB59
from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from v176.operators import clear
from v196.recover import build_trainable_from_paid_source
from calibri2.independent_135m.s2.rootcause import measure, prompt_ids

S0_SHA = "50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
PROMPTS=(
    "Answer with one digit: How much is 2 plus 2?",
    "Answer with one word: What is the opposite of hot?",
    "Complete briefly: The sun rises in the ...",
)
DEPTH=30
WIDTH=576


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,
         indent=2,sort_keys=True)+"\n",encoding="utf-8")


def emit(stage,**values):
    print(json.dumps({"stage":stage,**values},ensure_ascii=False),flush=True)


@contextmanager
def restore_fp32_original_selected(student,teacher,layers,submodule=None):
    """Always return student to byte-identical original source module topology."""
    if submodule not in (None,"self_attn","mlp"):
        raise ValueError("Predeclared intervention only")
    old_embed=student.model.embed_tokens
    old_head=student.lm_head
    positions=sorted(layers)
    originals=[]
    try:
        student.model.embed_tokens=teacher.model.embed_tokens
        student.lm_head=teacher.lm_head
        for i in positions:
            if submodule is None:
                original=student.model.layers[i]
                student.model.layers[i]=teacher.model.layers[i]
            else:
                original=getattr(student.model.layers[i],submodule)
                setattr(student.model.layers[i],submodule,
                        getattr(teacher.model.layers[i],submodule))
            originals.append((i,original))
        yield
    finally:
        for i,mod in originals:
            if submodule is None:
                student.model.layers[i]=mod
            else:
                setattr(student.model.layers[i],submodule,mod)
        student.model.embed_tokens=old_embed
        student.lm_head=old_head


@torch.inference_mode()
def first_token_stats(teacher,student,sources,tok,contexts):
    ret=[]
    for ctx in contexts:
        a=teacher(input_ids=ctx,use_cache=False).logits[0,-1].float()
        clear(sources)
        b=student(input_ids=ctx,use_cache=False).logits[0,-1].float()
        ta=int(a.argmax())
        tb=int(b.argmax())
        rank=int((b>b[ta]).sum())+1
        ret.append({
          "original_FP32_first_id":ta,
          "original_FP32_first_token":tok.decode([ta]),
          "hybrid_first_id":tb,
          "hybrid_first_token":tok.decode([tb]),
          "teacher_rank_in_hybrid":rank,
          "same_FP32_first_token":ta==tb,
          "hybrid_top5":[tok.decode([int(x)]) for x in b.topk(5).indices],
        })
        clear(sources)
    return ret


def param_count(module):
    # Original tied FP32 embed/head counted ONCE in theoretical hybrid.
    return sum(int(p.numel()) for p in module.parameters())


def run(source,out):
    torch.manual_seed(13551117)
    torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=S0_SHA:
        raise AssertionError("Wrong original pinned 135M CB59 provenance")
    metadata={
      "stage":"S5 trained original FP32 135M hybrid decoder sensitivity",
      "S0_sha":S0_SHA,
      "original_trained_FP32_revision":MODEL_SHA,
      "original_unique_params":EXPECTED_UNIQUE,
      "diagnostic_only_unpaid_no_0p7_bpp_model_export":True,
      "old_debug_TRAIN_prompts":list(PROMPTS),
      "full_original_embedding_and_head_restored_in_each_hybrid":True,
      "prefix_counts":[0,5,10,15,20,25,30],
      "suffix_counts":[5,10,15,20,25,30],
      "all_attention_only_control":True,
      "all_MLP_only_control":True,
      "never_load_NRA_and_S3_adapters":True,
      "no_heldout_EVAL_or_QA_access":True,
    }
    dump(out/"S5_PRECOMMITTED_ONLY_DEBUG_BEFORE_MODEL_LOAD.json",metadata)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    contexts=[prompt_ids(tok,q) for q in PROMPTS]
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).cpu().eval()
    if (sum(p.numel() for p in teacher.parameters())!=EXPECTED_UNIQUE or
        teacher.config.num_hidden_layers!=DEPTH or
        teacher.config.hidden_size!=WIDTH):
        raise AssertionError("Original trained 135M geometry mismatch")
    student,sources=build_trainable_from_paid_source(CB59(raw))
    student.cpu().eval().requires_grad_(False)
    teacher_answers=[measure(teacher,None,tok,ctx) for ctx in contexts]
    base_answers=[measure(student,sources,tok,ctx) for ctx in contexts]
    cases={}
    plan=[("fp32_input_head_only",[],None)]
    for n in (5,10,15,20,25,30):
        plan.append((f"FP32_first_{n}_full_decoder",[i for i in range(n)],None))
        plan.append((f"FP32_last_{n}_full_decoder",
                     [i for i in range(DEPTH-n,DEPTH)],None))
    plan.append(("FP32_all_30_attention_only",list(range(DEPTH)),"self_attn"))
    plan.append(("FP32_all_30_MLP_only",list(range(DEPTH)),"mlp"))
    # Storage estimates count the original teacher input embedding/head only
    # once; extra FP32 block weights are intentionally NOT physically payable.
    input_fp32_bytes=teacher.model.embed_tokens.weight.numel()*4
    for key,indices,submodule in plan:
        with restore_fp32_original_selected(student,teacher,indices,submodule):
            stat=first_token_stats(teacher,student,sources,tok,contexts)
            output=None
            if key in ("fp32_input_head_only","FP32_first_10_full_decoder",
                       "FP32_last_10_full_decoder",
                       "FP32_first_20_full_decoder",
                       "FP32_last_20_full_decoder",
                       "FP32_first_30_full_decoder",
                       "FP32_all_30_attention_only",
                       "FP32_all_30_MLP_only"):
                output=[measure(student,sources,tok,x) for x in contexts]
        changed_weights=(sum(param_count(
                         teacher.model.layers[i] if submodule is None else
                         getattr(teacher.model.layers[i],submodule))
                         for i in indices))
        case={
          "mode":key,"which_original_decoder_layers_restored":indices,
          "only_sublayer":submodule,
          "teacher_full_FP32_input_and_output_embedding_bytes":
               input_fp32_bytes,
          "additional_original_FP32_decoder_weight_bytes":4*changed_weights,
          "unpaid_extra_BYTES_not_in_original_CB59":input_fp32_bytes+4*changed_weights,
          "may_NOT_be_shipped_inside_0p7bpp":True,
          "same_original_FP32_first_count":sum(int(r["same_FP32_first_token"])
                                             for r in stat),
          "first_token_accuracy_debug_count":len(PROMPTS),
          "raw_first_token_results":stat,
          "full_greedy_results_if_predeclared":output,
        }
        cases[key]=case
        dump(out/"S5_INCREMENTAL_UNPAID_CAUSAL_DEBUG_RESULTS.json",{
            "reference_original_FP32":teacher_answers,
            "reference_original_S0_CB59":base_answers,
            "causal_interventions":cases,
        })
        emit("S5_FP32_original_layer_causal_probe",mode=key,
             FP32_first_same=case["same_original_FP32_first_count"],
             n=len(PROMPTS),
             restored_original_layer_count=len(indices),
             sublayer=submodule,
             memory_bytes_unpaid=case["unpaid_extra_BYTES_not_in_original_CB59"],
             example=(output[0]["text"][:75] if output else None))
    control=cases["FP32_first_30_full_decoder"][
        "full_greedy_results_if_predeclared"]
    if [x["tokens"] for x in control]!=[x["tokens"] for x in teacher_answers]:
        raise AssertionError(
             "Restoring all teacher 30 layers + both head/embed MUST give exact FP32")
    if any(x["same_original_FP32_first_count"]!=len(PROMPTS) for
           k,x in cases.items() if k=="FP32_first_30_full_decoder"):
        raise AssertionError("Full FP32 hybrid did not preserve trained FP32 first tokens")
    # Restoration must not mutate actual original source comparison topology.
    restored_back=[measure(student,sources,tok,x) for x in contexts]
    if restored_back!=base_answers:
        # Ignore wall-clock elapsed; actual full tokens+EOS must be identical.
        if any(x["tokens"]!=y["tokens"] or x["EOS_seen"]!=y["EOS_seen"]
               for x,y in zip(restored_back,base_answers)):
            raise AssertionError("Temporary FP32 diagnostic hybrid changed source after restore")
    report={
        **metadata,
        "completed":True,
        "original_FP32_actual_8token_answers":teacher_answers,
        "actual_original_CB59_8token_answers":base_answers,
        "all_causal_interventions":cases,
        "full_FP32_restoration_positive_control_exact":True,
        "CB59_original_model_restored_exact_after_all_diagnostics":True,
        "new_physically_valid_under_0p7_model_created":False,
        "independent_QA_restoration_proven":False,
        "next_step":"Use critical-layer results to design source-aware mixed rate and train from ORIGINAL FP32 without exceeding 0.7 bpp",
    }
    dump(out/"S5_FINAL_30LAYER_ORIGINAL_FP32_CAUSAL_ERROR_AUDIT.json",report)
    emit("S5_FINAL_REAL_135M_original_layer_causality",
         tested=len(cases),
         best_early_10=cases["FP32_first_10_full_decoder"]["same_original_FP32_first_count"],
         best_late_10=cases["FP32_last_10_full_decoder"]["same_original_FP32_first_count"],
         full_30_control_exact=True,
         no_physically_valid_FP32_hybrid_produced=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("S0_original_CB59",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.S0_original_CB59,a.out)
