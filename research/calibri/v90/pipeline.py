"""CALIBRI v90: original trained physical CB59 versus CB88 TRAIN-only diagnosis.
Does not tune against independent TEST, train new weights or test Jessica.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v88.physical import unpack
from v88.tensor import TensorPackedHead
from .provenance import verify_sources
from .qa import build_cases,score_cases
from .diagnostics import paired_diagnosis
from .generation import generate_diagnostic

def run(original_path:Path,cb88_path:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    original=original_path.read_bytes()
    cb88=cb88_path.read_bytes()
    frozen=verify_sources(original,cb88)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases,sources,answer_tokens=build_cases(tok)
    student,info,_=make_model("baseline",SimpleNamespace(original=original_path,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("dense source leaked into compressed original model")
    baseline=score_cases(student,cases,answer_tokens)
    baseline_greedy=generate_diagnostic(student,tok)
    _,A,B,bias=unpack(cb88)
    head=student.lm_head
    student.lm_head=TensorPackedHead(head,A,B,bias)
    try:
        corrected=score_cases(student,cases,answer_tokens)
        corrected_greedy=generate_diagnostic(student,tok)
    finally:
        student.lm_head=head
    report={"experiment":"CALIBRI_v90_frozen_4D_TRAIN_VALID_error_diagnosis",
            "model":MODEL,"pinned_model_revision":MODEL_SHA,
            "sources":frozen,"BOOLQ_TRAIN_source":sources,
            "paired_BOOLQ_diagnostics":{split:paired_diagnosis(baseline[split],corrected[split])
                                       for split in cases},
            "fresh_TRAIN_prompt_greedy": {"CB59":baseline_greedy,"CB88":corrected_greedy},
            "weights_unmodified":True,"official_independent_TEST_not_accessed":True,
            "no_scientific_PASS_claim":True,
            "Jessica_Render_Supabase_APK_main_not_touched":True}
    path=out/"v90_TRAIN_VALID_BoolQ_greedy_paired_diagnosis.json"
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"v90_TRAIN_VALID_diagnostics_finished",
        "QA": {s:{k:report["paired_BOOLQ_diagnostics"][s][k] for k in
                   ("count","CB59_correct","CB88_correct")}
               for s in cases},
        "greedy_distinct": {k:v["mean_distinct_ratio"] for k,v in
                            report["fresh_TRAIN_prompt_greedy"].items()},
        "output":str(path),"TEST_untouched":True},ensure_ascii=False),flush=True)
    return report
