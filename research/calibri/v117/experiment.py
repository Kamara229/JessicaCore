"""Actual SmolLM2 135M: preserve original 0.667839bpp native CB59.

Pay rank-2 structural corrections for 60 attention/MLP roles. Initial
corrections are zero and quality is unchanged (still FAILED); no sham pass.
"""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from direct_codebook_v59 import CB59
from v94.generation import generate_one
from v108.generation import build_chat
from .archive import encode,Archive
from .runtime import attach,restore
from .protocol import ORIGINAL_CB59_SHA,SOURCE_BYTES,TOTAL_UNIQUE,MAX_SOURCE_BYTES

PROMPTS=(
    "What is the capital of Germany?",
    "What is 12 plus 19?",
    "Explain why trees need sunlight."
)

def audit(source_path,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    source=source_path.read_bytes()
    if len(source)!=SOURCE_BYTES or hashlib.sha256(source).hexdigest()!=ORIGINAL_CB59_SHA:
        raise AssertionError("frozen original real v59 missing")
    original=CB59(source)
    if (original.meta["original_unique_parameter_count"]!=TOTAL_UNIQUE
        or EXPECTED_UNIQUE!=TOTAL_UNIQUE):
        raise AssertionError("incorrect actual original distinct trained weights")
    blob=encode(source)
    paid=Archive(blob)
    info=paid.stats()
    (out/"v117_full_pinned_hybrid.cb117").write_bytes(blob)
    if len(blob)>MAX_SOURCE_BYTES:
        raise AssertionError("0.700bpp physically violated")
    folder=out/"native_runtime"
    folder.mkdir(exist_ok=True)
    model,native,_=make_model("baseline",SimpleNamespace(original=source_path,folder=folder))
    if not native["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("native packed model was replaced with dense weights")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    actual=[]
    with torch.inference_mode():
        for question in PROMPTS:
            prompt=build_chat(tok,question)
            old=generate_one(model,tok,prompt,stop_at_eos=True)
            attached=attach(model,paid)
            try:
                current=generate_one(model,tok,prompt,stop_at_eos=True)
                if old["ids"]!=current["ids"] or old["text"]!=current["text"]:
                    raise AssertionError("zero correction changed real user answer")
            finally:
                restore(model,attached)
            actual.append({"question":question,"original_CB59_answer":old["text"],
                           "zero_corrected_answer":current["text"],
                           "token_exact_identity":True,
                           "source_language_not_repaired":True})
    report={
        "experiment":"CALIBRI v117 CB59 old achievement preserved; structural adapters physically paid",
        "original_model":MODEL,"source_revision":MODEL_SHA,
        "original_archive_SHA256":ORIGINAL_CB59_SHA,
        "recovered_original_source_bytes":SOURCE_BYTES,
        "immutable_native_CB59_checked":True,
        "whole_actual_all_in_physical_archive":info,
        "not_int6_or_toy_checkpoint":True,
        "rank2_source_correctors_zero_initialized":True,
        "actual_native_model_role_count":60,
        "actual_native_model_start_to_finish_text_verified":len(actual),
        "original_135M_semantic_quality_NOT_FIXED":True,
        "next_step_requires_REAL_ground_truth_training_in_this_0_7_budget":True,
        "tests_only_real_frozen_generation_no_independent_v89_TEST":True,
        "no_prod_change":True,
        "answers":actual,
    }
    (out/"v117_paid_integration.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v117_paid_real",
        "source_bytes":len(blob),"bpp":info["all_in_bpp"],
        "original_CB59_SHA_locked":True,
        "zero_corrector_answer_identity":len(actual),
        "original_language_still_failed":True}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    audit(args.source,args.out)
