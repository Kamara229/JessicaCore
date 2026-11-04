"""Causal native operator ablation in one physically prepaid original layer.

Uses PREVIOUSLY USED eight fixed prompts from v152 (no new validation).
Compares original CB59, seven original-operator one-bit corrections together,
seven single-operator-only and seven leave-one-out arms. 16 in total.
Every 2D pretrained weight is always served through original 5bit+paid 1bit
C kernels; original FP32 is only an evaluation teacher, never an operator
in any of the 16 compressed arms.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v108.generation import build_chat
from v152.scoring import comparison
from v153.codec import unpack,report
from v153.runtime import install_paid_block

from .protocol import ARMS,check_protocol,EXPECTED_DIAGNOSTIC_PROMPTS
from .arms import make_arms,subset
from .scoring import aggregate


@torch.inference_mode()
def evaluate_last(model,ids):
    logits=model(input_ids=ids,use_cache=False).logits[0,-1].detach().clone()
    if logits.shape!=(49152,) or not torch.isfinite(logits).all():
        raise ValueError("invalid native original-model last-token output")
    return logits


def run(paid_archive,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    physical=unpack(paid_archive.read_bytes())
    questions=check_protocol(physical)
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("source's original complete packed pretrained weights altered")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    prepared=[]
    for name,side,question,opposite,gold,foil in questions:
        prompt=build_chat(tok,question)
        tokens=tok.encode(prompt,add_special_tokens=False)
        if not 1<len(tokens)<160:
            raise ValueError("v152 locked diagnostic question altered")
        prepared.append({
            "name":name,"side":side,"question":question,
            "ids":torch.tensor([tokens],dtype=torch.long),
        })
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("wrong frozen FP32 original model")
    reference=[evaluate_last(teacher,case["ids"]) for case in prepared]
    del teacher
    arms=make_arms(physical)
    records={key:[] for key in ARMS}
    timings={key:0.0 for key in ARMS}
    eos=tok.eos_token_id
    if eos is None:raise ValueError("original EOS token missing")
    with TemporaryDirectory(dir=out) as directory:
        scratch=Path(directory)
        original=scratch/"native_original_CB59.cb59"
        original.write_bytes(physical.original)
        model,meta,_=make_model(
            "baseline",SimpleNamespace(original=original,folder=scratch))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("unexpected original FP32 pretrained matrices at runtime")
        prior=tuple(model.model.layers)
        embedding=model.model.embed_tokens
        head=model.lm_head
        if embedding.handle is not head.handle:
            raise ValueError("original physically tied output/input weights forked")
        with torch.inference_mode():
            for arm,keys in arms.items():
                if keys:
                    intervention=install_paid_block(model,subset(physical,keys),scratch)
                else:
                    from contextlib import nullcontext
                    intervention=nullcontext({"original_CB59_no_paid_ops":True})
                with intervention:
                    for n,case in enumerate(prepared):
                        start=perf_counter()
                        logits=evaluate_last(model,case["ids"])
                        timings[arm]+=perf_counter()-start
                        event=comparison(reference[n],logits,eos)
                        event["diagnostic_case_index"]=n
                        event["question"]=case["question"]
                        records[arm].append(event)
                if (any(x is not y for x,y in zip(model.model.layers,prior))
                        or model.model.embed_tokens is not embedding
                        or model.lm_head is not head):
                    raise AssertionError("source operator ablation leaked into later arm")
                if any(p.ndim>=2 for p in model.parameters()):
                    raise ValueError("unpaid pretrained FP32 layer after ablation")
    final=aggregate(records,timings)
    report_data={
        "stage":"v154_ORIGINAL_SOURCE_LAYER6_SEVEN_OPERATOR_ABLATIONS",
        "fully_serialized_CB153_SHA256":physical.sha256,
        "complete_original_CB59_SHA256":ORIGINAL_CB59_SHA,
        "physical_source_rate":report(paid_archive.read_bytes()),
        "analysis_ONLY_on_preexisting_v152_diagnostic_prompts":True,
        "fixed_case_count":EXPECTED_DIAGNOSTIC_PROMPTS,
        "cases":[{"name":p["name"],"side":p["side"],"question":p["question"]}
                 for p in prepared],
        "physical_entire_source_bytes_unchanged":True,
        "no_new_trained_parameters_or_source_archive":True,
        "no_original_dense_source_resident_in_any_compressed_arm":True,
        "true_F32_teacher_ONLY_for_evaluation":True,
        "single_original_source_operator_7x_and_leave_one_out_7x":True,
        "no_independent_v89_TEST_access":True,
        "no_new_validation_and_no_quality_release_claim":True,
        "no_Jessica_LIVE_Render_Supabase_APK_changes":True,
        "all_16_raw_first_token_results":records,
        "diagnostic":final,
        "next_step_requires_new_disjoint_domain_VALID_after_any_action":True,
    }
    path=out/"v154_seven_source_ops_16arms_existing_questions.json"
    path.write_text(json.dumps(report_data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report_data["stage"],
        "full_source_SHA256":physical.sha256,
        "baseline_KL":final["original_CB59_mean_teacher_KL"],
        "all7_KL":final["complete_CB153_mean_teacher_KL"],
        "all7_vs_base":final["all7_KL_delta_vs_CB59"],
        "all7_single_operator_ranking":final["per_original_operator_diagnostic"],
        "not_an_independent_VALID_or_TEST":True,
        "report":str(path)
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("pinned_original_CB153",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.pinned_original_CB153,a.out)
