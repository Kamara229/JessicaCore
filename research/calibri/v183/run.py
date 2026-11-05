"""Locate first-block V1 source corruption with *unpaid* exact-FP32 restores.

This is a diagnostic continuation of v182, NOT another compressed
physical candidate. The original full original-FP32 body remains resident
in source_model as an oracle, never in the native CALIBRI runtime.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.run import save
from v181.codec import CB181
from v182.diagnose import decode_handle, validate_teacher_source, distance
from . import protocol as p


def original_name(abbreviation):
    if abbreviation not in p.PROJECTIONS:
        raise ValueError("Non-registered 0th-block operator")
    return p.SCOPE+p.PROJECTIONS[abbreviation]


@torch.inference_mode()
def trace_first_block(model, token_ids):
    store=[]
    def recorder(_m,_inputs,output):
        hidden=output[0] if isinstance(output,(tuple,list)) else output
        if hidden.ndim!=3 or hidden.shape[0]!=1:
            raise AssertionError("Unrecognized first transformer block")
        store.append(hidden[0,-1].detach().cpu().numpy().copy())
    hook=model.model.layers[0].register_forward_hook(recorder)
    try:
        result=model(input_ids=token_ids,use_cache=False)
        logits=result.logits[0,-1].detach().cpu().numpy().copy()
    finally:
        hook.remove()
    if len(store)!=1 or not np.isfinite(store[0]).all() or not np.isfinite(logits).all():
        raise AssertionError("Invalid exact pretrained trace")
    return {"first_block":store[0],"last_logits":logits,
            "first_token":int(logits.argmax())}


def frozen_prompts(tokenizer):
    cases=[case for case in freeze_cases(all_cases())
           if case["question_id"] in p.FROZEN_CASE_IDS]
    if [x["question_id"] for x in cases]!=list(p.FROZEN_CASE_IDS):
        raise ValueError("Changed frozen public development questions")
    prompts=[]
    for case in cases:
        prompt=tokenizer.apply_chat_template(
            [{"role":"user","content":case["prompt"]}],
            tokenize=False,add_generation_prompt=True)
        ids=torch.tensor([tokenizer.encode(prompt,add_special_tokens=False)],
            dtype=torch.long)
        if not 1 < ids.shape[-1] < 256:
            raise ValueError("Changed original prompt geometry")
        prompts.append({"case_id":case["question_id"],"domain":case["domain"],
            "question":case["question"],"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
            "ids":ids})
    return prompts


@torch.no_grad()
def prepare_base(teacher,archive):
    source=teacher.state_dict()
    model=copy.deepcopy(teacher).eval()
    compact_state=model.state_dict()
    changed=0
    restored={}
    for name,desc in archive.meta["tensor_descriptors"].items():
        if desc["dtype"]!="CB181_K32V1":
            continue
        if name=="model.embed_tokens.weight":
            continue
        weight=decode_handle(archive.handles[name])
        compact_state[name].copy_(torch.from_numpy(weight))
        if name in {original_name(key) for key in p.PROJECTIONS}:
            restored[name]=torch.from_numpy(weight)
        changed+=1
    if changed!=210 or len(restored)!=7:
        raise AssertionError("V1 body not all 210 pretrained matrices")
    if model.model.embed_tokens.weight.data_ptr()!=model.lm_head.weight.data_ptr():
        raise AssertionError("Original FP32 tied source alias broken")
    if not torch.equal(model.model.embed_tokens.weight,
                       teacher.model.embed_tokens.weight):
        raise AssertionError("First block localization must retain original input and head")
    for key in restored:
        if torch.equal(compact_state[key],source[key]):
            raise AssertionError("First-block source already identical to original: "+key)
    return model,restored,{"changed_body_2D_tensors":changed,
        "tied_original_FP32_head_preserved":True,"diagnostic_not_physically_paid":True}


@torch.no_grad()
def apply_one_arm(scratch, source_state, decoded_first_block, arm):
    keys=p.OPERATOR_SET[arm]
    state=scratch.state_dict()
    for abbrev in keys:
        name=original_name(abbrev)
        state[name].copy_(source_state[name])
    return keys


@torch.no_grad()
def reset_arm(scratch, decoded_first_block, keys):
    state=scratch.state_dict()
    for abbrev in keys:
        name=original_name(abbrev)
        state[name].copy_(decoded_first_block[name])
        if not torch.equal(state[name],decoded_first_block[name]):
            raise AssertionError("Accidental mixture of previous unpaid intervention")
    return True


def evaluate_arm(scratch, teacher_cache, inputs):
    per_case=[]
    for prompt in inputs:
        result=trace_first_block(scratch,prompt["ids"])
        original=teacher_cache[prompt["case_id"]]
        per_case.append({"case_id":prompt["case_id"],"domain":prompt["domain"],
            "first_token":result["first_token"],
            "original_first_token":original["first_token"],
            "same_original_first_token":result["first_token"]==original["first_token"],
            "block0":distance(original["first_block"],result["first_block"]),
            "next_logits":distance(original["last_logits"],result["last_logits"])})
    return per_case


def summary(result):
    source=result["V1_BODY_ONLY"]
    base=float(np.mean([c["block0"]["relative_L2_to_reference"] for c in source]))
    if base<0.1:
        raise AssertionError("Baseline no longer agrees with v182 first-block failure")
    stats={}
    for key,rows in result.items():
        errs=[c["block0"]["relative_L2_to_reference"] for c in rows]
        logits=[c["next_logits"]["relative_L2_to_reference"] for c in rows]
        cur=float(np.mean(errs))
        stats[key]={
            "mean_block0_relative_L2":cur,
            "mean_next_logits_relative_L2":float(np.mean(logits)),
            "mean_firstblock_error_change_vs_v1_body":cur-base,
            "firstblock_error_reduction_fraction":(base-cur)/base,
            "first_token_matches_original":sum(c["same_original_first_token"] for c in rows),
            "public_development_cases":len(rows),
            "individual_block0_relative_errors":errs,
            "unpaid_full_FP32_substitution":key!="V1_BODY_ONLY"}
    # Verifies we actually have a causal intervention before trusting rankings.
    if max(stats["FP32_ALL_BLOCK0"]["individual_block0_relative_errors"])>2e-5:
        raise AssertionError("All original 7 first-block tensors did not recover original first block")
    return {"baseline_first_block_error":base,
        "ranked_firstblock_recovery":sorted(
            [(k,float(v["firstblock_error_reduction_fraction"]))
             for k,v in stats.items() if k!="V1_BODY_ONLY"],
            key=lambda x:x[1],reverse=True),
        "statistics":stats}


def run(source,out):
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    out.mkdir(parents=True,exist_ok=True)
    blob=source.read_bytes()
    if len(blob)!=p.MODEL_SOURCE_BYTES or hashlib.sha256(blob).hexdigest()!=p.MODEL_SOURCE_SHA256:
        raise ValueError("Unpinned original V1 source")
    archive=CB181(blob)
    if archive.meta.get("vector_length")!=1:
        raise ValueError("Wrong prepaid V1 source")
    protocol={"original_model":MODEL,"model_revision":MODEL_SHA,
        "sealed_source_bytes":p.MODEL_SOURCE_BYTES,
        "source_SHA256":p.MODEL_SOURCE_SHA256,
        "V1_source_physical_bpp_ABOVE_TARGET":p.V1_TOTAL_BPP,
        "public_probe_ids":list(p.FROZEN_CASE_IDS),
        "tested_arms":p.OPERATOR_SET,
        "all_FP32_restores_UNPAID_unreleasable_oracles":True,
        "not_0p7_candidate":True,
        "NO_private_TEST":True,"Jessica_no_tests":True}
    save(out/"protocol_before_any_inference.json",protocol)
    teacher=original_teacher()
    origin=validate_teacher_source(teacher,archive)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    frozen=frozen_prompts(tok)
    save(out/"source_and_public_prompt_manifest.json",{
        "source_integrity":origin,
        "prompts":[{k:v for k,v in item.items() if k!="ids"} for item in frozen]})
    baseline={row["case_id"]:trace_first_block(teacher,row["ids"]) for row in frozen}
    scratch,first_decoded,source_check=prepare_base(teacher,archive)
    data={}
    state_original=teacher.state_dict()
    for arm,abbrevs in p.OPERATOR_SET.items():
        selected=apply_one_arm(scratch,state_original,first_decoded,arm)
        try:
            data[arm]=evaluate_arm(scratch,baseline,frozen)
        finally:
            reset_arm(scratch,first_decoded,selected)
        print(json.dumps({"stage":"one_arm","arm":arm,
            "mean_block0_rel_l2":float(np.mean([x["block0"]["relative_L2_to_reference"]
                  for x in data[arm]])),
            "first_token_matches":sum(x["same_original_first_token"] for x in data[arm])}),flush=True)
        save(out/"v183_partial.json",{"completed_arms":list(data),
             "per_arm":data})
    scores=summary(data)
    out_data={"complete":True,"protocol":protocol,"source_check":source_check,
        "source_integrity":origin,"per_arm":data,"scores":scores,
        "NOT_a_physical_low_bit_candidate":True,
        "NO_Jessica_tests":True}
    save(out/"v183_complete.json",out_data)
    print(json.dumps({"stage":"complete","ranked":scores["ranked_firstblock_recovery"],
        "stats":{k:{"block_error":v["mean_block0_relative_L2"],
                  "first_token_matches":v["first_token_matches_original"]}
                 for k,v in scores["statistics"].items()}}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    opts=parser.parse_args()
    run(opts.source,opts.output)
