"""Frozen, paired, diagnostic-only FP32 early-block module ablations.

Does not train, serialize original weights, or change the CB59 file. The FP32
oracle operators in the student are intentionally *large and uncompressed*.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v108.generation import build_chat
from v117.protocol import ORIGINAL_CB59_SHA, TOTAL_UNIQUE
from v128.protocol import read_v119
from v134.protocol import corpus
from v134.trace import capture, compare
from .oracles import ARMS, teacher_components, expected_physical_oracle_bytes
from .scoring import summary


def run(host, out):
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    packed=read_v119(host)
    if hashlib.sha256(packed.base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("CB59 source fingerprint does not match frozen model")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    samples=corpus()
    inputs=[]
    for name,side,question,opposite,gold,foil in samples:
        ids=tokenizer.encode(build_chat(tokenizer,question),add_special_tokens=False)
        if not 1<len(ids)<160:raise ValueError("invalid fixed prompt length")
        inputs.append(torch.tensor([ids],dtype=torch.long))

    teacher=original_teacher()
    if len(teacher.model.layers)!=30:raise ValueError("wrong FP32 teacher")
    with torch.inference_mode():
        references=[capture(teacher,x) for x in inputs]

    result={
        "stage":"v136_FIRST_BLOCK_TEACHER_ORACLE_LOCALIZATION",
        "model":MODEL,"revision":MODEL_SHA,
        "frozen_host_sha256":hashlib.sha256(host.read_bytes()).hexdigest(),
        "native_0_697bpp_host_bytes":host.stat().st_size,
        "native_host_bpp_only":8*host.stat().st_size/TOTAL_UNIQUE,
        "FP32_oracle_is_NOT_compressed":True,
        "FP32_oracle_is_NOT_deployable":True,
        "no_training":True,"no_reference_gold_entered_model":True,
        "no_locked_v89_TEST":True,"no_Jessica_or_LIVE_changes":True,
        "paired_cases":[{"pair":case[0],"side":case[1]} for case in samples],
        "arms":{},"borrowed_uncompressed_teacher_bytes":{},
    }

    with TemporaryDirectory(dir=out) as directory:
        root=Path(directory)
        base=root/"source.cb59"
        base.write_bytes(packed.base)
        model,metadata,_=make_model("baseline",SimpleNamespace(original=base,folder=root))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("original packed host unexpected")
        original_input=model.model.embed_tokens
        original_head=model.lm_head
        original_blocks=list(model.model.layers)
        if len(original_blocks)!=30:raise ValueError("wrong native model geometry")

        for arm in ARMS:
            result["borrowed_uncompressed_teacher_bytes"][arm]=expected_physical_oracle_bytes(
                teacher,arm)
            entries=[]
            with teacher_components(model,teacher,arm):
                for x,(ref_states,ref_logits) in zip(inputs,references):
                    actual,logits=capture(model,x)
                    entries.append(compare(ref_states,actual,ref_logits,logits,
                                           tokenizer.eos_token_id))
            if model.model.embed_tokens is not original_input or model.lm_head is not original_head:
                raise AssertionError("oracle input/head not restored")
            if any(block is not original for block,original in zip(model.model.layers,original_blocks)):
                raise AssertionError("oracle transformer block not restored")
            result["arms"][arm]=entries

    result["summary"]=summary(result["arms"])
    report=out/"v136_first_block_oracle.json"
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":result["stage"],
                      "summary":result["summary"]["by_arm"],
                      "physical_FP32_oracle_bytes":result["borrowed_uncompressed_teacher_bytes"],
                      "report":str(report)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("host",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.host,args.out)
